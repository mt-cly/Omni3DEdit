
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
import wandb
from matplotlib import pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from natsort import natsorted
from omegaconf import OmegaConf
import time
from packaging import version
from PIL import Image
from argparse import ArgumentParser
import random
# 生成plucker坐标
from seva.geometry import get_plucker_coordinates
import json
import os
from seva.data_io import get_parser
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer import Trainer
import string
from seva.modules.preprocessor import VGGTPipeline
from tqdm.auto import tqdm
from torch.utils.data.distributed import DistributedSampler
from pytorch_lightning.utilities import rank_zero_only
from seva.eval import (
    IS_TORCH_NIGHTLY,
    compute_relative_inds,
    create_transforms_simple,
    infer_prior_inds,
    infer_prior_stats,
    run_one_scene,
    get_value_dict,
    chunk_input_and_test,
    pad_indices,
    assemble,
    load_img_and_K,
    transform_img_and_K,
    get_k_from_dict
    
)
from seva.geometry import (
    generate_interpolated_path,
    generate_spiral_path,
    get_arc_horizontal_w2cs,
    get_default_intrinsics,
    get_lookat,
    get_preset_pose_fov,
    
)


class SEVAEDITDATASET(torch.utils.data.Dataset):
    def __init__(self, rank_id, json_folder, VERSION_DICT, datasets=None, editing_types=None, balanced_sampling=False):
        self.json_folder = json_folder
        self.VERSION_DICT = VERSION_DICT   
        self.datasets = datasets if datasets is not None else None  # tuple of allowed datasets
        self.editing_types = editing_types if editing_types is not None else None  # tuple of allowed editing types
        self.balanced_sampling = balanced_sampling  # 是否使用数据集间均衡采样
        
        # 加载数据集
        json_files_list = [f for f in os.listdir(json_folder) if f.endswith('.json')]
        
        # 过滤JSON文件，只保留符合指定数据集和编辑类型的文件
        filtered_json_files = []
        for json_file in json_files_list:
            if self._is_valid_json_file(json_file):
                filtered_json_files.append(json_file)
        
        self.json_files = [os.path.join(self.json_folder, json_file) for json_file in filtered_json_files]
        
        # 如果启用均衡采样，按数据集分组
        if self.balanced_sampling:
            self.dataset_groups = {}
            for json_file in self.json_files:
                filename = os.path.basename(json_file).replace('.json', '')
                dataset = filename.split('_')[0]
                if dataset not in self.dataset_groups:
                    self.dataset_groups[dataset] = []
                self.dataset_groups[dataset].append(json_file)
            
            print(f"Balanced sampling enabled. Dataset distribution:")
            for dataset, files in self.dataset_groups.items():
                print(f"  {dataset}: {len(files)} files")
        
        print(f"Total JSON files found: {len(json_files_list)}")
        print(f"Filtered JSON files: {len(filtered_json_files)}")
        if self.datasets is not None:
            print(f"Allowed datasets: {self.datasets}")
        if self.editing_types is not None:
            print(f"Allowed editing types: {self.editing_types}")
        
        # 初始化VGGT pipeline
        self.vggt = VGGTPipeline(device=f'cuda:{rank_id}')
    
    def _is_valid_json_file(self, json_file):
        """
        检查JSON文件名是否符合指定的数据集和编辑类型要求
        文件名格式应为: {dataset}_{editing_type}_*.json
        
        Args:
            json_file: JSON文件名
            
        Returns:
            bool: 是否符合要求
        """
        # 如果没有指定过滤条件，则保留所有文件
        if self.datasets is None and self.editing_types is None:
            return True
            
        # 去掉.json后缀
        filename_without_ext = json_file.replace('.json', '')
        
        # 按下划线分割文件名
        parts = filename_without_ext.split('_')
        
        if len(parts) < 2:
            # 如果文件名格式不正确，默认不包含
            return False
            
        dataset = parts[0]
        editing_type = parts[1]
        
        # 检查数据集是否在允许的范围内
        if self.datasets is not None and dataset not in self.datasets:
            return False
            
        # 检查编辑类型是否在允许的范围内
        if self.editing_types is not None and editing_type not in self.editing_types:
            return False
            
        return True
    
    def __len__(self):
        return 10000000 # len(self.json_files)
    
    def imgs_2_c2ws_ks(self, img_paths):
        """
        接收图像路径列表，返回相机参数
        Args:
            img_paths: list of image paths
        Returns:
            ks: [f, 3, 3] 未归一化的内参矩阵
            c2ws: [f, 4, 4] 相机到世界坐标系的变换矩阵
        """
        _, vggt_Ks, vggt_c2ws = self.vggt.infer_cameras_and_points(img_paths, only_camera=True)
        return vggt_Ks, vggt_c2ws
    
    def __getitem__(self, idx):
        # 选择JSON文件
        if self.balanced_sampling and hasattr(self, 'dataset_groups'):
            # 均衡采样：先随机选择数据集，再从该数据集中随机选择文件
            dataset_names = list(self.dataset_groups.keys())
            selected_dataset = random.choice(dataset_names)
            json_file = random.choice(self.dataset_groups[selected_dataset])
        else:
            # 默认采样：基于文件数量的均匀采样
            idx = idx % len(self.json_files)
            json_file = self.json_files[idx]
        
        # 读取JSON文件
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 从文件名中提取editing_type
        filename = os.path.basename(json_file).replace('.json', '')
        parts = filename.split('_')
        editing_type = parts[1] if len(parts) >= 2 else "unknown"
        
        num_total_views = self.VERSION_DICT['T']
        h, w = self.VERSION_DICT['H'], self.VERSION_DICT['W']
        
        # 计算各类视图的数量
        # 假设 num_total_views = 5: source=2, cond=1, target=2
        # 假设 num_total_views = 7: source=3, cond=1, target=3
        num_cond_views = 1
        num_source_views = num_target_views = (num_total_views - num_cond_views) // 2
        
        # 确保总数正确
        assert num_source_views + num_cond_views + num_target_views == num_total_views
        
        scene_name = data['scene_name']
        # 从image_pairs中选择足够的pairs
        image_pairs = data['image_pairs']
        # 只选择成功的pairs
        successful_pairs = [pair for pair in image_pairs if pair['success']]
        
        # 需要的pairs数量应该足够覆盖source_views和target_views
        required_pairs = max(num_source_views, num_target_views)
        
        if len(successful_pairs) < required_pairs:
            # 如果成功的pairs不够，使用有放回采样
            selected_pairs = random.choices(successful_pairs, k=required_pairs)
        else:
            # 随机选择足够数量的pairs
            selected_pairs = random.sample(successful_pairs, required_pairs)
        
        # 分别提取路径（将相对路径转为绝对路径，相对于 json_folder）
        all_original_paths = [os.path.normpath(os.path.join(self.json_folder, pair['original_path'])) for pair in selected_pairs]  # 原始图像
        all_edited_paths = [os.path.normpath(os.path.join(self.json_folder, pair['edited_path'])) for pair in selected_pairs]      # 编辑后图像

        # 选择source_views（编辑后的图像）
        source_view_paths = all_original_paths[:num_source_views]
        
        # 选择target_views（原始图像）
        target_view_paths = all_edited_paths[:num_target_views]
        
        # 选择cond_views（从target_views的前几个中选择）
        cond_view_paths = target_view_paths[:num_cond_views]
        
        # 根据编辑类型选择VGGT使用的路径
        if editing_type == "add":
            # 对于add类型的编辑，使用edited_path（编辑后的图像）提供给VGGT
            path_for_vggt = target_view_paths + cond_view_paths + target_view_paths
            # path_for_vggt = source_view_paths + cond_view_paths + target_view_paths
        elif editing_type in ["remove", "color"]:
            # 对于remove和color_change类型的编辑，使用source_path（原始图像）提供给VGGT
            path_for_vggt = source_view_paths + source_view_paths[:1] + source_view_paths
            # path_for_vggt = source_view_paths + cond_view_paths + target_view_paths
        else:
            # 默认情况使用all_paths
            assert False , "unknown editing"
        
        # 按照 [source + cond + target] 的顺序组织所有路径
        all_paths = source_view_paths + cond_view_paths + target_view_paths

        # 使用VGGT获取相机参数
        all_ks, all_c2ws = self.imgs_2_c2ws_ks(path_for_vggt)
        all_ks = torch.tensor(all_ks).contiguous()
        all_c2ws = torch.tensor(all_c2ws).contiguous()

        # all_ks, all_c2ws = self.imgs_2_c2ws_ks(path_for_vggt)
        # all_ks = np.concatenate([all_ks, all_ks[:1], all_ks],axis=0)
        # all_c2ws = np.concatenate([all_c2ws, all_c2ws[:1], all_c2ws],axis=0)
        # all_ks = torch.tensor(all_ks).contiguous()
        # all_c2ws = torch.tensor(all_c2ws).contiguous()

        # 加载和处理图像
        imgs = []
        imgs_clip = []
         
        for i, img_path in enumerate(all_paths):
            # 加载图像并调整大小
            img, K = load_img_and_K(img_path, None, K=all_ks[i], device="cpu")
            img, K = transform_img_and_K(img, (w, h), K=K[None])
            assert K is not None
            K = K[0]
            K[0] /= w
            K[1] /= h
            all_ks[i] = K
            img_clip = img
            imgs_clip.append(img_clip)
            imgs.append(img)
        
        imgs_clip = torch.cat(imgs_clip, dim=0)
        imgs = torch.cat(imgs, dim=0)
        
        # 定义哪些是input frames (source_views + cond_views)
        curr_input_frame_indices = list(range(num_source_views + num_cond_views))
        curr_input_camera_indices = list(range(num_source_views + num_cond_views))
        
        # 使用get_value_dict函数获取标准化的value_dict
        value_dict = get_value_dict(
            curr_imgs=imgs,
            curr_imgs_clip=imgs_clip,
            curr_input_frame_indices=curr_input_frame_indices,
            curr_c2ws=all_c2ws,
            curr_Ks=all_ks,
            curr_input_camera_indices=curr_input_camera_indices,
            all_c2ws=all_c2ws,  # 使用相同的相机参数作为参考
            camera_scale=self.VERSION_DICT["options"].get("camera_scale", 2.0),
        )
        
        # 添加source_view_mask: 只有source视图为1.0，其余为0.0
        source_view_mask = torch.tensor(
            [1.0] * num_source_views +     # source views
            [0.0] * num_cond_views +       # cond views
            [0.0] * num_target_views       # target views
        )[:, None, None, None]
        
        value_dict['source_view_mask'] = source_view_mask
        value_dict['scene_name'] = scene_name
        value_dict['edit_type'] = editing_type
        return value_dict
        # ['cond_frames_without_noise', 'cond_frames', 'cond_frames_mask', 'cond_aug', 'plucker_coordinate', 'c2w', 'K', 'camera_mask', 'source_view_mask']






if __name__ == "__main__":

    VERSION_DICT = {
        "H": 576,
        "W": 576,
        "T": 21,
        "C": 4,
        "f": 8,
        "options": {
            "chunk_strategy": "nearest-gt",
            "video_save_fps": 30.0,
            "beta_linear_start": 5e-6,
            "log_snr_shift": 2.4,
            "guider_types": 1,
            "cfg": 2.0,
            "camera_scale": 2.0,
            "num_steps": 50,
            "cfg_min": 1.2,
            "encoding_t": 1,
            "decoding_t": 1,
            "num_inputs": None,
            "seed": 23
        }
    }
    
    print("=== 测试默认采样（基于文件数量）===")
    dataset_default = SEVAEDITDATASET(0, '/data/liyi_chen/code/generative-models-edit/image_editing_utils/sevaedit_0808/jsons',
                                         VERSION_DICT, ("co3d", "dl3dv"), ("remove"), balanced_sampling=False)
    
    print("\n=== 测试均衡采样（数据集间均衡）===")
    dataset_balanced = SEVAEDITDATASET(0, '/data/liyi_chen/code/generative-models-edit/image_editing_utils/sevaedit_0808/jsons',
                                          VERSION_DICT, ("co3d", "dl3dv"), ("remove"), balanced_sampling=True)
    
    # 测试采样分布
    dataset_counts = {}
    for i in range(100):
        sample = dataset_balanced[i]
        # 从scene_name或其他方式推断数据集
        scene_name = sample['scene_name']
        # 这里需要根据实际的scene_name格式来推断数据集
        print(f"Sample {i}: scene_name = {scene_name}")
        if i >= 10:  # 只测试前10个样本
            break


        