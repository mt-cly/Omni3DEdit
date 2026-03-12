import contextlib
import os
import os.path as osp
import sys
from typing import cast
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
import imageio.v3 as iio
from PIL import Image
import numpy as np
import torch


class Dust3rPipeline(object):
    def __init__(self, device: str | torch.device = "cuda"):
        # submodule_path = osp.realpath(
        #     osp.join(osp.dirname(__file__), "../../third_party/dust3r/")
        # )
        # if submodule_path not in sys.path:
        #     sys.path.insert(0, submodule_path)
        # try:
        #     with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        #         from dust3r.cloud_opt import (  # type: ignore[import]
        #             GlobalAlignerMode,
        #             global_aligner,
        #         )
        #         from dust3r.image_pairs import make_pairs  # type: ignore[import]
        #         from dust3r.inference import inference  # type: ignore[import]
        #         from dust3r.model import AsymmetricCroCo3DStereo  # type: ignore[import]
        #         from dust3r.utils.image import load_images  # type: ignore[import]
        # except ImportError:
        #     raise ImportError(
        #         "Missing required submodule: 'dust3r'. Please ensure that all submodules are properly set up.\n\n"
        #         "To initialize them, run the following command in the project root:\n"
        #         "  git submodule update --init --recursive"
        #     )

        self.device = torch.device(device)
        self.model = AsymmetricCroCo3DStereo.from_pretrained(
            "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
        ).to(self.device)

        self._GlobalAlignerMode = GlobalAlignerMode
        self._global_aligner = global_aligner
        self._make_pairs = make_pairs
        self._inference = inference
        self._load_images = load_images

    def infer_cameras_and_points(
        self,
        img_paths: list[str],
        Ks: list[list] = None,
        c2ws: list[list] = None,
        batch_size: int = 16,
        schedule: str = "cosine",
        lr: float = 0.01,
        niter: int = 500,
        min_conf_thr: int = 3,
    ) -> tuple[
        list[np.ndarray], np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]
    ]:
        num_img = len(img_paths)
        if num_img == 1:
            print("Only one image found, duplicating it to create a stereo pair.")
            img_paths = img_paths * 2

        images = self._load_images(img_paths, size=512)
        pairs = self._make_pairs(
            images,
            scene_graph="complete",
            prefilter=None,
            symmetrize=True,
        )
        output = self._inference(pairs, self.model, self.device, batch_size=batch_size)

        ori_imgs = [iio.imread(p) for p in img_paths]
        ori_img_whs = np.array([img.shape[1::-1] for img in ori_imgs])
        img_whs = np.concatenate([image["true_shape"][:, ::-1] for image in images], 0)

        scene = self._global_aligner(
            output,
            device=self.device,
            mode=self._GlobalAlignerMode.PointCloudOptimizer,
            same_focals=True,
            optimize_pp=False,  # True,
            min_conf_thr=min_conf_thr,
        )

        # if Ks is not None:
        #     scene.preset_focal(
        #         torch.tensor([[K[0, 0], K[1, 1]] for K in Ks])
        #     )

        if c2ws is not None:
            scene.preset_pose(c2ws)

        _ = scene.compute_global_alignment(
            init="msp", niter=niter, schedule=schedule, lr=lr
        )

        imgs = cast(list, scene.imgs)
        Ks = scene.get_intrinsics().detach().cpu().numpy().copy()
        c2ws = scene.get_im_poses().detach().cpu().numpy()  # type: ignore
        pts3d = [x.detach().cpu().numpy() for x in scene.get_pts3d()]  # type: ignore
        if num_img > 1:
            masks = [x.detach().cpu().numpy() for x in scene.get_masks()]
            points = [p[m] for p, m in zip(pts3d, masks)]
            point_colors = [img[m] for img, m in zip(imgs, masks)]
        else:
            points = [p.reshape(-1, 3) for p in pts3d]
            point_colors = [img.reshape(-1, 3) for img in imgs]

        # Convert back to the original image size.
        imgs = ori_imgs
        Ks[:, :2, -1] *= ori_img_whs / img_whs
        Ks[:, :2, :2] *= (ori_img_whs / img_whs).mean(axis=1, keepdims=True)[..., None]

        return imgs, Ks, c2ws, points, point_colors

class VGGTPipeline(object):
    def __init__(self, device: str | torch.device = "cuda"):
        print("be")
        self.device = torch.device(device)
        # 检查是否支持 bfloat16
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        # 初始化模型
        self.model = VGGT.from_pretrained("facebook/VGGT-1B").to(self.dtype).to(self.device)
        self.model.eval()


    def closed_form_inverse_se3(self, se3, R=None, T=None):
        """
        Compute the inverse of each 4x4 (or 3x4) SE3 matrix in a batch.

        If `R` and `T` are provided, they must correspond to the rotation and translation
        components of `se3`. Otherwise, they will be extracted from `se3`.

        Args:
            se3: Nx4x4 or Nx3x4 array or tensor of SE3 matrices.
            R (optional): Nx3x3 array or tensor of rotation matrices.
            T (optional): Nx3x1 array or tensor of translation vectors.

        Returns:
            Inverted SE3 matrices with the same type and device as `se3`.

        Shapes:
            se3: (N, 4, 4)
            R: (N, 3, 3)
            T: (N, 3, 1)
        """
        # Check if se3 is a numpy array or a torch tensor
        is_numpy = isinstance(se3, np.ndarray)

        # Validate shapes
        if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
            raise ValueError(f"se3 must be of shape (N,4,4), got {se3.shape}.")

        # Extract R and T if not provided
        if R is None:
            R = se3[:, :3, :3]  # (N,3,3)
        if T is None:
            T = se3[:, :3, 3:]  # (N,3,1)

        # Transpose R
        if is_numpy:
            # Compute the transpose of the rotation for NumPy
            R_transposed = np.transpose(R, (0, 2, 1))
            # -R^T t for NumPy
            top_right = -np.matmul(R_transposed, T)
            inverted_matrix = np.tile(np.eye(4), (len(R), 1, 1))
        else:
            R_transposed = R.transpose(1, 2)  # (N,3,3)
            top_right = -torch.bmm(R_transposed, T)  # (N,3,1)
            inverted_matrix = torch.eye(4, 4)[None].repeat(len(R), 1, 1)
            inverted_matrix = inverted_matrix.to(R.dtype).to(R.device)

        inverted_matrix[:, :3, :3] = R_transposed
        inverted_matrix[:, :3, 3:] = top_right

        return inverted_matrix

    def infer_cameras_and_points(
        self,
        img_paths: list[str],
        only_camera: bool = False,
        batch_size: int = 1,
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
        """
        从输入图像中估计相机参数和3D点云
        
        Args:
            img_paths: 输入图像路径列表
            batch_size: 批处理大小
            
        Returns:
            input_imgs: 原始图像列表 [h, w, 3], 范围 [0, 255]
            input_Ks: 相机内参矩阵 [f, 3, 3]
            input_c2ws: 相机外参矩阵 [f, 4, 4]
            points: 3D点云列表 [num_pnt, 3]
            point_colors: 点云颜色列表 [num_pnt, 3], 范围 [0, 1]
        """
        # 加载和预处理图像
        ori_imgs, images = load_and_preprocess_images(img_paths)
        images = images.to(self.dtype).to(self.device)
        self.model = self.model.to(self.device)
        # 保存原始图像尺寸
        # ori_imgs = [iio.imread(p) for p in img_paths]
        ori_img_whs = np.array([img.size for img in ori_imgs])
        ori_imgs = [np.array(img) for img in ori_imgs]
        img_whs = np.array([img.shape[-2:][::-1] for img in images])

        
        with torch.inference_mode():
            with torch.amp.autocast('cuda', dtype=self.dtype):
                # 添加批次维度
                images = images[None]  # [1, n, 3, h, w]
                
                # 特征聚合
                aggregated_tokens_list, ps_idx = self.model.aggregator(images)
                
                # 预测相机参数
                pose_enc = self.model.camera_head(aggregated_tokens_list)[-1]
            # 转换为外参和内参矩阵 (OpenCV 坐标系)
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
            input_Ks = intrinsic.squeeze(0).cpu().float().numpy()
            input_w2cs = extrinsic.squeeze(0).cpu().float().numpy()
            input_w2cs = np.concatenate([input_w2cs, np.array([[0, 0, 0, 1]])[None].repeat(input_w2cs.shape[0], axis=0)], axis=1)
            input_c2ws = self.closed_form_inverse_se3(input_w2cs)

            # 转换回原始图像尺寸
            input_imgs = ori_imgs
            input_Ks[:, :2, -1] *= ori_img_whs / img_whs
            input_Ks[:, :2, :2] *= (ori_img_whs / img_whs).mean(axis=1, keepdims=True)[..., None]

            if only_camera:
                del aggregated_tokens_list, ps_idx, pose_enc, extrinsic, intrinsic
                return ori_imgs, input_Ks, input_c2ws
            
            # 预测深度图
            depth_map, depth_conf = self.model.depth_head(aggregated_tokens_list, images, ps_idx)
            
            # 通过反投影构建3D点云
            point_map = unproject_depth_map_to_point_map(
                depth_map.squeeze(0), 
                extrinsic.squeeze(0), 
                intrinsic.squeeze(0)
            )
            
            # 转换为 numpy
            input_imgs = [img.permute(1, 2, 0).contiguous().cpu().numpy() for img in images.squeeze(0)]
            
            
            points = [p.reshape(-1, 3) for p in point_map]
            point_colors = [img.reshape(-1, 3) for img in input_imgs]
            
            
            return input_imgs, input_Ks, input_c2ws, points, point_colors, aggregated_tokens_list, images, ps_idx

    def track_points(
        self,
        img_paths: list[str],
        query_points: torch.Tensor,
        batch_size: int = 1,
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        跟踪指定点在图像序列中的轨迹
        
        Args:
            img_paths: 输入图像路径列表
            query_points: 要跟踪的点坐标 [N, 2]
            batch_size: 批处理大小
            
        Returns:
            track_list: 跟踪点列表
            vis_score: 可见性分数
            conf_score: 置信度分数
        """
        # 加载和预处理图像
        images = load_and_preprocess_images(img_paths).to(self.device)
        
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=self.dtype):
                images = images[None]  # [1, n, 3, h, w]
                aggregated_tokens_list, ps_idx = self.model.aggregator(images)
                
                # 预测轨迹
                track_list, vis_score, conf_score = self.model.track_head(
                    aggregated_tokens_list, 
                    images, 
                    ps_idx, 
                    query_points=query_points[None]
                )
                
                return track_list, vis_score, conf_score

if __name__ == "__main__":
    vggt = VGGTPipeline()
    dust3r = Dust3rPipeline(device='cuda')
    img_paths = ["/data/liyi_chen/code/stable-virtual-camera/assets_demo_cli/garden_flythrough/images/000.png", 
                 "/data/liyi_chen/code/stable-virtual-camera/assets_demo_cli/garden_flythrough/images/001.png", 
                 "/data/liyi_chen/code/stable-virtual-camera/assets_demo_cli/garden_flythrough/images/002.png",
                 "/data/liyi_chen/code/stable-virtual-camera/assets_demo_cli/garden_flythrough/images/003.png",
                 "/data/liyi_chen/code/stable-virtual-camera/assets_demo_cli/garden_flythrough/images/004.png",
                 ]
    (
    vggt_input_imgs, # a list of images [h, w, 3], range [0, 255]
    vggt_input_Ks, # [f, 3, 3]
    vggt_input_c2ws, # [f, 4, 4]
    vggt_points, # a list of points [num_pnt, 3]
    vggt_point_colors, # a list of points [num_pnt, 3], range [0, 1  ]
    vggt_aggregated_tokens_list,
    vggt_images,
    vggt_ps_idx ) = vggt.infer_cameras_and_points(img_paths)    
    
    (
    dust3r_input_imgs, # a list of images [h, w, 3], range [0, 255]
    dust3r_input_Ks, # [f, 3, 3]
    dust3r_input_c2ws, # [f, 4, 4]
    dust3r_points, # a list of points [num_pnt, 3]
    dust3r_point_colors, # a list of points [num_pnt, 3], range [0, 1  ]
    ) = dust3r.infer_cameras_and_points(img_paths)
    print()