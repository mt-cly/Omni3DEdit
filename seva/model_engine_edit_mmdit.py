import math
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union
import time
import re
import pytorch_lightning as pl
import torch
from torchvision.utils import save_image
from omegaconf import ListConfig, OmegaConf
import os
from safetensors.torch import load_file as load_safetensors
from torch.optim.lr_scheduler import LambdaLR
import random
import torch.nn.functional as F
import deepspeed
import wandb
from sgm.modules import UNCONDITIONAL_CONFIG
from sgm.modules.autoencoding.temporal_ae import VideoDecoder
from sgm.modules.diffusionmodules.wrappers import OPENAIUNETWRAPPER
from sgm.modules.ema import LitEma
from sgm.util import (default, disabled_train, get_obj_from_str, append_dims,
                    instantiate_from_config, log_txt_as_img)
from pytorch_lightning.utilities import rank_zero_only
import copy
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from seva.ssim_psnr import calculate_ssim_pt, calculate_psnr_pt
from seva.eval import (
    IS_TORCH_NIGHTLY,
    compute_relative_inds,
    create_transforms_simple,
    infer_prior_inds,
    infer_prior_stats,
    run_one_scene,
)
from seva.geometry import (
    generate_interpolated_path,
    generate_spiral_path,
    get_arc_horizontal_w2cs,
    get_default_intrinsics,
    get_lookat,
    get_preset_pose_fov,
)
from seva.model_mmdit import SGMWrapper as SGMWrapper
from seva.utils_mmdit import load_model
from seva.model import SGMWrapper as SGMWrapper_baseline
from seva.utils import load_model as load_model_baseline
from seva.modules.autoencoder import AutoEncoder
from seva.modules.conditioner import CLIPConditioner
from seva.sampling import DDPMDiscretization, DiscreteDenoiser
from seva.model_conditioner import SevaConditioner
from seva.eval import create_samplers



class SevaEngine(pl.LightningModule):
    def __init__(
        self,
        VERSION_DICT,
        network_config,
        baseline_denoiser_config,
        denoiser_config,
        first_stage_config,
        sigma_sampler_config,
        loss_weighting_config,
        offset_noise_level=0.0,
        loss_type="l2",
        ucg_rate=0.0,
        conditioner_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        sampler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        optimizer_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        scheduler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        loss_fn_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        network_wrapper: Union[None, str] = None,
        ckpt_path: Union[None, str] = None,
        use_ema: bool = False,
        ema_decay_rate: float = 0.9999,
        scale_factor: float = 1.0,
        disable_first_stage_autocast=False,
        input_key: str = "cond_frames",
        log_keys: Union[List, None] = None,
        no_cond_log: bool = False,
        compile_model: bool = False,
        en_and_decode_n_samples_a_time: Optional[int] = None,
        inference_save_dir: Optional[str] = None,
    ):
        super().__init__()

        self.ucg_rate = ucg_rate
        self.MODEL = SGMWrapper(load_model(device="cpu", verbose=True)).to(torch.float32)
        # for name, param in self.MODEL.named_parameters():
        #     if not name.__contains__('condition_sourceview_module'):
        #         param.requires_grad = False
        

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
        # Initialize the model and load the pretrained weights.
        # This will automatically download the model weights the first time it's run, which may take a while.


        self.BASELINE_DENOISER = instantiate_from_config(baseline_denoiser_config)

        self.DENOISER = instantiate_from_config(denoiser_config)
        self.VERSION_DICT = VERSION_DICT
        self.options = self.VERSION_DICT["options"]
        self._init_seva_conditioner()
        # self._init_vggt()
        self.sigma_sampler = instantiate_from_config(sigma_sampler_config)
        self.loss_weighting = instantiate_from_config(loss_weighting_config)
        self.offset_noise_level = offset_noise_level    
        self.loss_type = loss_type
        self.sampler = create_samplers(
                self.options["guider_types"],
                self.DENOISER.discretization,
                [self.VERSION_DICT['T']],
                self.options["num_steps"],
                self.options["cfg_min"],
                abort_event=None,
            )[0]
        #############

        self.log_keys = log_keys
        self.input_key = input_key
        self.optimizer_config = default(
            optimizer_config, {"target": "torch.optim.AdamW"}
        )

        self.scheduler_config = scheduler_config
        self._init_first_stage(first_stage_config)

        # self.loss_fn = (
        #     instantiate_from_config(loss_fn_config)
        #     if loss_fn_config is not None
        #     else None
        # )

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model, decay=ema_decay_rate)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.scale_factor = scale_factor
        self.disable_first_stage_autocast = disable_first_stage_autocast
        self.no_cond_log = no_cond_log

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

        self.en_and_decode_n_samples_a_time = en_and_decode_n_samples_a_time

        # 保存inference_save_dir参数
        self.inference_save_dir = inference_save_dir

        # 用于保存checkpoint中的global_step（test时PL不会恢复global_step）
        self._ckpt_global_step = 0

        # 添加统计字典
        self.statistics = {
            'sigma_losses': {},  # 存储每个sigma值的loss列表
            'sigma_means': {},   # 存储每个sigma值的平均loss
            'total_samples': 0   # 总样本数
        }
        
        # 初始化sigma值的统计
        for i in range(1, 84, 5):
            self.statistics['sigma_losses'][i] = []
            self.statistics['sigma_means'][i] = 0.0

    def init_from_ckpt(
        self,
        path: str,
    ) -> None:
        if path.endswith("ckpt"):
            sd = torch.load(path, map_location="cpu")["state_dict"]
        elif path.endswith("safetensors"):
            sd = load_safetensors(path)
        else:
            raise NotImplementedError

        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(
            f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def _init_first_stage(self, config):
        model = instantiate_from_config(config).eval()
        model.train = disabled_train
        model = instantiate_from_config(config)
        for param in model.parameters():
            param.requires_grad = False
        self.first_stage_model = model

    def _init_vggt(self):
        vggt = VGGT.from_pretrained("facebook/VGGT-1B").eval()
        vggt.train = disabled_train
        for param in vggt.parameters():
            param.requires_grad = False
        self.vggt = vggt

    def _init_seva_conditioner(self):
        AE = AutoEncoder(chunk_size=1).eval()
        CONDITIONER = CLIPConditioner().eval()
        for param in AE.parameters():
            param.requires_grad = False
        for param in CONDITIONER.parameters():
            param.requires_grad = False
        self.SEVA_CONDITIONER = SevaConditioner(AE, CONDITIONER, self.VERSION_DICT, self.options["encoding_t"])

    # def _init_seva_conditioner(self, config):
    #     # delete the 'emb_models' key in config
    #     if "emb_models" in config.params:
    #         del config.params["emb_models"]
        
        
    #     AE = instantiate_from_config(config.params.ae).eval()
    #     CONDITIONER = instantiate_from_config(config.params.conditioner).eval()
    #     for param in AE.parameters():
    #         param.requires_grad = False
    #     for param in CONDITIONER.parameters():
    #         param.requires_grad = False
    #     self.SEVA_CONDITIONER = SevaConditioner(AE, CONDITIONER, self.VERSION_DICT, self.options["encoding_t"])
        
    #     seva_conditioner.ae.eval()
    #     seva_conditioner.conditioner.eval()
    #     for param in seva_conditioner.ae.parameters():
    #         param.requires_grad = False
    #     for param in seva_conditioner.conditioner.parameters(): 
    #         param.requires_grad = False
    #     self.SEVA_CONDITIONER = seva_conditioner


    def get_input(self, batch):
        # assuming unified data format, dataloader returns a dict.
        # image tensors should be scaled to -1 ... 1 and in bchw format
        return batch[self.input_key].contiguous()

    # @torch.no_grad()
    def decode_first_stage(self, z):
        # z = 1.0 / self.scale_factor * z
        n_samples = default(self.en_and_decode_n_samples_a_time, z.shape[0])

        n_rounds = math.ceil(z.shape[0] / n_samples)
        all_out = []
        # with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
        for n in range(n_rounds):
            if isinstance(self.first_stage_model.decoder, VideoDecoder):
                kwargs = {"timesteps": len(z[n * n_samples : (n + 1) * n_samples])}
            else:
                kwargs = {}
            # out = self.first_stage_model.decode(
            #     z[n * n_samples : (n + 1) * n_samples], **kwargs
            # )
            out = deepspeed.checkpointing.checkpoint(self.SEVA_CONDITIONER.ae.decode,
                z[n * n_samples : (n + 1) * n_samples], self.options["decoding_t"]
            )
            all_out.append(out)
        out = torch.cat(all_out, dim=0)
        return out

    @torch.no_grad()
    def encode_first_stage(self, x):
        n_samples = default(self.en_and_decode_n_samples_a_time, x.shape[0])
        n_rounds = math.ceil(x.shape[0] / n_samples)
        all_out = []
        # with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
        for n in range(n_rounds):
            out = self.SEVA_CONDITIONER.ae.encode(
                    x[n * n_samples : (n + 1) * n_samples], self.options["encoding_t"]
                    )
            all_out.append(out)
        z = torch.cat(all_out, dim=0)
        # z = self.scale_factor * z
        return z



    def visualize_denoised_results(self, noise, sigmas, x_target_latent, c, additional_model_inputs):
        with torch.inference_mode():
            for i in range(1, 84, 5):
                sigmas_bc = append_dims(torch.ones_like(sigmas).to(x_target_latent)* i, x_target_latent.ndim)
                noised_input = self.get_noised_input(sigmas_bc, noise, x_target_latent)
                model_output = self.DENOISER(
                    self.MODEL, noised_input, sigmas, c, **additional_model_inputs
                )
                
                # save the model_output
                loss = self.get_loss(model_output, x_target_latent, 1)
                print(f'when sigma: {i}, loss: {loss.mean()}')
                # rgb = self.decode_first_stage(model_output)
                # rgb.save(f'model_output_sigma{i}.png')


    def forward(self, x_target, x_target_latent, batch):
        '''
        x_target: input rgb (bs*f, 3, h, w) all visible views
        x_target_latent: input latent (bs*f, 5, h, w) target latent
        batch: Dict, used to obtain conditioning
        return: loss, loss_dict
        '''
        bs, num_f = x_target_latent.shape[0]//self.VERSION_DICT['T'], self.VERSION_DICT['T']
        c, uc, additional_model_inputs, additional_sampler_inputs = self.SEVA_CONDITIONER(batch)
        additional_model_inputs['source_view_mask'] = batch['source_view_mask']



        sigmas = self.sigma_sampler(bs).unsqueeze(1).repeat(1, num_f).flatten().to(x_target_latent)
        print(f'at rank {int(os.environ["LOCAL_RANK"])}, sigmas: {sigmas[::21]}, target_latent_max:{x_target_latent.max()}')

        noise = torch.randn_like(x_target_latent)
        
        if self.offset_noise_level > 0.0:
            offset_shape = (
                (x_target_latent.shape[0], 1, x_target_latent.shape[2])
                if self.n_frames is not None
                else (x_target_latent.shape[0], x_target_latent.shape[1])
            )
            noise = noise + self.offset_noise_level * append_dims(
                torch.randn(offset_shape, device=x_target_latent.device),
                x_target_latent.ndim,
            )
        sigmas_bc = append_dims(sigmas, x_target_latent.ndim)
        # todo: liyi
        noised_input = self.get_noised_input(sigmas_bc, noise, x_target_latent)
        # noised_input = self.get_noised_input(sigmas_bc, noise, torch.randn_like(x_target_latent))
        
        # uc: (1) using unconditional guidance (2) override original view with target view (3) no source view in additional_model_inputs
        if self.training and random.random() < self.ucg_rate:
            c = uc  # unconditional guidance
            noised_input[ batch['source_view_mask'].squeeze()==1] =noised_input[~batch['cond_frames_mask']].clone()  # override original view with target view
            additional_model_inputs['source_view_mask'] = additional_model_inputs['source_view_mask'] * 0 
      

        model_output = self.DENOISER(
            self.MODEL, noised_input, sigmas, c, **additional_model_inputs
        )

        # original generative loss  
        w = append_dims(self.loss_weighting(sigmas), x_target_latent.ndim)
        w[additional_sampler_inputs['input_frame_mask']] = 0
        loss = self.get_loss(model_output, x_target_latent, w)
        if int(os.environ['LOCAL_RANK']) == -1:
            print(f'sigma: {sigmas.view(bs, num_f).mean(dim=1)}')
            unint_weight = torch.ones_like(w)
            unint_weight[additional_sampler_inputs['input_frame_mask']] = 0
            mse = self.get_loss(model_output, x_target_latent, unint_weight).view(bs, num_f).mean(dim=1)
            # x_source_latent_expand = model_output.clone()
            # x_source_latent_expand[~batch['cond_frames_mask']] = x_source_latent
            print(f'MSE: {mse}')
            print(f'loss: {loss.view(bs, num_f).mean(dim=1)}')
            with torch.no_grad():
                for bs_id in range(bs):
                    model_output_rgb = torch.utils.checkpoint.checkpoint(self.SEVA_CONDITIONER.ae.decode, model_output[bs_id*num_f:(bs_id+1)*num_f], self.options["decoding_t"])
                    save_vis = model_output_rgb/2+0.5
                    save_image(save_vis, f'source_pred_gt_bs{bs_id}_sigma{sigmas[bs_id*num_f]}_mse{mse[bs_id]}.png')
                    pass
        loss_mean = loss.mean()


        # ######################################
        # FROZEN_MODEL = SGMWrapper_baseline(load_model_baseline(device="cpu", verbose=True)).to(x_target_latent)
        # with torch.inference_mode():
        #     for i in range(0, len(self.sigma_sampler.sigmas), 49):
        #         _sigmas = (torch.ones_like(sigmas) * self.sigma_sampler.sigmas[i]).to(sigmas)
        #         _sigmas_bc = append_dims(_sigmas.to(x_target_latent), x_target_latent.ndim)
        #         _noised_input = self.get_noised_input(_sigmas_bc, noise, x_target_latent)
        #         _model_output = self.DENOISER(
        #             self.MODEL, _noised_input, _sigmas, c, **additional_model_inputs
        #         )       
        #         # _model_output = self.DENOISER(
        #             # FROZEN_MODEL, _noised_input, _sigmas, c, **additional_model_inputs
        #         # )       
        #         _w = append_dims(self.loss_weighting(_sigmas), x_target_latent.ndim)
        #         _w[additional_sampler_inputs['input_frame_mask']] = 0
        #         _loss = self.get_loss(_model_output, x_target_latent, _w)
        #         print(f'when sigma: {_sigmas.mean()}, loss: {_loss.mean()}')
        #########################################
        return loss_mean



    def get_loss(self, model_output, target, w):
        if self.loss_type == "l2":
            return torch.mean(
                (w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1
            )
        elif self.loss_type == "l1":
            return torch.mean(
                (w * (model_output - target).abs()).reshape(target.shape[0], -1), 1
            )
        elif self.loss_type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
            return loss
        else:
            raise NotImplementedError(f"Unknown loss type {self.loss_type}")
        
    def get_noised_input(
        self, sigmas_bc: torch.Tensor, noise: torch.Tensor, input: torch.Tensor
    ) -> torch.Tensor:
        noised_input = input + noise * sigmas_bc
        return noised_input
    
    def shared_step(self, batch: Dict) -> Any:
        # with torch.inference_mode():
        for key in ["cond_frames", "cond_frames_mask", "plucker_coordinate", "c2w", "K", "camera_mask", "source_view_mask"]:
            batch[key] = batch[key].flatten(start_dim=0, end_dim=1).contiguous()
        x_target = self.get_input(batch)
        x_target_latent = self.encode_first_stage(x_target)
        
        batch["global_step"] = self.global_step
        loss = self(x_target, x_target_latent, batch)
        torch.cuda.empty_cache()
        return loss

    def validation_step(self, batch, batch_idx):
        print('validation step')
        return 0

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)

        self.logger.log_metrics({'loss': loss,
                                'global_step': self.global_step,
                                'lr': self.optimizers().param_groups[0]["lr"] if self.scheduler_config is not None else 0,
                                })
        
        # self.log_dict(
        #     {'loss': loss}, prog_bar=True, logger=True, on_step=True, on_epoch=False
        # )

        self.log(
            "global_step",
            self.global_step,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )
    
        if self.scheduler_config is not None:
            lr = self.optimizers().param_groups[0]["lr"]
            self.log(
                "lr_abs", lr, prog_bar=True, logger=True, on_step=True, on_epoch=False
            )
        return loss


    def on_load_checkpoint(self, checkpoint):
        self._ckpt_global_step = checkpoint.get("global_step", 0)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        self.BASELINE_MODEL = SGMWrapper_baseline(
            load_model_baseline(device="cpu", verbose=True)
        ).to(torch.float32).to(self.device)

        base_save_dir = self.inference_save_dir or self.VERSION_DICT["options"].get("test_save_dir", "results")
        save_dir = f"{base_save_dir}/{batch['edit_type'][0]}/{batch['scene_name'][0]}_iter_{self._ckpt_global_step}_cfg_{self.options['cfg_min']}_{self.options['cfg']}"
        os.makedirs(save_dir, exist_ok=True)

        with torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            original_batch = copy.deepcopy(batch)
            bs, num_f = 1, self.VERSION_DICT['T']

            for key in ["cond_frames", "cond_frames_mask", "plucker_coordinate", "c2w", "K", "camera_mask", "source_view_mask"]:
                original_batch[key] = original_batch[key][0:bs].flatten(start_dim=0, end_dim=1).contiguous()

            x_target = self.get_input(original_batch)
            c, uc, additional_model_inputs, additional_sampler_inputs = self.SEVA_CONDITIONER(original_batch)

            # Prepare zeroshot conditioning (exclude source views)
            zeroshot_batch = copy.deepcopy(original_batch)
            source_mask_2d = (zeroshot_batch['source_view_mask'].squeeze(-1).squeeze(-1).squeeze(-1) == 1)
            zeroshot_batch['cond_frames_mask'] = zeroshot_batch['cond_frames_mask'] & (~source_mask_2d)
            zeroshot_c, zeroshot_uc, zeroshot_additional_model_inputs, _ = self.SEVA_CONDITIONER(zeroshot_batch)
            doubled_mask = torch.cat([original_batch['source_view_mask'] * 0, original_batch['source_view_mask']], dim=0)
            shape = (bs * num_f, self.VERSION_DICT["C"], self.VERSION_DICT["H"] // self.VERSION_DICT["f"], self.VERSION_DICT["W"] // self.VERSION_DICT["f"])

            def denoise_func(_input, _sigma, _c):
                return self.DENOISER(self.MODEL, _input, _sigma, _c,
                                     **additional_model_inputs,
                                     source_view_mask=doubled_mask)

            def zeroshot_denoise_func(_input, _sigma, _c):
                return self.BASELINE_DENOISER(self.BASELINE_MODEL, _input, _sigma, _c,
                                              **zeroshot_additional_model_inputs)

            samples_z = self.sampler.mixed_denoise_zeroshot(
                denoise_func,
                zeroshot_denoise_func,
                torch.randn(shape, device=self.device),
                cond=c,
                uc=uc,
                zeroshot_cond=zeroshot_c,
                zeroshot_uc=zeroshot_uc,
                scale=(self.options["cfg"][0] if isinstance(self.options["cfg"], (list, tuple)) else self.options["cfg"]),
                verbose=True,
                return_intermediate=False,
                **additional_sampler_inputs
            )
            samples_rgb = self.SEVA_CONDITIONER.ae.decode(samples_z, self.options["decoding_t"])
            samples_rgb = samples_rgb[shape[0] // 2 + 1:]

            if self.global_rank == 0:
                out_name = os.path.join(save_dir, "sampled.png")
                edited_imgs = (samples_rgb.clamp(-1, 1) + 1) / 2
                source_imgs = (batch["cond_frames"][0][:edited_imgs.shape[0]].to(edited_imgs.device).clamp(-1, 1) + 1) / 2
                image_names = batch.get('image_names', None)
                names_list = image_names[0] if (image_names is not None and isinstance(image_names[0], list)) else image_names

                # Save 10 pair images: left=source, right=edited
                for i in range(edited_imgs.shape[0]):
                    if names_list is not None and i < len(names_list):
                        raw = names_list[i]
                        if isinstance(raw, (list, tuple)):
                            raw = next((e for e in raw if isinstance(e, (str, bytes))), raw[0])
                        if isinstance(raw, bytes):
                            raw = raw.decode('utf-8', errors='ignore')
                        if hasattr(raw, '__fspath__'):
                            raw = os.fspath(raw)
                        base_name = os.path.splitext(os.path.basename(str(raw)))[0]
                        # Keep output names compact and deterministic: original_00003 / edited_00003.
                        base_name = base_name.split("__", 1)[0]
                        m = re.match(r"^(original|edited)_(\d{5})(?:_.+)?$", base_name)
                        if m:
                            base_name = f"{m.group(1)}_{m.group(2)}"
                    else:
                        base_name = f"view{i:02d}"
                    pair_img = torch.cat([source_imgs[i:i+1], edited_imgs[i:i+1]], dim=-1)
                    save_image(pair_img, os.path.join(save_dir, f"{base_name}.png"))

                # Save summary image: 2x10 grid (top=source 10, bottom=edited 10)
                summary = torch.cat([source_imgs, edited_imgs], dim=0)
                summary = F.interpolate(summary, scale_factor=0.5, mode='bilinear', align_corners=False)
                save_image(summary, out_name, nrow=source_imgs.shape[0])

        return {}



    def on_train_start(self, *args, **kwargs):
        if self.sampler is None:
            raise ValueError("Sampler and loss function need to be set for training.")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def instantiate_optimizer_from_config(self, params, lr, cfg):
        return get_obj_from_str(cfg["target"])(
            params, lr=lr, **cfg.get("params", dict())
        )

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.MODEL.parameters())
        opt = self.instantiate_optimizer_from_config(params, lr, self.optimizer_config)
        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)
            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    "scheduler": LambdaLR(opt, lr_lambda=scheduler.schedule),
                    "interval": "step",
                    "frequency": 1,
                }
            ]
            return [opt], scheduler
        return opt

    @torch.no_grad()
    def sample(
        self,
        cond: Dict,
        uc: Union[Dict, None] = None,
        batch_size: int = 16,
        shape: Union[None, Tuple, List] = None,
        **kwargs,
    ):
        randn = torch.randn(batch_size, *shape).to(self.device)

        denoiser = lambda input, sigma, c: self.denoiser(
            self.model, input, sigma, c, **kwargs
        )
        samples = self.sampler(denoiser, randn, cond, uc=uc)
        return samples

    @torch.no_grad()
    def log_conditionings(self, batch: Dict, n: int) -> Dict:
        """
        Defines heuristics to log different conditionings.
        These can be lists of strings (text-to-image), tensors, ints, ...
        """
        image_h, image_w = batch[self.input_key].shape[2:]
        log = dict()

        for embedder in self.conditioner.embedders:
            if (
                (self.log_keys is None) or (embedder.input_key in self.log_keys)
            ) and not self.no_cond_log:
                x = batch[embedder.input_key][:n]
                if isinstance(x, torch.Tensor):
                    if x.dim() == 1:
                        # class-conditional, convert integer to string
                        x = [str(x[i].item()) for i in range(x.shape[0])]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 4)
                    elif x.dim() == 2:
                        # size and crop cond and the like
                        x = [
                            "x".join([str(xx) for xx in x[i].tolist()])
                            for i in range(x.shape[0])
                        ]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        raise NotImplementedError()
                elif isinstance(x, (List, ListConfig)):
                    if isinstance(x[0], str):
                        # strings
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()
                log[embedder.input_key] = xc
        return log


    @rank_zero_only
    @torch.no_grad()
    def log_images(
        self,
        batch: Dict,
        N: int = 8,
        sample: bool = True,
        ucg_keys: List[str] = None,
        **kwargs,
    ) -> Dict:
        
        log = dict()    
        return log

    

