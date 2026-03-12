import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
import deepspeed
import math
import xformers.ops as xops

class LoRA_Linear(nn.Module):
    """
    A linear layer with two independent LoRA adapters, selectable via a mask.
    Preserves original parameter names for compatibility with pretrained weights.
    """
    def __init__(self, in_features, out_features, rank=4, alpha=4, bias=True):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.in_features = in_features
        self.out_features = out_features

        # Use the same parameter names as nn.Linear for compatibility
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
            
        # Initialize like nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        if rank > 0:
            # LoRA Adapter 1 ('default' for target views)
            self.lora_A1 = nn.Parameter(torch.zeros(rank, in_features))
            self.lora_B1 = nn.Parameter(torch.zeros(out_features, rank))

            # LoRA Adapter 2 ('source_lora' for source views)
            self.lora_A2 = nn.Parameter(torch.zeros(rank, in_features))
            self.lora_B2 = nn.Parameter(torch.zeros(out_features, rank))
            
            self.scaling = self.alpha / self.rank
            self.reset_lora_parameters()

    def reset_lora_parameters(self):
        if self.rank > 0:
            nn.init.kaiming_uniform_(self.lora_A1, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B1)
            nn.init.kaiming_uniform_(self.lora_A2, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B2)

    def forward(self, x, source_view_mask=None):
        assert source_view_mask is not None
        # Standard linear transformation
        base_out = F.linear(x, self.weight, self.bias)

        if self.rank == 0 or source_view_mask is None:
            return base_out
            
        # [优化点]：利用矩阵乘法结合律，预先合并 LoRA 权重，避免产生巨大的中间激活值
        merged_weight1 = (self.lora_B1 @ self.lora_A1) * self.scaling
        merged_weight2 = (self.lora_B2 @ self.lora_A2) * self.scaling
        
        delta1 = F.linear(x, merged_weight1)
        delta2 = F.linear(x, merged_weight2)

        source_view_mask = source_view_mask.to(delta1.dtype)

        # [优化点]：使用原生底层算子 torch.lerp 进行 Mask 融合
        final_delta = torch.lerp(delta1, delta2, source_view_mask)

        return base_out + final_delta

    def __repr__(self):
        return f"LoRA_Linear(in={self.in_features}, out={self.out_features}, rank={self.rank})"


class LoRA_Conv2d(nn.Module):
    """
    A Conv2d layer with two independent LoRA adapters, selectable via a mask.
    Preserves original parameter names for compatibility with pretrained weights.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, rank=4, alpha=4):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Use the same parameter names as nn.Conv2d for compatibility
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize like nn.Conv2d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        
        # Store conv parameters
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        
        if rank > 0:
            # LoRA Adapter 1 for target views
            self.lora_A1 = nn.Parameter(torch.zeros(rank, in_channels))
            self.lora_B1 = nn.Parameter(torch.zeros(out_channels, rank))
            
            # LoRA Adapter 2 for source views
            self.lora_A2 = nn.Parameter(torch.zeros(rank, in_channels))
            self.lora_B2 = nn.Parameter(torch.zeros(out_channels, rank))
            
            self.scaling = self.alpha / self.rank
            self.reset_lora_parameters()

    def reset_lora_parameters(self):
        if self.rank > 0:
            nn.init.kaiming_uniform_(self.lora_A1, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B1)
            nn.init.kaiming_uniform_(self.lora_A2, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B2)

    def forward(self, x, source_view_mask=None):
        assert source_view_mask is not None
        # Standard convolution
        base_out = F.conv2d(x, self.weight, self.bias, self.stride, self.padding)
        
        if self.rank == 0 or source_view_mask is None:
            return base_out
            
        b, in_c, h, w = x.shape
        out_c = base_out.shape[1]
        
        # [优化点]：预合并 LoRA 权重，并将其 reshape 为 1x1 卷积核格式
        weight1_1x1 = (self.lora_B1 @ self.lora_A1).view(out_c, in_c, 1, 1) * self.scaling
        weight2_1x1 = (self.lora_B2 @ self.lora_A2).view(out_c, in_c, 1, 1) * self.scaling
        
        # [优化点]：使用原生 1x1 卷积彻底消灭 permute 和 reshape 带来的显存开销
        delta1 = F.conv2d(x, weight1_1x1)
        delta2 = F.conv2d(x, weight2_1x1)

        source_view_mask = source_view_mask.to(delta1.dtype)
        
        # [优化点]：使用 torch.lerp 替代手动的 expand 和加减法
        final_delta = torch.lerp(delta1, delta2, source_view_mask)
        
        return base_out + final_delta

    def __repr__(self):
        return f"LoRA_Conv2d(in={self.in_channels}, out={self.out_channels}, kernel_size={self.kernel_size[0]}, rank={self.rank})"


def load_pretrained_weights_with_lora(model_with_lora, pretrained_state_dict, strict=True):
    """
    Load pretrained weights into a model with LoRA layers.
    
    Args:
        model_with_lora: Model with LoRA layers
        pretrained_state_dict: State dict from pretrained model
        strict: Whether to strictly enforce that all keys match
    
    Returns:
        Missing keys and unexpected keys
    """
    model_state_dict = model_with_lora.state_dict()
    
    # Filter out LoRA parameters from the current model
    filtered_pretrained = {}
    missing_keys = []
    unexpected_keys = []
    
    for key, value in pretrained_state_dict.items():
        if key in model_state_dict:
            # Check if shapes match
            if model_state_dict[key].shape == value.shape:
                filtered_pretrained[key] = value
            else:
                print(f"Shape mismatch for {key}: model {model_state_dict[key].shape} vs pretrained {value.shape}")
        else:
            unexpected_keys.append(key)
    
    # Find missing keys (excluding LoRA parameters)
    for key in model_state_dict.keys():
        if not any(lora_key in key for lora_key in ['lora_A1', 'lora_A2', 'lora_B1', 'lora_B2']):
            if key not in filtered_pretrained:
                missing_keys.append(key)
    
    # Load the filtered state dict
    model_with_lora.load_state_dict(filtered_pretrained, strict=False)
    
    if not strict:
        print(f"Loaded pretrained weights. Missing: {len(missing_keys)} keys, Unexpected: {len(unexpected_keys)} keys")
        if missing_keys:
            print(f"Missing keys: {missing_keys[:10]}{'...' if len(missing_keys) > 10 else ''}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys[:10]}{'...' if len(unexpected_keys) > 10 else ''}")
    
    return missing_keys, unexpected_keys


class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, lora_rank: int = 4):
        super().__init__()
        self.proj = LoRA_Linear(dim_in, dim_out * 2, rank=lora_rank)

    def forward(self, x: torch.Tensor, source_view_mask: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(x, source_view_mask).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: int = 4,
        dropout: float = 0.0,
        lora_rank: int = 4,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out or dim
        self.net = nn.Sequential(
            GEGLU(dim, inner_dim, lora_rank=lora_rank), nn.Dropout(dropout), LoRA_Linear(inner_dim, dim_out, rank=lora_rank)
        )

    def forward(self, x: torch.Tensor, source_view_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Handle the Sequential layers manually to pass source_view_mask to LoRA_Linear
        x = self.net[0](x, source_view_mask=source_view_mask)  # GEGLU
        x = self.net[1](x)  # Dropout
        x = self.net[2](x, source_view_mask=source_view_mask)  # LoRA_Linear
        return x


class Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        lora_rank: int = 4,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads
        context_dim = context_dim or query_dim

        self.to_q = LoRA_Linear(query_dim, inner_dim, rank=lora_rank, bias=False)
        self.to_k = LoRA_Linear(context_dim, inner_dim, rank=lora_rank, bias=False)
        self.to_v = LoRA_Linear(context_dim, inner_dim, rank=lora_rank, bias=False)
        
        self.to_out = nn.Sequential(
            LoRA_Linear(inner_dim, query_dim, rank=lora_rank), nn.Dropout(dropout)
        )

    # flashattn
    # def _perform_attention(self, q, k, v):
    #     q, k, v = map(
    #         lambda t: rearrange(t, "b l (h d) -> b h l d", h=self.heads).contiguous(),
    #         (q, k, v),
    #     )
    #     try:
    #         with torch.amp.autocast("cuda", dtype=torch.float16): # !!! bfloat16 fail to convergency due to limited precision
    #             with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    #                 out = F.scaled_dot_product_attention(q, k, v)

    #     except Exception as e:
    #         print(f'Error in _perform_attention: {e}')
    #         raise e

    #     return rearrange(out, "b h l d -> b l (h d)").contiguous()
    
    # xformers
    def _perform_attention(self, q, k, v, return_attn_weights: bool = False):
        q, k, v = map(
            lambda t: rearrange(t, "b l (h d) -> b l h d", h=self.heads).contiguous(),
            (q, k, v),
        )
        try:
            if return_attn_weights:
                # 计算 attention weights: q @ k^T, shape [b, l, h, d] @ [b, l, h, d] -> [b, h, l, l]
                # 使用 einsum: 'blhd,bkhd->bhlk' 表示 query_seq 和 key_seq 的点积
                attn_weights = torch.einsum('blhd,bkhd->bhlk', q, k) / math.sqrt(self.dim_head)
                attn_weights = F.softmax(attn_weights, dim=-1)  # softmax over key dimension
                # 立即移到 CPU 以节省显存
                attn_weights = attn_weights.detach().cpu()
                # 使用 xformers 计算输出
                out = xops.memory_efficient_attention(q, k, v)
                out = rearrange(out, "b l h d -> b l (h d)").contiguous()
                return out, attn_weights
            else:
                # 不需要 attention weights 时,直接使用 xformers
                out = xops.memory_efficient_attention(q, k, v)
                out = rearrange(out, "b l h d -> b l (h d)").contiguous()
                return out
            
        except Exception as e:
            print(f'Error in _perform_attention: {e}')
            raise e

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        source_view_mask: torch.Tensor | None = None,
        return_attn_weights: bool = False,
    ) -> torch.Tensor | tuple:
        
        ctx = context if context is not None else x

        q = self.to_q(x, source_view_mask=source_view_mask)
        k = self.to_k(ctx, source_view_mask=source_view_mask)
        v = self.to_v(ctx, source_view_mask=source_view_mask)

        if return_attn_weights:
            out, attn_weights = self._perform_attention(q, k, v, return_attn_weights=True)
        else:
            out = self._perform_attention(q, k, v, return_attn_weights=False)
        
        # Manually handle LoRA_Linear within Sequential
        out_linear, out_dropout = self.to_out[0], self.to_out[1]
        out = out_linear(out, source_view_mask=source_view_mask)
        out = out_dropout(out)
        
        if return_attn_weights:
            return out, attn_weights
        return out


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        d_head: int,
        context_dim: int,
        dropout: float = 0.0,
        lora_rank: int = 4,
    ):
        super().__init__()
        self.attn1 = Attention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout, lora_rank=lora_rank)
        self.ff = FeedForward(dim, dropout=dropout, lora_rank=lora_rank)
        self.attn2 = Attention(query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout, lora_rank=lora_rank)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        source_view_mask: torch.Tensor | None = None,
        save_attn_folder: str | None = None,
        denoise_step: int | None = None,
        block_name: str | None = None,
        original_images: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 判断是否需要保存 attention map
        should_save_attn = False # (save_attn_folder is not None and denoise_step is not None and block_name is not None)
        
        if should_save_attn:
            # Self-attention with attn weights
            attn1_out, attn1_weights = self.attn1(self.norm1(x), source_view_mask=source_view_mask, return_attn_weights=True)
            x = attn1_out + x
            self._save_attention_map(attn1_weights, save_attn_folder, denoise_step, f"{block_name}_selfattn", original_images)
            # Save t-SNE visualization of vision tokens
            # self._save_tsne_visualization(x, save_attn_folder, denoise_step, f"{block_name}_tsne")
            
            # Cross-attention with attn weights
            attn2_out, attn2_weights = self.attn2(self.norm2(x), context=context, source_view_mask=source_view_mask, return_attn_weights=True)
            x = attn2_out + x
            self._save_attention_map(attn2_weights, save_attn_folder, denoise_step, f"{block_name}_crossattn", original_images)
        else:
            # Normal forward without saving
            x = self.attn1(self.norm1(x), source_view_mask=source_view_mask) + x
            x = self.attn2(self.norm2(x), context=context, source_view_mask=source_view_mask) + x
        
        x = self.ff(self.norm3(x), source_view_mask=source_view_mask) + x
        return x
    
    def _save_attention_map(self, attn_weights: torch.Tensor, save_folder: str, step: int, name: str, original_images: torch.Tensor | None = None):
        """保存 attention map 为图像,并叠加到原始图像上"""
        import os
        from torchvision.utils import save_image
        import math
        import torch.nn.functional as F
        from PIL import Image
        import numpy as np
        import matplotlib.pyplot as plt
        
        # attn_weights shape: [b, h, l, l] where h is num_heads, l is seq_len
        # 对 heads 维度取平均
        attn_map = attn_weights.mean(dim=1)  # [b, l, l]
        
        bs, l, _ = attn_map.shape
        num_frames = 21
        spatial_tokens = l // num_frames  # l = 21 * h * w
        spatial_size = int(math.sqrt(spatial_tokens))  # sqrt(l/21)
        
        # 创建保存目录
        step_folder = os.path.join(save_folder, f"step_{step:04d}")
        os.makedirs(step_folder, exist_ok=True)
        
        # 调试信息
        if original_images is not None:
            print(f"[DEBUG] original_images shape: {original_images.shape}, dtype: {original_images.dtype}, min: {original_images.min():.3f}, max: {original_images.max():.3f}")
        else:
            print(f"[DEBUG] original_images is None!")
        
        # 如果有原始图像,准备用于叠加
        # original_images shape: [num_frames, c, H, W] (注意：只有21帧，不是 b*num_frames)
        if original_images is not None:
            # 将原始图像从 [num_frames, c, H, W] 移到 CPU 并转换到 [0, 1]
            orig_imgs = original_images.detach().cpu()
            # 如果图像范围是 [-1, 1], 转换到 [0, 1]
            if orig_imgs.min() < 0:
                orig_imgs = (orig_imgs + 1) / 2
            orig_imgs = orig_imgs.clamp(0, 1)
        
        # 遍历每个 batch (从0开始，因为bs是batch数量)
        for bs_id in range(1, bs):
            # 重塑 attn_map: [l, l] -> [21, h, w, 21, h, w]
            attn = attn_map[bs_id]  # [l, l]
            attn = attn.reshape(num_frames, spatial_size, spatial_size, num_frames, spatial_size, spatial_size)
            
            # 只保存特定的帧和空间位置，减少可视化数量
            selected_frames = [11, 13, 15, 17, 19]
            selected_h = [0, 4, 8]
            selected_w = [0, 4, 8]
            
            # 遍历选定的 query token
            for frame_id in selected_frames:
                if frame_id >= num_frames:
                    continue
                for h_id in selected_h:
                    if h_id >= spatial_size:
                        continue
                    for w_id in selected_w:
                        if w_id >= spatial_size:
                            continue
                        # 获取该 query token 对所有 key tokens 的 attention
                        query_attn = attn[frame_id, h_id, w_id, :, :, :]  # [21, h, w]
                        
                        # 归一化到 [0, 1]
                        query_attn = (query_attn - query_attn.min()) / (query_attn.max() - query_attn.min() + 1e-8)
                        
                        if original_images is not None:
                            # 将 attention map 叠加到原始图像上
                            overlayed_frames = []
                            original_frames = []
                            for target_frame_id in range(num_frames):
                                # 获取原始图像 (直接使用 target_frame_id，因为 orig_imgs 只有 num_frames 帧)
                                orig_img = orig_imgs[target_frame_id]  # [c, H, W]
                                H, W = orig_img.shape[1], orig_img.shape[2]
                                
                                # 保存原始图像用于后续保存
                                original_frames.append(orig_img)
                                
                                # 获取对应的 attention map
                                attn_frame = query_attn[target_frame_id]  # [h, w]
                                
                                # 上采样 attention map 到原始图像尺寸
                                attn_frame_upsampled = F.interpolate(
                                    attn_frame.unsqueeze(0).unsqueeze(0),  # [1, 1, h, w]
                                    size=(H, W),
                                    mode='bilinear',
                                    align_corners=False
                                )[0, 0]  # [H, W]
                                
                                # 转换为 numpy 用于 matplotlib colormap
                                attn_np = attn_frame_upsampled.numpy()
                                
                                # 使用 matplotlib 的 jet colormap 生成热力图
                                cmap = plt.get_cmap('jet')
                                heatmap = cmap(attn_np)[:, :, :3]  # [H, W, 3], 去掉 alpha 通道
                                heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float()  # [3, H, W]
                                
                                # 叠加: 0.5 * 原图 + 0.5 * 热力图
                                overlayed = 0.5 * orig_img + 0.5 * heatmap
                                overlayed = overlayed.clamp(0, 1)
                                overlayed_frames.append(overlayed)
                            
                            # 将所有帧拼接成网格
                            overlayed_grid = torch.stack(overlayed_frames)  # [21, 3, H, W]
                            original_grid = torch.stack(original_frames)  # [21, 3, H, W]
                            
                            # 保存叠加后的图像
                            save_path_overlay = os.path.join(step_folder, f"{name}_overlay_bs{bs_id}_f{frame_id:02d}_h{h_id:02d}_w{w_id:02d}.png")
                            save_image(overlayed_grid, save_path_overlay, nrow=num_frames)
                            
                            # 保存原始图像
                            save_path_original = os.path.join(step_folder, f"{name}_original_bs{bs_id}_f{frame_id:02d}_h{h_id:02d}_w{w_id:02d}.png")
                            save_image(original_grid, save_path_original, nrow=num_frames)
                        else:
                            # 如果没有原始图像,保存原始的 attention map
                            query_attn_vis = query_attn.unsqueeze(1)  # [21, 1, h, w]
                            save_path = os.path.join(step_folder, f"{name}_bs{bs_id}_f{frame_id:02d}_h{h_id:02d}_w{w_id:02d}.png")
                            save_image(query_attn_vis, save_path, nrow=num_frames)

    def _save_tsne_visualization(self, x: torch.Tensor, save_folder: str, step: int, name: str):
        """保存 vision token 的 t-SNE 可视化"""
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        import math
        
        # x shape: [b, l, d] where l = num_frames * h * w
        # 只取第一个 batch
        tokens = x[0].detach().cpu().numpy()  # [l, d]
        
        l, d = tokens.shape
        num_frames = 21
        spatial_tokens = l // num_frames
        spatial_size = int(math.sqrt(spatial_tokens))
        
        # 创建保存目录
        step_folder = os.path.join(save_folder, f"step_{step:04d}")
        os.makedirs(step_folder, exist_ok=True)
        
        # 执行 t-SNE 降维到 2D
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, l-1))
        tokens_2d = tsne.fit_transform(tokens)  # [l, 2]
        
        # 创建颜色映射：三类 latent 用不同颜色
        # 0-9: source latent, 10: cond latent, 11-20: target latent
        frame_indices = np.repeat(np.arange(num_frames), spatial_tokens)
        
        # 调试信息
        print(f"[t-SNE DEBUG] x shape: {x.shape}, tokens shape: {tokens.shape}")
        print(f"[t-SNE DEBUG] l={l}, num_frames={num_frames}, spatial_tokens={spatial_tokens}")
        print(f"[t-SNE DEBUG] frame_indices shape: {frame_indices.shape}, unique values: {np.unique(frame_indices)}")
        
        # 定义三类的颜色
        colors_map = {
            'source': '#1f77b4',  # 蓝色
            'cond': '#2ca02c',    # 绿色
            'target': '#d62728',  # 红色
        }
        
        # Source latent (frame 0-9): 10帧
        source_mask = frame_indices <= 9
        # Cond latent (frame 10): 1帧
        cond_mask = frame_indices == 10
        # Target latent (frame 11-20): 10帧
        target_mask = frame_indices >= 11
        
        print(f"[t-SNE DEBUG] source count: {source_mask.sum()}, cond count: {cond_mask.sum()}, target count: {target_mask.sum()}")
        
        # 绘制 t-SNE 图 - 使用随机顺序绘制避免覆盖
        plt.figure(figsize=(12, 10))
        
        # 创建颜色数组和随机索引
        colors = np.empty(l, dtype=object)
        colors[source_mask] = colors_map['source']
        colors[cond_mask] = colors_map['cond']
        colors[target_mask] = colors_map['target']
        
        # 随机打乱顺序绘制
        shuffle_idx = np.random.permutation(l)
        plt.scatter(tokens_2d[shuffle_idx, 0], tokens_2d[shuffle_idx, 1], 
                    c=colors[shuffle_idx], alpha=0.6, s=10)
        
        # 添加图例（手动创建）
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=colors_map['source'], label='Source Latent (0-9)'),
            Patch(facecolor=colors_map['cond'], label='Cond Latent (10)'),
            Patch(facecolor=colors_map['target'], label='Target Latent (11-20)')
        ]
        plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.title(f't-SNE of Vision Tokens - Step {step} - {name}')
        plt.xlabel('t-SNE dim 1')
        plt.ylabel('t-SNE dim 2')
        plt.tight_layout()
        
        # 保存图像
        save_path = os.path.join(step_folder, f"{name}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


class TransformerBlockTimeMix(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        d_head: int,
        context_dim: int,
        dropout: float = 0.0,
        lora_rank: int = 4,
    ):
        super().__init__()
        inner_dim = n_heads * d_head
        self.norm_in = nn.LayerNorm(dim)
        self.ff_in = FeedForward(dim, dim_out=inner_dim, dropout=dropout, lora_rank=lora_rank)
        self.attn1 = Attention(query_dim=inner_dim, heads=n_heads, dim_head=d_head, dropout=dropout, lora_rank=lora_rank)
        self.ff = FeedForward(inner_dim, dim_out=dim, dropout=dropout, lora_rank=lora_rank)
        self.attn2 = Attention(query_dim=inner_dim, context_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout, lora_rank=lora_rank)
        self.norm1 = nn.LayerNorm(inner_dim)
        self.norm2 = nn.LayerNorm(inner_dim)
        self.norm3 = nn.LayerNorm(inner_dim)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        num_frames: int,
        source_view_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _, s, _ = x.shape
        x_rearranged = rearrange(x, "(b t) s c -> (b s) t c", t=num_frames).contiguous()
        
        temp_mask = None
        if source_view_mask is not None:
            b = x.shape[0] // num_frames
            # source_view_mask shape is (b*t, 1, 1) or (b*t,), need to reshape properly
            temp_mask = source_view_mask.view(b, num_frames).unsqueeze(-1)  # (b, t, 1)
            temp_mask = temp_mask.unsqueeze(1).repeat(1, s, 1, 1).reshape(-1, num_frames, 1)  # (b*s, t, 1)

        ff_in_out = self.ff_in(self.norm_in(x_rearranged), source_view_mask=temp_mask) + x_rearranged
        attn1_out = self.attn1(self.norm1(ff_in_out), context=None, source_view_mask=temp_mask) + ff_in_out
        attn2_out = self.attn2(self.norm2(attn1_out), context=context, source_view_mask=temp_mask) + attn1_out
        ff_out = self.ff(self.norm3(attn2_out), source_view_mask=temp_mask)
        
        return rearrange(ff_out, "(b s) t c -> (b t) s c", s=s).contiguous()


class SkipConnect(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_spatial: torch.Tensor, x_temporal: torch.Tensor) -> torch.Tensor:
        return x_spatial + x_temporal


class MultiviewTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_heads: int,
        d_head: int,
        name: str,
        unflatten_names: list[str] = [],
        depth: int = 1,
        context_dim: int = 1024,
        dropout: float = 0.0,
        lora_rank: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.name = name
        self.unflatten_names = unflatten_names

        inner_dim = n_heads * d_head
        self.norm = nn.GroupNorm(32, in_channels, eps=1e-6)
        self.proj_in = LoRA_Linear(in_channels, inner_dim, rank=lora_rank)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(inner_dim, n_heads, d_head, context_dim=context_dim, dropout=dropout, lora_rank=lora_rank) for _ in range(depth)]
        )
        self.proj_out = LoRA_Linear(inner_dim, in_channels, rank=lora_rank)
        self.time_mixer = SkipConnect()
        self.time_mix_blocks = nn.ModuleList(
            [TransformerBlockTimeMix(inner_dim, n_heads, d_head, context_dim=context_dim, dropout=dropout, lora_rank=lora_rank) for _ in range(depth)]
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        num_frames: int,
        source_view_mask: torch.Tensor,
        save_attn_folder: str | None = None,
        denoise_step: int | None = None,
        original_images: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert context.ndim == 3
        b, c, h, w = x.shape
        x_in = x

        time_context = context
        time_context_first_timestep = time_context[::num_frames]
        time_context = repeat(time_context_first_timestep, "b ... -> (b n) ...", n=h * w)

        if self.name in self.unflatten_names:
            context = context[::num_frames]

        x = self.norm(x)
        x_rearranged = rearrange(x, "b c h w -> b (h w) c").contiguous()
        source_view_mask_rearranged = repeat(source_view_mask, "b c h w -> b (h w) c").contiguous() # (b*num_frames 1 1 1) -> (b*num_frames 1 1)
        x_projected = self.proj_in(x_rearranged,source_view_mask_rearranged )

        current_x = x_projected # [(b t) (h w) c]
        current_mask = source_view_mask_rearranged # [(b t) 1 1]
        for block_idx, (block, mix_block) in enumerate(zip(self.transformer_blocks, self.time_mix_blocks)):
            if self.name in self.unflatten_names:
                current_x = rearrange(current_x, "(b t) (h w) c -> b (t h w) c", t=num_frames, h=h, w=w).contiguous()
                current_mask = repeat(current_mask, "(b t) 1 1 -> b (t h w) 1", t=num_frames, h=h, w=w).contiguous()

            # 只在 unflatten_names 中的 block 才保存 attention maps
            block_name = f"{self.name}_block{block_idx}"
            if save_attn_folder is not None and self.name in self.unflatten_names:
                # 传递原始图像用于可视化叠加
                current_x = block(current_x, context, current_mask, save_attn_folder, denoise_step, block_name, original_images)
            else:
                current_x = torch.utils.checkpoint.checkpoint(block, current_x, context, current_mask, use_reentrant=False)
            
            if self.name in self.unflatten_names:
                current_x = rearrange(current_x, "b (t h w) c -> (b t) (h w) c", t=num_frames, h=h, w=w).contiguous()
                current_mask = rearrange(current_mask, "b (t h w) 1 -> (b t) (h w) 1", t=num_frames, h=h, w=w).mean(1, keepdim=True).contiguous() # (b*num_frames 1 1)

            x_mix = torch.utils.checkpoint.checkpoint(mix_block, current_x, time_context, num_frames, current_mask, use_reentrant=False)
            current_x = self.time_mixer(x_spatial=current_x, x_temporal=x_mix)

        x_projected_out = self.proj_out(current_x, source_view_mask=current_mask)
        x_rearranged_out = rearrange(x_projected_out, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        return x_rearranged_out + x_in


if __name__ == "__main__":
    batch_size = 1
    num_frames = 21
    n_heads = 16
    d_head = 64
    name = "middle_ds8"
    depth = 1
    context_dim = 1024
    height = 32
    width = 32
    dropout = 0.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if device.type == "cuda" else torch.float32
    lora_rank_test = 8
    
    mvtransformer = MultiviewTransformer(
        in_channels=context_dim, 
        n_heads=n_heads, 
        d_head=d_head, 
        name=name, 
        depth=depth, 
        context_dim=context_dim, 
        dropout=dropout,
        lora_rank=lora_rank_test,
    ).to(device, dtype)
    
    print(mvtransformer)

    x = torch.randn(batch_size*num_frames, context_dim, height, width, device=device, dtype=dtype)
    context = torch.randn(batch_size*num_frames, 1, context_dim, device=device, dtype=dtype)
    
    source_view_mask = (torch.rand(batch_size * num_frames, 1,  1, 1, device=device) > 0.5).to(dtype)

    print(f"\nTesting with {int(source_view_mask.sum())} source views and {int((1-source_view_mask).sum())} target views.")

    out = mvtransformer(x, context, num_frames, source_view_mask=source_view_mask)
    print(f"\nOutput shape: {out.shape}")
    assert out.shape == x.shape, f"Output shape mismatch: {out.shape} vs {x.shape}"
    print("Manual Dual LoRA test finished successfully!")