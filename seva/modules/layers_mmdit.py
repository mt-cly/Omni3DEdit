import math

import torch
import torch.nn.functional as F
from einops import repeat
from torch import nn

from seva.modules.transformer_mmdit import MultiviewTransformer, LoRA_Linear, LoRA_Conv2d, load_pretrained_weights_with_lora


def timestep_embedding(
    timesteps: torch.Tensor,
    dim: int,
    max_period: int = 10000,
    repeat_only: bool = False,
) -> torch.Tensor:
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
    else:
        embedding = repeat(timesteps, "b -> b d", d=dim)
    return embedding


class Upsample(nn.Module):
    def __init__(self, channels: int, out_channels: int | None = None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.conv = nn.Conv2d(self.channels, self.out_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, channels: int, out_channels: int | None = None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.op = nn.Conv2d(self.channels, self.out_channels, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.channels
        return self.op(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # return super().forward(input.float()).type(input.dtype)
        return super().forward(input)


class TimestepEmbedSequential(nn.Sequential):
    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
        context: torch.Tensor,
        dense_emb: torch.Tensor,
        num_frames: int,
        source_view_mask: torch.Tensor,
        save_attn_folder: str | None = None,
        denoise_step: int | None = None,
        original_images: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, MultiviewTransformer):
                assert num_frames is not None
                x = layer(x, context, num_frames, source_view_mask, save_attn_folder, denoise_step, original_images)
            elif isinstance(layer, ResBlock):
                x = layer(x, emb, dense_emb, source_view_mask)
            else:
                x = layer(x)
        return x


class ResBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        emb_channels: int,
        out_channels: int | None,
        dense_in_channels: int,
        dropout: float,
        lora_rank: int = 4,
    ):
        super().__init__()
        out_channels = out_channels or channels

        self.in_layers = nn.Sequential(
            GroupNorm32(32, channels),
            nn.SiLU(),
            LoRA_Conv2d(channels, out_channels, 3, 1, 1, rank=lora_rank),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(), LoRA_Linear(emb_channels, out_channels, rank=lora_rank)
        )
        self.dense_emb_layers = nn.Sequential(
            LoRA_Conv2d(dense_in_channels, 2 * channels, 1, 1, 0, rank=lora_rank)
        )
        self.out_layers = nn.Sequential(
            GroupNorm32(32, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            LoRA_Conv2d(out_channels, out_channels, 3, 1, 1, rank=lora_rank),
        )
        if out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = LoRA_Conv2d(channels, out_channels, 1, 1, 0, rank=lora_rank)

    def forward(
        self, x: torch.Tensor, emb: torch.Tensor, dense_emb: torch.Tensor, source_view_mask: torch.Tensor
    ) -> torch.Tensor:
        # Handle sequential layers with LoRA manually
        in_norm, in_silu, in_conv = self.in_layers[0], self.in_layers[1], self.in_layers[2]
        h = in_norm(x)
        h = in_silu(h)
        
        # Interpolate dense embedding and apply LoRA
        dense_interp = F.interpolate(
            dense_emb, size=h.shape[2:], mode="bilinear", align_corners=True
        ).type(h.dtype)
        dense = self.dense_emb_layers[0](dense_interp, source_view_mask=source_view_mask)
        dense_scale, dense_shift = torch.chunk(dense, 2, dim=1)
        h = h * (1 + dense_scale) + dense_shift
        
        # Apply input conv with LoRA
        h = in_conv(h, source_view_mask=source_view_mask)
        
        # Apply embedding layers with LoRA
        emb_silu, emb_linear = self.emb_layers[0], self.emb_layers[1]
        emb_out = emb_silu(emb)
        # Prepare mask for linear layer (needs to match embedding dimensions)
        emb_mask = source_view_mask.squeeze().unsqueeze(-1)  # (b, 1)
        emb_out = emb_linear(emb_out, source_view_mask=emb_mask).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        h = h + emb_out
        
        # Apply output layers with LoRA
        out_norm, out_silu, out_dropout, out_conv = self.out_layers[0], self.out_layers[1], self.out_layers[2], self.out_layers[3]
        h = out_norm(h)
        h = out_silu(h)
        h = out_dropout(h)
        h = out_conv(h, source_view_mask=source_view_mask)
        
        # Apply skip connection with LoRA if needed
        if isinstance(self.skip_connection, LoRA_Conv2d):
            skip_out = self.skip_connection(x, source_view_mask=source_view_mask)
        else:
            skip_out = self.skip_connection(x)
            
        return skip_out + h


if __name__ == "__main__":
    # Test LoRA ResBlock and parameter compatibility
    print("Testing LoRA ResBlock and parameter name compatibility...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    
    batch_size = 4
    num_frames = 21
    channels = 64
    emb_channels = 128
    dense_in_channels = 6
    height = 32
    width = 32
    lora_rank = 8
    
    # Test LoRA_Linear parameter compatibility
    print("\n=== Testing LoRA_Linear ===")
    lora_linear = LoRA_Linear(128, 64, rank=lora_rank)
    standard_linear = nn.Linear(128, 64)
    
    print("LoRA_Linear parameters:")
    for name, param in lora_linear.named_parameters():
        print(f"  {name}: {param.shape}")
    
    print("\nStandard Linear parameters:")
    for name, param in standard_linear.named_parameters():
        print(f"  {name}: {param.shape}")
    
    # Test parameter loading compatibility
    print("\n=== Testing Parameter Loading ===")
    standard_state = standard_linear.state_dict()
    lora_state = lora_linear.state_dict()
    
    # Check if standard parameters exist in LoRA model
    compatible_params = []
    for name in standard_state.keys():
        if name in lora_state and standard_state[name].shape == lora_state[name].shape:
            compatible_params.append(name)
    
    print(f"Compatible parameters: {compatible_params}")
    
    # Test loading standard weights into LoRA model
    try:
        lora_linear.load_state_dict(standard_state, strict=False)
        print("✓ Successfully loaded standard weights into LoRA model")
    except Exception as e:
        print(f"✗ Failed to load standard weights: {e}")
    
    # Test ResBlock
    print("\n=== Testing LoRA ResBlock ===")
    # Create test inputs
    x = torch.randn(batch_size * num_frames, channels, height, width, device=device, dtype=dtype)
    emb = torch.randn(batch_size * num_frames, emb_channels, device=device, dtype=dtype)
    dense_emb = torch.randn(batch_size * num_frames, dense_in_channels, height//2, width//2, device=device, dtype=dtype)
    source_view_mask = (torch.rand(batch_size * num_frames, 1, 1, 1, device=device) > 0.5).to(dtype)
    
    # Create ResBlock with LoRA
    resblock = ResBlock(
        channels=channels,
        emb_channels=emb_channels,
        out_channels=channels,
        dense_in_channels=dense_in_channels,
        dropout=0.1,
        lora_rank=lora_rank
    ).to(device, dtype)
    
    total_params = sum(p.numel() for p in resblock.parameters())
    lora_params = sum(p.numel() for p in resblock.parameters() if any(x in str(p) for x in ['lora_A1', 'lora_A2', 'lora_B1', 'lora_B2']))
    base_params = total_params - lora_params
    
    print(f"Total parameters: {total_params:,}")
    print(f"Base parameters: {base_params:,}")
    print(f"LoRA parameters: {lora_params:,}")
    print(f"LoRA overhead: {lora_params/base_params:.2%}")
    
    # Test forward pass
    with torch.no_grad():
        output = resblock(x, emb, dense_emb, source_view_mask)
        print(f"\nInput shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output mean: {output.mean().item():.6f}")
        print(f"Output std: {output.std().item():.6f}")
        
        # Test with different masks
        all_source_mask = torch.ones_like(source_view_mask)
        all_target_mask = torch.zeros_like(source_view_mask)
        
        output_source = resblock(x, emb, dense_emb, all_source_mask)
        output_target = resblock(x, emb, dense_emb, all_target_mask)
        
        diff = (output_source - output_target).abs().mean()
        print(f"Difference between source and target LoRA: {diff.item():.6f}")
        
    print("\n✓ All tests completed successfully!")