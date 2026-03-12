from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torchvision.utils import save_image
from seva.modules.layers_mmdit import (
    Downsample,
    GroupNorm32,
    ResBlock,
    TimestepEmbedSequential,
    Upsample,
    timestep_embedding,
)
from seva.modules.transformer_mmdit import MultiviewTransformer

    
@dataclass
class SevaParams(object):
    in_channels: int = 11
    model_channels: int = 320
    out_channels: int = 4
    num_frames: int = 21
    num_res_blocks: int = 2
    attention_resolutions: list[int] = field(default_factory=lambda: [4, 2, 1])
    channel_mult: list[int] = field(default_factory=lambda: [1, 2, 4, 4])
    num_head_channels: int = 64
    transformer_depth: list[int] = field(default_factory=lambda: [1, 1, 1, 1])
    context_dim: int = 1024
    dense_in_channels: int = 6
    dropout: float = 0.0
    unflatten_names: list[str] = field(
        default_factory=lambda: ["middle_ds8", "output_ds4", "output_ds2"]
    )
    lora_rank: int = 8

    def __post_init__(self):
        assert len(self.channel_mult) == len(self.transformer_depth)


class Seva(nn.Module):
    def __init__(self, params: SevaParams) -> None:
        super().__init__()
        self.params = params
        self.model_channels = params.model_channels
        self.out_channels = params.out_channels
        self.num_head_channels = params.num_head_channels

        time_embed_dim = params.model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(params.model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    nn.Conv2d(params.in_channels, params.model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = params.model_channels
        input_block_chans = [params.model_channels]
        ch = params.model_channels
        ds = 1
        for level, mult in enumerate(params.channel_mult):
            for _ in range(params.num_res_blocks):
                input_layers: list[ResBlock | MultiviewTransformer | Downsample] = [
                    ResBlock(
                        channels=ch,
                        emb_channels=time_embed_dim,
                        out_channels=mult * params.model_channels,
                        dense_in_channels=params.dense_in_channels,
                        dropout=params.dropout,
                    )
                ]
                ch = mult * params.model_channels
                if ds in params.attention_resolutions:
                    num_heads = ch // params.num_head_channels
                    dim_head = params.num_head_channels
                    input_layers.append(
                        MultiviewTransformer(
                            ch,
                            num_heads,
                            dim_head,
                            name=f"input_ds{ds}",
                            depth=params.transformer_depth[level],
                            context_dim=params.context_dim,
                            unflatten_names=params.unflatten_names,
                            lora_rank=params.lora_rank,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*input_layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(params.channel_mult) - 1:
                ds *= 2
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, out_channels=out_ch))
                )
                ch = out_ch
                input_block_chans.append(ch)
                self._feature_size += ch

        num_heads = ch // params.num_head_channels
        dim_head = params.num_head_channels

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                channels=ch,
                emb_channels=time_embed_dim,
                out_channels=None,
                dense_in_channels=params.dense_in_channels,
                dropout=params.dropout,
            ),
            MultiviewTransformer(
                ch,
                num_heads,
                dim_head,
                name=f"middle_ds{ds}",
                depth=params.transformer_depth[-1],
                context_dim=params.context_dim,
                unflatten_names=params.unflatten_names,
            ),
            ResBlock(
                channels=ch,
                emb_channels=time_embed_dim,
                out_channels=None,
                dense_in_channels=params.dense_in_channels,
                dropout=params.dropout,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(params.channel_mult))[::-1]:
            for i in range(params.num_res_blocks + 1):
                ich = input_block_chans.pop()
                output_layers: list[ResBlock | MultiviewTransformer | Upsample] = [
                    ResBlock(
                        channels=ch + ich,
                        emb_channels=time_embed_dim,
                        out_channels=params.model_channels * mult,
                        dense_in_channels=params.dense_in_channels,
                        dropout=params.dropout,
                    )
                ]
                ch = params.model_channels * mult
                if ds in params.attention_resolutions:
                    num_heads = ch // params.num_head_channels
                    dim_head = params.num_head_channels

                    output_layers.append(
                        MultiviewTransformer(
                            ch,
                            num_heads,
                            dim_head,
                            name=f"output_ds{ds}",
                            depth=params.transformer_depth[level],
                            context_dim=params.context_dim,
                            unflatten_names=params.unflatten_names
                        )
                    )
                if level and i == params.num_res_blocks:
                    out_ch = ch
                    ds //= 2
                    output_layers.append(Upsample(ch, out_ch))
                self.output_blocks.append(TimestepEmbedSequential(*output_layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            GroupNorm32(32, ch),
            nn.SiLU(),
            nn.Conv2d(self.model_channels, params.out_channels, 3, padding=1),
        )

        # todo liyi 当前为全参+lora都是TRAINABLE，后续需要修改
        # 设置参数的可训练状态 
        # self._setup_trainable_parameters()


    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        dense_y: torch.Tensor,
        num_frames: int | None = None,
        source_view_mask: torch.Tensor | None = None,
        save_attn_folder: str | None = None,
        denoise_step: int | None = None,
        original_images: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_frames = num_frames or self.params.num_frames
        t_emb = timestep_embedding(t, self.model_channels).type(x.dtype)
        t_emb = self.time_embed(t_emb)
        hs = []
        h = x
        for input_idx, module in enumerate(self.input_blocks):
            h = module(
                h,
                emb=t_emb,
                context=y,
                dense_emb=dense_y,
                num_frames=num_frames,
                source_view_mask=source_view_mask,
                save_attn_folder=save_attn_folder,
                denoise_step=denoise_step,
                original_images=original_images,
            )
            hs.append(h)
        h = self.middle_block(
            h,
            emb=t_emb,
            context=y,
            dense_emb=dense_y,
            num_frames=num_frames,
            source_view_mask=source_view_mask,
            save_attn_folder=save_attn_folder,
            denoise_step=denoise_step,
            original_images=original_images,
        )
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(
                h,
                emb=t_emb,
                context=y,
                dense_emb=dense_y,
                num_frames=num_frames,
                source_view_mask=source_view_mask,
                save_attn_folder=None,  # output_blocks 不保存 attention maps
                denoise_step=None,
                original_images=None,
            )
        h = h.type(x.dtype)
        return self.out(h)


class SGMWrapper(nn.Module):
    def __init__(self, module: Seva):
        super().__init__()
        self.module = module

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: dict, **kwargs
    ) -> torch.Tensor:
        x = torch.cat((x, c.get("concat", torch.Tensor([]).type_as(x))), dim=1)
        return self.module(
            x,
            t=t,
            y=c["crossattn"],
            dense_y=c["dense_vector"],
            **kwargs,
        )


if __name__ == "__main__":
    params = SevaParams()
    model = Seva(params)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    model = SGMWrapper(model).to(device, dtype)
    bs = 4
    num_frames = 21
    h, w = 64, 64
    x = torch.randn(bs * num_frames, 4, h, w, device=device, dtype=dtype)
    c = {"concat": torch.randn(bs * num_frames, 7, h, w, device=device, dtype=dtype),
         "crossattn": torch.randn(bs * num_frames, 1, 1024, device=device, dtype=dtype), 
         "dense_vector": torch.randn(bs * num_frames, 6, h, w, device=device, dtype=dtype)}
    t = torch.randint(0, 1000, (bs * num_frames,), device=device, dtype=torch.long)
    source_view_mask = (torch.rand(bs * num_frames, 1, 1, 1, device=device) > 0.5).to(dtype)
    output = model(x, t, c, num_frames=num_frames, source_view_mask=source_view_mask)
    print(output.shape)
