"""3D U-Net-based prompt predictor for SAM2 using image encodings and LoRA adaptation modules.

Prompt predictor architecture is based on the prompt encoder U-Net in https://github.com/ChengyinLee/AutoProSAM_2024/ and https://github.com/aswahd/SamRadiology/tree/main.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.modeling.sam.transformer import Attention
from torch import Tensor


class PromptConvBlock(nn.Module):
    """A basic convolutional block used in the prompt predictor."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        bias: bool = False,
        norm=nn.InstanceNorm3d,
        act=nn.GELU,
        preact: bool = False,
    ):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=k_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.norm = (
            norm(in_channels if preact else out_channels)
            if norm
            else nn.Identity()
        )
        self.act = act() if act else nn.Identity()
        self.preact = preact

    def forward(self, x: Tensor) -> Tensor:
        if self.preact:
            x = self.act(self.norm(x))
        x = self.conv(x)
        if not self.preact:
            x = self.act(self.norm(x))
        return x


class PromptInConv(nn.Module):
    """The input convolutional block used in the prompt predictor."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            PromptConvBlock(in_channels, out_channels),
            PromptConvBlock(out_channels, out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class PromptDownBlock(nn.Module):
    """A single block in the down-sampling UNet architecture for prompt prediction."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_blocks: int = 2,
        scale: int = 2,
    ):
        super().__init__()
        block_list = []
        block_list.append(nn.MaxPool3d(scale))
        for _ in range(n_blocks):
            block_list.append(PromptConvBlock(in_channels, out_channels))
            in_channels = out_channels
        self.layers = nn.Sequential(*block_list)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class PromptUpBlock(nn.Module):
    """A single block in the up-sampling UNet architecture for prompt prediction."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        n_blocks: int = 2,
        scale: int = 2,
    ):
        super().__init__()
        block_list = [
            PromptConvBlock(in_channels + skip_channels, out_channels)
        ]
        for _ in range(n_blocks - 1):
            block_list.append(PromptConvBlock(out_channels, out_channels))
        self.layers = nn.Sequential(*block_list)
        self.scale = scale

    def forward(self, x: Tensor, skip_x: Tensor) -> Tensor:
        x = F.interpolate(
            x, size=skip_x.shape[-3:], mode="trilinear", align_corners=True
        )  # handle size mismatches
        x = torch.cat([skip_x, x], dim=1)
        x = self.layers(x)
        return x


class PromptBoxPredictor(nn.Module):
    """A simple box predictor using a linear layer."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(
            in_channels, 4
        )  # 4 channels for box (x1, y1, x2, y2)

    def forward(self, x: Tensor) -> Tensor:
        # Global average pooling
        B, C, D, H, W = x.shape
        x = self.pool(x)  # [B, C, D, 1, 1]
        x = x.transpose(1, 2).view(B * D, -1, 1, 1)  # [B*D, C, 1, 1]
        x = x.flatten(1)  # [B*D, C]
        x = self.fc(x)  # [B*D, 4]
        x = torch.sigmoid(x)  # Normalize to [0, 1]
        x1y1 = x[:, :2]
        x2y2 = x[:, 2:] + x1y1  # Ensure x2y2 >= x1y1
        x = torch.cat([x1y1, x2y2], dim=1)  # [B*D, 4]
        return x


class PromptPredictor(nn.Module):
    """A simple UNet to predict mask prompts for the SAMv2 model from image encodings."""

    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 16,
        depth: int = 4,
        layer_scale: int = 2,
        channel_mults: list[int] | None = None,
    ):
        if channel_mults is None:
            channel_mults = [1, 2, 4, 8, 10]
        super().__init__()
        assert depth + 1 == len(
            channel_mults
        ), "Depth must match the length of channel multipliers - 1"
        self.scale_factor = 4  # Scale factor for mask prediction
        # Original SAM2 has 4x patch embedding

        self.init_conv = PromptInConv(in_channels, hidden_channels)

        self.down_layers = nn.ModuleList(
            [
                PromptDownBlock(
                    channel_mults[i] * hidden_channels,
                    channel_mults[i + 1] * hidden_channels,
                    scale=layer_scale,
                )
                for i in range(depth)
            ]
        )

        self.up_layers = nn.ModuleList(
            [
                PromptUpBlock(
                    channel_mults[i + 1] * hidden_channels,
                    channel_mults[i] * hidden_channels,
                    channel_mults[i] * hidden_channels,
                    scale=layer_scale,
                )
                for i in reversed(range(depth))
            ]
        )

        self.prompt_out = nn.Conv3d(
            channel_mults[0] * hidden_channels,
            1,
            kernel_size=1,
            padding="same",
        )
        self.box_out = PromptBoxPredictor(channel_mults[0] * hidden_channels)

    def forward(self, x: Tensor, num_batches: int) -> tuple[Tensor, Tensor]:
        BD, C, H, W = x.shape
        x = x.view(num_batches, -1, C, H, W).transpose(1, 2)  # (B, C, D, H, W)
        x = self.init_conv(x)  # (B, C', D, H, W)

        outputs = []

        for block in self.down_layers:
            outputs.append(x)
            x = block(x)
        for layer, skip_x in zip(
            self.up_layers, reversed(outputs), strict=True
        ):
            x = layer(x, skip_x)

        prompt_outs = self.prompt_out(x)  # (B, 1, D, H, W)
        prompt_outs = prompt_outs.view(BD, 1, H, W)  # (B*D, 1, H, W)
        prompt_outs = F.interpolate(
            prompt_outs,
            scale_factor=self.scale_factor,
            mode="bilinear",
            align_corners=True,
        )  # Upscale to original size
        box_outs = self.box_out(x)  # (B*D, 4)
        return box_outs, prompt_outs


class LoRALinear(nn.Module):
    """A linear layer with LoRA (Low-Rank Adaptation) applied."""

    def __init__(
        self, proj: nn.Module, input_dim: int, output_dim: int, r: int, a: int
    ):
        super().__init__()
        self.proj = proj
        self.r = r
        self.a = a
        self.scaling = a / r
        self.w_a = nn.Linear(input_dim, r, bias=False)
        self.w_b = nn.Linear(r, output_dim, bias=False)

        self.initialize_parameters()

    def initialize_parameters(self):
        """Initializes LoRA parameters."""
        nn.init.kaiming_uniform_(self.w_a.weight, a=np.sqrt(5))
        nn.init.zeros_(self.w_b.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x) + self.w_b(self.w_a(x)) * self.scaling


class LoRAMaskDecoderFactory:
    """Module to apply LoRA to the transformer blocks in the SAM MaskDecoder."""

    def __init__(self, lora_r: int = 32, lora_alpha: int = 64):
        self.r = lora_r
        self.a = lora_alpha

    def _apply_lora(self, attn_block: Attention) -> None:
        """Applies LoRA to the attention block."""
        in_features = attn_block.embedding_dim
        out_features = attn_block.internal_dim

        # Original projection layer
        q_proj = attn_block.q_proj
        v_proj = attn_block.v_proj

        # Initialize LoRA layers
        attn_block.q_proj = LoRALinear(q_proj, in_features, out_features, self.r, self.a)  # type: ignore
        attn_block.v_proj = LoRALinear(v_proj, in_features, out_features, self.r, self.a)  # type: ignore

    def apply(self, mask_decoder: nn.Module) -> nn.Module:
        """Applies LoRA to all transformer blocks in the MaskDecoder."""

        for p in mask_decoder.parameters():
            p.requires_grad = False

        transformer = mask_decoder.transformer

        for _, blk in enumerate(transformer.layers):  # type: ignore
            self._apply_lora(blk.self_attn)
            self._apply_lora(blk.cross_attn_token_to_image)
            self._apply_lora(blk.cross_attn_image_to_token)

        self._apply_lora(transformer.final_attn_token_to_image)  # type: ignore

        return mask_decoder
