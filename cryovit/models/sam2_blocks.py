"""Transformer-based prompt predictor for SAM2 using image encodings and learnable prompt tokens. Based on the prompt predictor in https://github.com/aswahd/SamRadiology/"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.modeling.sam.transformer import Attention
from torch import Tensor


class PromptConvBlock(nn.Module):
    """A convolutional block used in the prompt predictor."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(
                in_channels, out_channels, kernel_size=3, padding="same"
            ),
            nn.InstanceNorm3d(out_channels, eps=1e-3, affine=True),
            nn.GELU(),
            nn.Conv3d(
                out_channels, out_channels, kernel_size=3, padding="same"
            ),
            nn.InstanceNorm3d(out_channels, eps=1e-3, affine=True),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class PromptAnalysisBlock(nn.Module):
    """A single block in the down-sampling UNet architecture for prompt prediction."""

    def __init__(self, in_channels: int, out_channels: int, scale: int = 2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.MaxPool3d(scale),
            PromptConvBlock(in_channels, out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class PromptSynthesisBlock(nn.Module):
    """A single block in the up-sampling UNet architecture for prompt prediction."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        scale: int = 2,
    ):
        super().__init__()
        self.upconv = nn.ConvTranspose3d(
            in_channels, in_channels, kernel_size=scale, stride=scale
        )
        self.layers = nn.Sequential(
            PromptConvBlock(in_channels + skip_channels, out_channels),
            PromptConvBlock(out_channels, out_channels),
        )
        self.scale = scale

    def forward(self, x: Tensor, skip_x: Tensor) -> Tensor:
        x = self.upconv(x)
        x = F.interpolate(
            x, size=skip_x.shape[-3:], mode="trilinear", align_corners=True
        )  # handle size mismatches
        x = torch.cat([x, skip_x], dim=1)
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
        depth: int = 3,
        layer_scale: int = 2,
        channel_dims: list[int] | None = None,
    ):
        if channel_dims is None:
            channel_dims = [64, 128, 256]
        super().__init__()
        assert depth == len(
            channel_dims
        ), "Depth must match the length of channel_dims"
        self.scale_factor = 4  # Scale factor for mask prediction
        # Original SAM2 has 4x patch embedding

        self.init_conv = PromptConvBlock(in_channels, channel_dims[0])

        self.down_layers = nn.ModuleList(
            [
                PromptAnalysisBlock(
                    channel_dims[i], channel_dims[i + 1], scale=layer_scale
                )
                for i in range(depth - 1)
            ]
        )

        self.up_layers = nn.ModuleList(
            [
                PromptSynthesisBlock(
                    channel_dims[-i],
                    channel_dims[-i - 1],
                    channel_dims[-i - 1],
                    scale=layer_scale,
                )
                for i in range(1, depth)
            ]
        )

        self.prompt_out = nn.Conv3d(
            channel_dims[0], 1, kernel_size=1, padding="same"
        )
        self.box_out = PromptBoxPredictor(channel_dims[0])

    def forward(self, x: Tensor, num_batches: int) -> tuple[Tensor, Tensor]:
        BD, C, H, W = x.shape
        x = x.view(num_batches, -1, C, H, W).transpose(1, 2)  # (B, C, D, H, W)
        x = self.init_conv(x)  # (B, C', D, H, W)

        outputs = []

        for block in self.down_layers:
            outputs.append(x)
            x = block(x)
        for layer, skip_x in zip(
            self.up_layers, reversed(outputs), strict=False
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
        prompt_outs = torch.sigmoid(prompt_outs)  # Normalize to [0, 1]
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
