"""UNet3D model architecture for 3D tomogram segmentation."""

import math

import torch
from torch import Tensor, nn

from cryovit.models.base_model import BaseModel
from cryovit.types import BatchedTomogramData


class UNet3D(BaseModel):
    """UNet3D model implementation."""

    def __init__(self, **kwargs) -> None:
        """Initializes the UNet3D model with specific analysis and synthesis blocks."""
        super().__init__(**kwargs)

        self.bottom_layer = nn.Sequential(
            nn.Conv3d(256, 384, 3, padding="same"),
            nn.InstanceNorm3d(384, eps=1e-3, affine=True),
            nn.GELU(),
            nn.Conv3d(384, 256, 3, padding="same"),
            nn.InstanceNorm3d(256, eps=1e-3, affine=True),
            nn.GELU(),
        )

        self.analysis_layers = nn.ModuleList(
            [
                AnalysisBlock(1, 16),
                AnalysisBlock(16, 64),
                AnalysisBlock(64, 256),
            ]
        )

        self.synthesis_layers = nn.ModuleList(
            [
                SynthesisBlock(256, 256, 64),
                SynthesisBlock(64, 64, 16),
                SynthesisBlock(16, 16, 16),
            ]
        )

        self.output_layer = nn.Conv3d(16, 1, 1, padding="same")
        self.PAD = max(16, 2 ** len(self.analysis_layers))

    def forward_volume(self, x: Tensor) -> Tensor:
        """Memory optimized forward pass for the UNet3D model."""
        analysis_outputs = []

        for block in self.analysis_layers:
            x, prev_x = block(x)
            analysis_outputs.append(prev_x)

        for layer in self.bottom_layer:
            x = layer(x)

        for block in self.synthesis_layers:
            for layer in block.upconv:  # type: ignore
                x = layer(x)

            x = torch.cat([x, analysis_outputs.pop()], 1)

            for layer in block.layers:  # type: ignore
                x = layer(x)

        x = self.output_layer(x)
        x = torch.clip(x, -5.0, 5.0)
        return x

    def forward(self, batch: BatchedTomogramData) -> Tensor:  # type: ignore
        x = batch.tomo_batch  # (B, D, C, H, W)
        x = x.permute(0, 2, 1, 3, 4)  # (B, C, D, H, W)

        # Add padding
        new_size = [
            self.PAD * math.ceil(dim / self.PAD) for dim in x.size()[-3:]
        ]
        D, H, W = x.size()[-3:]
        if new_size != [D, H, W]:
            x = self._add_padding(x, new_size)

        x = self.forward_volume(x)  # (B, 1, D, H, W)

        # Remove padding
        if new_size != [D, H, W]:
            x = x[..., :D, :H, :W]

        x = x.squeeze(1)  # (B, D, H, W)
        return torch.sigmoid(x)

    @torch.inference_mode()
    def _add_padding(self, x: Tensor, new_size: list[int]) -> Tensor:
        """Adds padding to the input to match the U-Net dimensions"""
        # pad to multiple of self.PAD
        D, H, W = x.size()[-3:]
        new_shape = list(x.shape[:-3]) + new_size
        x_new = torch.zeros(*new_shape, dtype=x.dtype, device=x.device)
        x_new[..., :D, :H, :W] = x

        return x_new


class AnalysisBlock(nn.Module):
    """Block that performs down-sampling and feature extraction in UNet3D's analysis path."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initializes the AnalysisBlock with convolutional layers and pooling.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super().__init__()
        self.pool = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 2, stride=2),
            nn.InstanceNorm3d(out_channels, eps=1e-3, affine=True),
            nn.GELU(),
        )

        self.layers = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding="same"),
            nn.InstanceNorm3d(out_channels, eps=1e-3, affine=True),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, 3, padding="same"),
            nn.InstanceNorm3d(out_channels, eps=1e-3, affine=True),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass for the analysis block."""
        x = self.layers(x)
        y = self.pool(x)
        return y, x


class SynthesisBlock(nn.Module):
    """Block that performs up-sampling and feature combination in UNet3D's synthesis path."""

    def __init__(
        self, in_channels: int, skip_channels: int, out_channels: int
    ) -> None:
        """Initializes the SynthesisBlock with up-sampling and convolutional layers.

        Args:
            in_channels (int): Number of input channels.
            skip_channels (int): Number of channels from the skip connection.
            out_channels (int): Number of output channels.
        """
        super().__init__()
        self.upconv = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, 2, stride=2),
            nn.InstanceNorm3d(out_channels, eps=1e-3, affine=True),
            nn.GELU(),
        )

        self.layers = nn.Sequential(
            LinearProjection(out_channels + skip_channels, out_channels),
            nn.InstanceNorm3d(out_channels, eps=1e-3, affine=True),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, 3, padding="same"),
            nn.InstanceNorm3d(out_channels, eps=1e-3, affine=True),
            nn.GELU(),
        )

    def forward(self, x: Tensor, skip_x: Tensor) -> Tensor:
        """Forward pass for the synthesis block."""
        x = self.upconv(x)
        x = torch.cat([x, skip_x], 1)  # channel concat
        x = self.layers(x)
        return x


class LinearProjection(nn.Module):
    """Linear projection of an input tensor."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initializes the LinearProjection with a linear transformation.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels after projection.
        """
        super().__init__()
        self.proj = nn.Linear(in_channels, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for the linear projection layer."""
        x = torch.permute(x, (0, 2, 3, 4, 1))
        x = self.proj(x)
        x = torch.permute(x, (0, 4, 1, 2, 3))
        return x
