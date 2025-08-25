"""Transformer-based prompt predictor for SAM2 using image encodings and learnable prompt tokens. Based on the prompt predictor in https://github.com/aswahd/SamRadiology/"""

from typing import List, Tuple
import numpy as np
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F

from sam2.modeling.sam.transformer import Attention

class PromptSynthesisBlock(nn.Module):
    """A single block in the up-sampling UNet architecture for prompt prediction."""
    def __init__(self, in_channels: int, hidden_channels: int, skip_channels: int, out_channels: int, scale: int = 2):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, hidden_channels, scale, stride=scale)
        self.layers = nn.Sequential(
            nn.Conv2d(hidden_channels + skip_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU()
        )

    def forward(self, x: Tensor, skip_x: Tensor) -> Tensor:
        x = self.upconv(x)
        x = torch.cat([x, skip_x], dim=1)
        x = self.layers(x)
        return x

class PromptPredictor(nn.Module):
    """
    Simple UNet to predict mask prompts for the SAMv2 model from image encodings.
    """
    
    def __init__(self, in_channels: int = 256, hidden_channels: int = 256, out_channels: int = 1, depth: int = 2, layer_scale: int = 2, channel_dims: List[int] = [64, 32]):
        super().__init__()
        assert depth == len(channel_dims), "Depth must match the length of channel_dims"
        self.scale_factor = 4  # Scale factor for mask prediction
        # Original SAM2 has 4x patch embedding
        self.layers = nn.ModuleList()
        curr_channels = in_channels
        for i in range(depth):
            self.layers.append(
                PromptSynthesisBlock(
                    in_channels=curr_channels,
                    hidden_channels=hidden_channels,
                    skip_channels=channel_dims[i],
                    out_channels=hidden_channels,
                    scale=layer_scale
                )
            )
            curr_channels = hidden_channels
        self.final_conv = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        
        self.init_parameters()

    def init_parameters(self):
        """Initializes the parameters of the model."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, image_features: List[Tensor]) -> Tensor:
        x = image_features[-1]
        for layer, skip_x in zip(self.layers, reversed(image_features[:-1])):
            x = layer(x, skip_x)
        x = self.final_conv(x)
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        return torch.sigmoid(x)

class LoRALinear(nn.Module):
    """A linear layer with LoRA (Low-Rank Adaptation) applied."""
    def __init__(self, proj: nn.Module, input_dim: int, output_dim: int, r: int, a: int):
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
        attn_block.q_proj = LoRALinear(q_proj, in_features, out_features, self.r, self.a)
        attn_block.v_proj = LoRALinear(v_proj, in_features, out_features, self.r, self.a)
    
    def apply(self, mask_decoder: nn.Module) -> nn.Module:
        """Applies LoRA to all transformer blocks in the MaskDecoder."""
        for p in mask_decoder.parameters():
            p.requires_grad = False
            
        transformer = mask_decoder.transformer
        
        for _, blk in enumerate(transformer.layers):
            self._apply_lora(blk.self_attn)
            self._apply_lora(blk.cross_attn_token_to_image)
            self._apply_lora(blk.cross_attn_image_to_token)
        
        self._apply_lora(transformer.final_attn_token_to_image)
        
        return mask_decoder