"""Transformer-based prompt predictor for SAM2 using image encodings and learnable prompt tokens. Based on the prompt predictor in https://github.com/aswahd/SamRadiology/"""

from typing import List, Tuple
import numpy as np
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F

from sam2.modeling.position_encoding import PositionEmbeddingSine as PositionEmbedding2d
from sam2.modeling.sam.transformer import TwoWayAttentionBlock
from sam2.modeling.sam2_utils import MLP

class PromptPredictor(nn.Module):
    """
    A class to handle sampling of prompts for the SAMv2 model.
    This class is responsible for generating prompts based on the input data and model configuration.
    """
    
    def __init__(self, prompt_encoder, embed_dim: int = 256, num_heads: int = 8, depth: int = 3, channel_dims=[256, 64, 32], mlp_dim: int = 2048, activation: nn.Module = nn.GELU):
        super().__init__()
        assert len(channel_dims) == depth, "Number of channel dimensions for image features must match the depth of the model."
        
        self.prompt_encoder = prompt_encoder
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.depth = depth
        self.mlp_dim = mlp_dim
        self.scale_factor = 4  # Scale factor for mask prediction since the features are downsampled by 4x in the original SAM2 implementation

        # Initialize the position embeddings for image and prompt
        self.image_pos = PositionEmbedding2d(embed_dim)
        self.prompt_pos = PositionEmbedding1d(embed_dim)
        
        # Initialize the transformer encoder for prompt prediction
        self.layers = nn.ModuleList([
            TwoWayAttentionBlock(embedding_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim, activation=activation) for _ in range(depth)
        ])
        self.channel_adapters = nn.ModuleList([
            nn.Conv2d(in_channels=dim, out_channels=embed_dim, kernel_size=1, stride=1, padding=0) if dim != embed_dim else nn.Identity() for dim in channel_dims
        ])
        
        # Initialize prediction heads for prompt prediction
        self.box_predictor = BoxPredictor(in_channels=embed_dim * 2, hidden_channels=256)
        self.mask_predictor = MaskPredictor(in_channels=embed_dim, hidden_channels=64, scale_factor=self.scale_factor)
        
        self._freeze_encoder()
        
    def _freeze_encoder(self):
        """Freeze the prompt encoder parameters."""
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False
        
    def _interpolate_image_features(self, image_features: Tensor, original_size: Tuple[int, int], target_size: Tuple[int, int], mode: str = "bilinear") -> Tensor:
        # Assumes image_features is of shape (B, HW, C)
        if original_size == target_size:
            return image_features
        image_features = image_features.permute(0, 2, 1).view(-1, image_features.shape[-1], original_size[0], original_size[1])  # B C H W
        # Interpolate to the target size
        image_features = F.interpolate(image_features, size=target_size, mode=mode, align_corners=None if mode == "nearest" else False)
        return image_features.flatten(2).permute(0, 2, 1)  # B HW C

    def forward(self, image_features: List[Tensor], prompt_tokens: Tensor) -> Tuple[Tensor]:
        image_features = image_features[::-1] # High to low dimensional features
        image_features = [adapter(feat) for adapter, feat in zip(self.channel_adapters, image_features)]
        
        B, C, H, W = image_features[0].shape
        prompt_pos = self.prompt_pos(prompt_tokens)

        # Get embeddings for a blank prompt
        dense_embeddings = self.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(B, -1, H, W).flatten(2).permute(0, 2, 1)  # B H*W C
        initial_size = (H, W)

        for i, layer in enumerate(self.layers):
            image_embed = image_features[i] # B C H W
            image_pos = self.image_pos(image_embed)
            target_size = image_embed.shape[-2:]  # Target size for interpolation
            # Reshape to transformer inputs
            image_embed = image_embed.flatten(2).permute(0, 2, 1)  # B H*W C
            image_pos = image_pos.flatten(2).permute(0, 2, 1)  # B H*W C
            # Add mask embeddings
            image_embed = image_embed + self._interpolate_image_features(dense_embeddings, initial_size, target_size, mode="nearest")
            initial_size = target_size  # Update initial size
            
            prompt_tokens, dense_embeddings = layer(queries=prompt_tokens, keys=image_embed, query_pe=prompt_pos, key_pe=image_pos)

        # Reshape dense embeddings to image shape (use low res features and upscale in mask predictor)
        B, _, H, W = image_features[-1].shape
        dense_embeddings = dense_embeddings.view(B, H, W, -1).permute(0, 3, 1, 2)  # B C H*W

        # Pass the output through the prediction heads
        box_outputs, box_labels = self.box_predictor(prompt_tokens[:, :2])
        points_scale = torch.tensor([H, W], device=box_outputs.device, dtype=box_outputs.dtype)[-2:].view(1, 2) * self.scale_factor # [1, 2]
        box_outputs = box_outputs * points_scale  # Scale the box outputs to the image size
        mask_prompt = self.mask_predictor(dense_embeddings)

        return box_outputs, box_labels, mask_prompt

class PositionEmbedding1d(nn.Module):
    """1D position embedding for prompts."""
    
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = int(np.ceil(embed_dim / 2) * 2)  # Ensure even dimension for sinusoidal embeddings
        inv_freqs = 1.0 / (10000 ** (torch.arange(0, self.embed_dim, 2).float() / self.embed_dim))
        self.register_buffer("inv_freqs", inv_freqs)
        self.register_buffer("cached_pos", None, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        # x is expected to be of shape (B, N, C)
        
        if self.cached_pos is not None and self.cached_pos.shape[1] == x.shape[1]:
            return self.cached_pos
        
        self.cached_pos = None
        B, N, C = x.shape
        pos_x = torch.arange(N, device=x.device, dtype=self.inv_freqs.dtype)
        sincos_x = torch.einsum("i,j->ij", pos_x, self.inv_freqs)
        emb_x = torch.stack((sincos_x.sin(), sincos_x.cos()), dim=-1).flatten(-2, -1)  # Shape: (N, C)
        # Save to cached_pos
        emb = torch.zeros((N, self.embed_dim), device=x.device, dtype=x.dtype)
        emb[:, :self.embed_dim] = emb_x
        self.cached_pos = emb[None, :, :C].repeat(B, 1, 1)  # Shape: (B, N, C)
        return self.cached_pos
        

class BoxPredictor(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 256, num_layers: int = 2, top_left_label: int = 2, bottom_right_label: int = 3):
        super().__init__()
        self.top_left_label = top_left_label
        self.bottom_right_label = bottom_right_label
        
        self.mlp = MLP(in_channels, hidden_channels, 4, num_layers=num_layers)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # x is expected to be of shape (B, n_tokens, 256)
        x = x.flatten(1)
        x = self.mlp(x).unsqueeze(1).view(-1, 2, 2) # B, 2, 2 (i.e., B, N, 2)
        x = torch.sigmoid(x) # Convert to [0, 1] coordinates
        labels = torch.tensor([[self.top_left_label, self.bottom_right_label]], device=x.device, dtype=torch.int32).repeat(x.shape[0], 1)  # B, N
        return x, labels
    
class MaskPredictor(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, num_layers: int = 2, scale_factor: int = 4, dropout: float = 0.1, use_batch_norm: bool = False):
        super().__init__()
        self.scale_factor = scale_factor
        
        self.layers = nn.ModuleList([nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=not use_batch_norm)] + [
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=not use_batch_norm)
            for _ in range(num_layers - 1)
        ])
        self.norms = nn.ModuleList([
            nn.BatchNorm2d(hidden_channels)
            for _ in range(num_layers)
        ]) if use_batch_norm else None
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else None
        
        self.out = nn.Conv2d(hidden_channels, 1, kernel_size=1, stride=1)
        
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x: Tensor) -> Tensor:
        # x is expected to be of shape (B, C, H, W)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.norms is not None:
                x = self.norms[i](x)
            x = F.relu(x)
            if self.dropout is not None:
                x = self.dropout(x)
            x = F.relu(x)
        x = self.out(x)
        # Upsample to desired size
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        return x