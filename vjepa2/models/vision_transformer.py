"""
Vision Transformer (ViT) implementation for V-JEPA 2.
Supports both 2D images and 3D video with spatiotemporal patch embedding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math

from ..utils.position_encoding import (
    get_3d_sincos_pos_embed,
    RoPE3D,
    interpolate_pos_embed_3d
)


class PatchEmbed3D(nn.Module):
    """
    3D patch embedding for video (spatiotemporal tubelets).
    Converts video to sequence of patch tokens.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        tubelet_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 768
    ):
        """
        Args:
            img_size: Input image size (H = W)
            patch_size: Spatial patch size
            tubelet_size: Temporal patch size
            in_channels: Number of input channels
            embed_dim: Embedding dimension
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.grid_size = img_size // patch_size
        self.num_patches_per_frame = self.grid_size ** 2

        # 3D convolution for spatiotemporal patch embedding
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        """
        Args:
            x: Input video (B, C, T, H, W)

        Returns:
            tokens: Patch tokens (B, N, embed_dim) where N = (T/tubelet_size) * (H/patch_size) * (W/patch_size)
            grid_shape: (T_patches, H_patches, W_patches)
        """
        B, C, T, H, W = x.shape

        # Apply 3D convolution
        x = self.proj(x)  # (B, embed_dim, T', H', W')

        # Get grid dimensions
        T_patches = x.shape[2]
        H_patches = x.shape[3]
        W_patches = x.shape[4]

        # Reshape to sequence
        x = x.flatten(2).transpose(1, 2)  # (B, T'*H'*W', embed_dim)

        return x, (T_patches, H_patches, W_patches)


class Attention(nn.Module):
    """
    Multi-head self-attention with optional RoPE support.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_rope: bool = False,
        grid_size: int = 14,
        grid_depth: int = 8,
        use_sdpa: bool = True
    ):
        """
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in qkv projection
            qk_scale: Override default qk scale
            attn_drop: Attention dropout rate
            proj_drop: Projection dropout rate
            use_rope: Whether to use RoPE
            grid_size: Spatial grid size for RoPE
            grid_depth: Temporal grid size for RoPE
            use_sdpa: Whether to use PyTorch scaled_dot_product_attention
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.use_rope = use_rope
        self.use_sdpa = use_sdpa

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if use_rope:
            self.rope = RoPE3D(dim, grid_size, grid_depth)
        else:
            self.rope = None

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, N, C)
            mask: Attention mask (B, N, N) or None

        Returns:
            Output tensor (B, N, C)
        """
        B, N, C = x.shape

        # QKV projection
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply RoPE if enabled
        if self.rope is not None:
            # Reshape for RoPE application
            q = q.transpose(1, 2).reshape(B, N, C)  # (B, N, C)
            k = k.transpose(1, 2).reshape(B, N, C)

            q, k = self.rope(q, k)

            # Reshape back
            q = q.reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
            k = k.reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)

        # Attention
        if self.use_sdpa and hasattr(F, 'scaled_dot_product_attention'):
            # Use PyTorch's efficient SDPA
            attn_mask = mask if mask is not None else None
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0
            )
        else:
            # Manual attention computation
            attn = (q @ k.transpose(-2, -1)) * self.scale

            if mask is not None:
                attn = attn + mask

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        # Reshape and project
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MLP(nn.Module):
    """
    MLP block with GELU activation.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """
    Transformer block with pre-norm architecture.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        use_rope: bool = False,
        grid_size: int = 14,
        grid_depth: int = 8,
        use_sdpa: bool = True
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_rope=use_rope,
            grid_size=grid_size,
            grid_depth=grid_depth,
            use_sdpa=use_sdpa
        )
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop=drop
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer for V-JEPA 2.
    Supports both image and video inputs with optional RoPE.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        tubelet_size: int = 2,
        in_channels: int = 3,
        num_frames: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        use_rope: bool = False,
        use_cls_token: bool = False,
        use_learnable_pos_embed: bool = False,
        use_sdpa: bool = True
    ):
        """
        Args:
            img_size: Input image size
            patch_size: Spatial patch size
            tubelet_size: Temporal patch size
            in_channels: Number of input channels
            num_frames: Number of input frames
            embed_dim: Embedding dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dimension ratio
            qkv_bias: Whether to use bias in qkv projection
            qk_scale: Override default qk scale
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
            use_rope: Whether to use RoPE instead of absolute positional encoding
            use_cls_token: Whether to use [CLS] token
            use_learnable_pos_embed: Whether to use learnable positional embeddings
            use_sdpa: Whether to use scaled_dot_product_attention
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        self.use_rope = use_rope
        self.use_cls_token = use_cls_token

        # Patch embedding
        self.patch_embed = PatchEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )

        grid_size = img_size // patch_size
        grid_depth = num_frames // tubelet_size
        num_patches = grid_depth * grid_size * grid_size

        # CLS token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            num_patches += 1
        else:
            self.cls_token = None

        # Positional embedding
        if not use_rope:
            if use_learnable_pos_embed:
                self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            else:
                # Fixed sinusoidal position embedding
                pos_embed = get_3d_sincos_pos_embed(
                    embed_dim,
                    grid_size,
                    grid_depth,
                    cls_token=use_cls_token
                )
                self.register_buffer('pos_embed', pos_embed.unsqueeze(0))
        else:
            self.pos_embed = None

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                use_rope=use_rope,
                grid_size=grid_size,
                grid_depth=grid_depth,
                use_sdpa=use_sdpa
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        if hasattr(self, 'pos_embed') and isinstance(self.pos_embed, nn.Parameter):
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Initialize patch_embed
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize layers
        self.apply(self._init_layer_weights)

    def _init_layer_weights(self, m):
        """Initialize layer weights."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_all_tokens: bool = False
    ) -> torch.Tensor:
        """
        Args:
            x: Input video (B, C, T, H, W)
            mask: Boolean mask (N,) - True for kept tokens, False for masked
            return_all_tokens: Whether to return all tokens or just [CLS]

        Returns:
            Encoded features (B, N, embed_dim) or (B, embed_dim) if not return_all_tokens
        """
        B = x.shape[0]

        # Patch embedding
        x, (T_patches, H_patches, W_patches) = self.patch_embed(x)

        # Add CLS token
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)

        # Add positional embedding
        if self.pos_embed is not None:
            x = x + self.pos_embed

        x = self.pos_drop(x)

        # Apply mask if provided
        if mask is not None:
            # Expand mask for batch
            if mask.dim() == 1:
                mask = mask.unsqueeze(0).expand(B, -1)

            if self.cls_token is not None:
                # Keep CLS token always
                cls_mask = torch.ones(B, 1, dtype=torch.bool, device=mask.device)
                mask = torch.cat([cls_mask, mask], dim=1)

            # Select only visible tokens
            x = x[mask.unsqueeze(-1).expand_as(x)].reshape(B, -1, self.embed_dim)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Return features
        if return_all_tokens or self.cls_token is None:
            return x
        else:
            return x[:, 0]  # Return CLS token only

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: int = 1,
        mask: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """
        Get intermediate layer outputs.

        Args:
            x: Input video
            n: Number of last layers to return
            mask: Optional mask

        Returns:
            List of intermediate features
        """
        B = x.shape[0]

        # Patch embedding
        x, _ = self.patch_embed(x)

        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)

        if self.pos_embed is not None:
            x = x + self.pos_embed

        x = self.pos_drop(x)

        if mask is not None:
            if mask.dim() == 1:
                mask = mask.unsqueeze(0).expand(B, -1)
            if self.cls_token is not None:
                cls_mask = torch.ones(B, 1, dtype=torch.bool, device=mask.device)
                mask = torch.cat([cls_mask, mask], dim=1)
            x = x[mask.unsqueeze(-1).expand_as(x)].reshape(B, -1, self.embed_dim)

        # Collect last n layer outputs
        outputs = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i >= len(self.blocks) - n:
                outputs.append(self.norm(x))

        return outputs
