"""
Position encoding utilities for V-JEPA 2.
Implements 3D sinusoidal position embeddings and 3D RoPE (Rotary Position Embedding).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


def get_3d_sincos_pos_embed(
    embed_dim: int,
    grid_size: int,
    grid_depth: int,
    cls_token: bool = False,
    uniform_power: bool = False
) -> torch.Tensor:
    """
    Generate 3D sinusoidal position embeddings for video (T x H x W).

    Args:
        embed_dim: Embedding dimension (flexibly partitioned across T, H, W)
        grid_size: Spatial grid size (H = W)
        grid_depth: Temporal grid size (T, number of frames)
        cls_token: Whether to include [CLS] token position
        uniform_power: Whether to use uniform power distribution across dimensions

    Returns:
        Position embeddings of shape (T*H*W, embed_dim) or (1+T*H*W, embed_dim) with cls_token
    """
    # Flexible dimension partitioning (removed assertion for compatibility with standard ViT configs)
    # Partition embedding dimension into three parts for T, H, W
    # Ensure all dimensions are even (required for sinusoidal encoding)
    if uniform_power:
        # Equal distribution
        base_dim = (embed_dim // 3) // 2 * 2  # Round down to nearest even
        dim_t = dim_h = dim_w = base_dim
    else:
        # Following paper: partition into approximately equal segments
        dim_t = (embed_dim // 3) // 2 * 2  # Round down to nearest even number
        dim_h = ((embed_dim - dim_t) // 2) // 2 * 2  # Round down to nearest even number
        dim_w = embed_dim - dim_t - dim_h  # Remainder

    # Generate 3D grid coordinates
    grid_t = np.arange(grid_depth, dtype=np.float32)
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)

    # Create meshgrid
    grid = np.meshgrid(grid_t, grid_h, grid_w, indexing='ij')  # T x H x W
    grid = np.stack(grid, axis=0)  # 3 x T x H x W
    grid = grid.reshape([3, -1]).T  # (T*H*W) x 3

    # Generate sinusoidal embeddings for each dimension
    pos_embed_t = get_1d_sincos_pos_embed_from_grid(dim_t, grid[:, 0])  # T
    pos_embed_h = get_1d_sincos_pos_embed_from_grid(dim_h, grid[:, 1])  # H
    pos_embed_w = get_1d_sincos_pos_embed_from_grid(dim_w, grid[:, 2])  # W

    # Concatenate embeddings
    pos_embed = np.concatenate([pos_embed_t, pos_embed_h, pos_embed_w], axis=1)

    if cls_token:
        # Add zero vector for [CLS] token
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)

    return torch.from_numpy(pos_embed).float()


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """
    Generate 1D sinusoidal position embeddings.

    Args:
        embed_dim: Output dimension for each position
        pos: List of positions to encode (shape: [M,])

    Returns:
        Position embeddings of shape (M, embed_dim)
    """
    assert embed_dim % 2 == 0, "embed_dim must be even"

    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2)

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class RoPE3D(nn.Module):
    """
    3D Rotary Position Embedding (RoPE) for video transformers.
    Extends 1D RoPE to spatiotemporal dimensions.
    """

    def __init__(
        self,
        embed_dim: int,
        grid_size: int,
        grid_depth: int,
        theta: float = 10000.0
    ):
        """
        Args:
            embed_dim: Feature dimension (must be divisible by 3)
            grid_size: Spatial grid size (H = W)
            grid_depth: Temporal grid size (T)
            theta: Temperature parameter for frequency computation
        """
        super().__init__()
        assert embed_dim % 6 == 0, "embed_dim must be divisible by 6 for 3D RoPE"

        self.embed_dim = embed_dim
        self.grid_size = grid_size
        self.grid_depth = grid_depth
        self.theta = theta

        # Partition dimension into T, H, W (each gets dim/3, and we need sin/cos pairs)
        dim_per_axis = embed_dim // 3
        assert dim_per_axis % 2 == 0

        # Precompute frequency bands
        self.register_buffer('freqs_t', self._compute_freqs(grid_depth, dim_per_axis))
        self.register_buffer('freqs_h', self._compute_freqs(grid_size, dim_per_axis))
        self.register_buffer('freqs_w', self._compute_freqs(grid_size, dim_per_axis))

    def _compute_freqs(self, max_len: int, dim: int) -> torch.Tensor:
        """Compute frequency bands for RoPE."""
        freqs = 1.0 / (self.theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_len, dtype=torch.float32)
        freqs = torch.outer(t, freqs)  # (max_len, dim/2)
        freqs_cos = torch.cos(freqs)
        freqs_sin = torch.sin(freqs)
        return torch.stack([freqs_cos, freqs_sin], dim=-1)  # (max_len, dim/2, 2)

    def apply_rotary_pos_emb(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply rotary position embedding to input tensor.

        Args:
            x: Input tensor of shape (..., seq_len, dim)
            freqs: Frequency tensor of shape (seq_len, dim/2, 2)

        Returns:
            Rotated tensor of same shape as input
        """
        # Reshape x to (..., seq_len, dim/2, 2)
        x = x.reshape(*x.shape[:-1], -1, 2)

        # Extract cos and sin
        freqs_cos = freqs[..., 0]  # (seq_len, dim/2)
        freqs_sin = freqs[..., 1]  # (seq_len, dim/2)

        # Apply rotation
        x_r = x[..., 0]  # Real part
        x_i = x[..., 1]  # Imaginary part

        x_out_r = x_r * freqs_cos - x_i * freqs_sin
        x_out_i = x_r * freqs_sin + x_i * freqs_cos

        x_out = torch.stack([x_out_r, x_out_i], dim=-1)
        return x_out.reshape(*x.shape[:-2], -1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        t_idx: Optional[torch.Tensor] = None,
        h_idx: Optional[torch.Tensor] = None,
        w_idx: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply 3D RoPE to query and key tensors.

        Args:
            q: Query tensor (..., num_tokens, embed_dim)
            k: Key tensor (..., num_tokens, embed_dim)
            t_idx: Temporal indices (num_tokens,)
            h_idx: Height indices (num_tokens,)
            w_idx: Width indices (num_tokens,)

        Returns:
            Rotated (q, k) tensors
        """
        batch_shape = q.shape[:-2]
        num_tokens = q.shape[-2]

        # Default to sequential indices if not provided
        if t_idx is None:
            # Assume tokens are ordered as T x H x W
            t_idx = torch.arange(num_tokens, device=q.device) // (self.grid_size ** 2)
            t_idx = t_idx % self.grid_depth
        if h_idx is None:
            h_idx = (torch.arange(num_tokens, device=q.device) // self.grid_size) % self.grid_size
        if w_idx is None:
            w_idx = torch.arange(num_tokens, device=q.device) % self.grid_size

        # Get frequencies for each token
        freqs_t = self.freqs_t[t_idx]  # (num_tokens, dim/6, 2)
        freqs_h = self.freqs_h[h_idx]
        freqs_w = self.freqs_w[w_idx]

        # Partition q and k into T, H, W components
        dim_per_axis = self.embed_dim // 3

        q_t, q_h, q_w = torch.split(q, dim_per_axis, dim=-1)
        k_t, k_h, k_w = torch.split(k, dim_per_axis, dim=-1)

        # Apply rotary embedding to each component
        q_t = self.apply_rotary_pos_emb(q_t, freqs_t)
        q_h = self.apply_rotary_pos_emb(q_h, freqs_h)
        q_w = self.apply_rotary_pos_emb(q_w, freqs_w)

        k_t = self.apply_rotary_pos_emb(k_t, freqs_t)
        k_h = self.apply_rotary_pos_emb(k_h, freqs_h)
        k_w = self.apply_rotary_pos_emb(k_w, freqs_w)

        # Concatenate back
        q_out = torch.cat([q_t, q_h, q_w], dim=-1)
        k_out = torch.cat([k_t, k_h, k_w], dim=-1)

        return q_out, k_out


def interpolate_pos_embed_3d(
    pos_embed: torch.Tensor,
    orig_size: Tuple[int, int, int],
    new_size: Tuple[int, int, int],
    mode: str = 'trilinear'
) -> torch.Tensor:
    """
    Interpolate 3D position embeddings to new size.

    Args:
        pos_embed: Position embeddings (T*H*W, embed_dim)
        orig_size: Original (T, H, W) size
        new_size: Target (T, H, W) size
        mode: Interpolation mode ('trilinear', 'nearest')

    Returns:
        Interpolated position embeddings (new_T*new_H*new_W, embed_dim)
    """
    orig_t, orig_h, orig_w = orig_size
    new_t, new_h, new_w = new_size

    if orig_size == new_size:
        return pos_embed

    # Reshape to 3D grid
    embed_dim = pos_embed.shape[-1]
    pos_embed = pos_embed.reshape(orig_t, orig_h, orig_w, embed_dim)
    pos_embed = pos_embed.permute(3, 0, 1, 2).unsqueeze(0)  # (1, embed_dim, T, H, W)

    # Interpolate
    pos_embed = torch.nn.functional.interpolate(
        pos_embed,
        size=(new_t, new_h, new_w),
        mode=mode,
        align_corners=False if mode != 'nearest' else None
    )

    # Reshape back
    pos_embed = pos_embed.squeeze(0).permute(1, 2, 3, 0)  # (new_T, new_H, new_W, embed_dim)
    pos_embed = pos_embed.reshape(-1, embed_dim)  # (new_T*new_H*new_W, embed_dim)

    return pos_embed
