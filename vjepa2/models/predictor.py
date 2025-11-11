"""
Predictor network for V-JEPA 2.
Predicts representations of masked regions from visible context.
"""

import torch
import torch.nn as nn
from typing import Optional, List
import math

from .vision_transformer import Block


class Predictor(nn.Module):
    """
    Transformer predictor for V-JEPA 2.
    Takes visible context tokens and mask tokens, predicts masked region representations.
    """

    def __init__(
        self,
        embed_dim: int = 384,
        predictor_embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        use_rope: bool = False,
        grid_size: int = 14,
        grid_depth: int = 8,
        use_sdpa: bool = True
    ):
        """
        Args:
            embed_dim: Encoder embedding dimension
            predictor_embed_dim: Predictor embedding dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dimension ratio
            qkv_bias: Whether to use bias in qkv projection
            qk_scale: Override default qk scale
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
            use_rope: Whether to use RoPE
            grid_size: Spatial grid size
            grid_depth: Temporal grid size
            use_sdpa: Whether to use scaled_dot_product_attention
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.predictor_embed_dim = predictor_embed_dim
        self.num_patches = grid_depth * grid_size * grid_size

        # Projection from encoder dim to predictor dim
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim)

        # Mask tokens (learnable tokens for masked positions)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))

        # Predictor positional encoding (learnable or fixed)
        self.predictor_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, predictor_embed_dim)
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=predictor_embed_dim,
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

        self.norm = nn.LayerNorm(predictor_embed_dim)

        # Projection back to encoder dimension
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.predictor_pos_embed, std=0.02)

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
        masks_x: torch.Tensor,
        masks_pred: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Forward pass of predictor.

        Args:
            x: Encoded visible tokens (B, N_visible, embed_dim)
            masks_x: Boolean mask indicating visible positions in full sequence (N_total,)
            masks_pred: List of boolean masks for prediction targets (each N_total,)

        Returns:
            List of predictions, one for each mask in masks_pred
                Each prediction has shape (B, N_masked_i, embed_dim)
        """
        B = x.shape[0]
        N_total = self.num_patches

        # Project encoder features to predictor dimension
        x = self.predictor_embed(x)  # (B, N_visible, predictor_embed_dim)

        # Expand mask token for all positions
        mask_tokens = self.mask_token.expand(B, N_total, -1)

        # Expand masks for batch
        if masks_x.dim() == 1:
            masks_x = masks_x.unsqueeze(0).expand(B, -1)

        # Create full sequence with mask tokens at masked positions
        x_full = mask_tokens.clone()
        x_full[masks_x] = x.reshape(-1, self.predictor_embed_dim)

        # Add positional embedding to all tokens
        x_full = x_full + self.predictor_pos_embed

        # Apply transformer blocks
        for block in self.blocks:
            x_full = block(x_full)

        x_full = self.norm(x_full)

        # Project back to encoder dimension
        x_full = self.predictor_proj(x_full)

        # Extract predictions for each target mask
        predictions = []
        for mask_pred in masks_pred:
            if mask_pred.dim() == 1:
                mask_pred = mask_pred.unsqueeze(0).expand(B, -1)

            # Select predicted tokens
            pred = x_full[mask_pred].reshape(B, -1, self.embed_dim)
            predictions.append(pred)

        return predictions


class ActionConditionedPredictor(nn.Module):
    """
    Action-conditioned predictor for V-JEPA 2-AC (robotics).
    Predicts next latent state given current state and action.
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        action_dim: int = 7,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        use_causal_attention: bool = True,
        use_sdpa: bool = True
    ):
        """
        Args:
            embed_dim: Latent dimension
            action_dim: Action dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dimension ratio
            qkv_bias: Whether to use bias in qkv projection
            qk_scale: Override default qk scale
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
            use_causal_attention: Whether to use causal (block-causal) attention
            use_sdpa: Whether to use scaled_dot_product_attention
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.use_causal_attention = use_causal_attention

        # Action embedding
        self.action_embed = nn.Linear(action_dim, embed_dim)

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
                use_rope=False,
                use_sdpa=use_sdpa
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Output projection
        self.head = nn.Linear(embed_dim, embed_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
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
        z: torch.Tensor,
        actions: torch.Tensor,
        return_sequence: bool = False
    ) -> torch.Tensor:
        """
        Predict next latent state given current state and action.

        Args:
            z: Current latent state (B, embed_dim) or (B, T, embed_dim)
            actions: Actions (B, action_dim) or (B, T, action_dim)
            return_sequence: Whether to return full sequence or just last state

        Returns:
            Predicted next latent state (B, embed_dim) or (B, T, embed_dim)
        """
        # Handle both single-step and multi-step inputs
        if z.dim() == 2:
            z = z.unsqueeze(1)  # (B, 1, embed_dim)
            single_step = True
        else:
            single_step = False

        if actions.dim() == 2:
            actions = actions.unsqueeze(1)  # (B, 1, action_dim)

        B, T, _ = z.shape

        # Embed actions
        action_embed = self.action_embed(actions)  # (B, T, embed_dim)

        # Combine state and action
        x = z + action_embed  # (B, T, embed_dim)

        # Create causal mask if needed
        if self.use_causal_attention:
            # Block-causal mask: allow attention within current and previous steps
            causal_mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool),
                diagonal=1
            )
            causal_mask = causal_mask.masked_fill(causal_mask, float('-inf'))
        else:
            causal_mask = None

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask=causal_mask)

        x = self.norm(x)

        # Output projection
        x = self.head(x)

        if single_step and not return_sequence:
            return x.squeeze(1)  # (B, embed_dim)
        else:
            return x  # (B, T, embed_dim)

    def rollout(
        self,
        z0: torch.Tensor,
        actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Rollout predictions for a sequence of actions.

        Args:
            z0: Initial latent state (B, embed_dim)
            actions: Sequence of actions (B, T, action_dim)

        Returns:
            Predicted latent trajectory (B, T+1, embed_dim)
        """
        B, T, _ = actions.shape

        # Initialize trajectory with initial state
        trajectory = [z0]

        # Autoregressively predict next states
        z = z0
        for t in range(T):
            z = self.forward(z, actions[:, t], return_sequence=False)
            trajectory.append(z)

        return torch.stack(trajectory, dim=1)  # (B, T+1, embed_dim)
