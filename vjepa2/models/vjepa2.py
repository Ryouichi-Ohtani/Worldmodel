"""
V-JEPA 2: Video Joint Embedding Predictive Architecture.
Main model combining encoder, target encoder (EMA), and predictor.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Dict
import copy

from .vision_transformer import VisionTransformer
from .predictor import Predictor, ActionConditionedPredictor


class VJEPA2(nn.Module):
    """
    V-JEPA 2 model for self-supervised video representation learning.

    Architecture:
        - Context encoder: Processes visible (non-masked) video patches
        - Target encoder: EMA copy of context encoder for target representations
        - Predictor: Predicts masked region representations from visible context
    """

    def __init__(
        self,
        # Encoder config
        img_size: int = 224,
        patch_size: int = 16,
        tubelet_size: int = 2,
        in_channels: int = 3,
        num_frames: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        # Predictor config
        predictor_embed_dim: int = 384,
        predictor_depth: int = 6,
        predictor_num_heads: int = 6,
        # Training config
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        use_rope: bool = False,
        use_cls_token: bool = False,
        use_learnable_pos_embed: bool = False,
        use_sdpa: bool = True,
        # EMA config
        ema_momentum: float = 0.996
    ):
        """
        Args:
            img_size: Input image size
            patch_size: Spatial patch size
            tubelet_size: Temporal patch size (for 3D patches)
            in_channels: Number of input channels
            num_frames: Number of input frames
            embed_dim: Encoder embedding dimension
            depth: Number of encoder transformer blocks
            num_heads: Number of encoder attention heads
            mlp_ratio: MLP hidden dimension ratio
            predictor_embed_dim: Predictor embedding dimension
            predictor_depth: Number of predictor transformer blocks
            predictor_num_heads: Number of predictor attention heads
            qkv_bias: Whether to use bias in qkv projection
            qk_scale: Override default qk scale
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
            use_rope: Whether to use RoPE
            use_cls_token: Whether to use [CLS] token
            use_learnable_pos_embed: Whether to use learnable positional embeddings
            use_sdpa: Whether to use scaled_dot_product_attention
            ema_momentum: EMA momentum for target encoder
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.ema_momentum = ema_momentum

        grid_size = img_size // patch_size
        grid_depth = num_frames // tubelet_size

        # Context encoder (trainable)
        self.encoder = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            in_channels=in_channels,
            num_frames=num_frames,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            use_rope=use_rope,
            use_cls_token=use_cls_token,
            use_learnable_pos_embed=use_learnable_pos_embed,
            use_sdpa=use_sdpa
        )

        # Target encoder (EMA copy, not directly trainable)
        self.target_encoder = copy.deepcopy(self.encoder)

        # Freeze target encoder
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # Predictor
        self.predictor = Predictor(
            embed_dim=embed_dim,
            predictor_embed_dim=predictor_embed_dim,
            depth=predictor_depth,
            num_heads=predictor_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            use_rope=use_rope,
            grid_size=grid_size,
            grid_depth=grid_depth,
            use_sdpa=use_sdpa
        )

    def forward(
        self,
        x: torch.Tensor,
        masks_enc: torch.Tensor,
        masks_pred: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass for training.

        Args:
            x: Input video (B, C, T, H, W)
            masks_enc: Boolean mask for encoder (N,) - True for visible tokens
            masks_pred: List of boolean masks for prediction targets (each N,)

        Returns:
            predictions: List of predicted representations for each mask
            targets: List of target representations for each mask
        """
        B = x.shape[0]

        # Encode visible context
        with torch.cuda.amp.autocast(enabled=True):
            z_context = self.encoder(x, mask=masks_enc, return_all_tokens=True)

        # Predict masked regions
        predictions = self.predictor(z_context, masks_enc, masks_pred)

        # Get target representations (no gradient)
        with torch.no_grad():
            z_target = self.target_encoder(x, mask=None, return_all_tokens=True)

            # Extract targets for each prediction mask
            targets = []
            for mask_pred in masks_pred:
                if mask_pred.dim() == 1:
                    mask_pred_expanded = mask_pred.unsqueeze(0).expand(B, -1)
                else:
                    mask_pred_expanded = mask_pred

                target = z_target[mask_pred_expanded].reshape(B, -1, self.embed_dim)
                targets.append(target)

        return predictions, targets

    @torch.no_grad()
    def update_target_encoder(self, momentum: Optional[float] = None):
        """
        Update target encoder using exponential moving average.

        Args:
            momentum: EMA momentum (if None, uses self.ema_momentum)
        """
        if momentum is None:
            momentum = self.ema_momentum

        # Update each parameter
        for param_q, param_k in zip(
            self.encoder.parameters(),
            self.target_encoder.parameters()
        ):
            param_k.data.mul_(momentum).add_(param_q.data, alpha=1 - momentum)

    def encode(
        self,
        x: torch.Tensor,
        use_target_encoder: bool = False,
        return_all_tokens: bool = True
    ) -> torch.Tensor:
        """
        Encode video to latent representation.

        Args:
            x: Input video (B, C, T, H, W)
            use_target_encoder: Whether to use target encoder
            return_all_tokens: Whether to return all tokens or just [CLS]

        Returns:
            Encoded features
        """
        encoder = self.target_encoder if use_target_encoder else self.encoder

        with torch.no_grad():
            return encoder(x, mask=None, return_all_tokens=return_all_tokens)

    def get_config(self) -> Dict:
        """Get model configuration."""
        return {
            'embed_dim': self.embed_dim,
            'num_frames': self.num_frames,
            'patch_size': self.patch_size,
            'tubelet_size': self.tubelet_size,
            'ema_momentum': self.ema_momentum,
        }


class VJEPA2_AC(nn.Module):
    """
    V-JEPA 2 with Action-Conditioned predictor for robotics.

    Architecture:
        - Frozen encoder from pretrained V-JEPA 2
        - Action-conditioned predictor for world modeling
    """

    def __init__(
        self,
        vjepa2_encoder: nn.Module,
        action_dim: int = 7,
        predictor_depth: int = 24,
        predictor_num_heads: int = 16,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        use_causal_attention: bool = True,
        use_sdpa: bool = True,
        freeze_encoder: bool = True
    ):
        """
        Args:
            vjepa2_encoder: Pretrained V-JEPA 2 encoder
            action_dim: Action dimension
            predictor_depth: Number of predictor transformer blocks
            predictor_num_heads: Number of predictor attention heads
            mlp_ratio: MLP hidden dimension ratio
            qkv_bias: Whether to use bias in qkv projection
            qk_scale: Override default qk scale
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
            use_causal_attention: Whether to use causal attention
            use_sdpa: Whether to use scaled_dot_product_attention
            freeze_encoder: Whether to freeze encoder weights
        """
        super().__init__()

        self.encoder = vjepa2_encoder
        self.action_dim = action_dim

        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Get encoder embedding dimension
        embed_dim = self.encoder.embed_dim

        # Action-conditioned predictor
        self.predictor = ActionConditionedPredictor(
            embed_dim=embed_dim,
            action_dim=action_dim,
            depth=predictor_depth,
            num_heads=predictor_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            use_causal_attention=use_causal_attention,
            use_sdpa=use_sdpa
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode video/image to latent representation.

        Args:
            x: Input video (B, C, T, H, W) or image (B, C, H, W)

        Returns:
            Latent state (B, embed_dim)
        """
        with torch.no_grad():
            z = self.encoder(x, mask=None, return_all_tokens=False)
        return z

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
            return_sequence: Whether to return full sequence

        Returns:
            Predicted next latent state
        """
        return self.predictor(z, actions, return_sequence)

    def rollout(
        self,
        z0: torch.Tensor,
        actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Rollout predictions for action sequence.

        Args:
            z0: Initial latent state (B, embed_dim)
            actions: Action sequence (B, T, action_dim)

        Returns:
            Predicted latent trajectory (B, T+1, embed_dim)
        """
        return self.predictor.rollout(z0, actions)

    def plan(
        self,
        z_init: torch.Tensor,
        z_goal: torch.Tensor,
        horizon: int,
        num_iterations: int = 100,
        lr: float = 0.1
    ) -> torch.Tensor:
        """
        Plan actions to reach goal state using gradient-based optimization.

        Args:
            z_init: Initial latent state (B, embed_dim)
            z_goal: Goal latent state (B, embed_dim)
            horizon: Planning horizon
            num_iterations: Number of optimization iterations
            lr: Learning rate for action optimization

        Returns:
            Optimized action sequence (B, horizon, action_dim)
        """
        B = z_init.shape[0]

        # Initialize actions randomly
        actions = torch.randn(
            B, horizon, self.action_dim,
            device=z_init.device,
            requires_grad=True
        )

        optimizer = torch.optim.Adam([actions], lr=lr)

        for _ in range(num_iterations):
            optimizer.zero_grad()

            # Rollout with current actions
            trajectory = self.rollout(z_init, actions)

            # Compute distance to goal (energy function)
            final_state = trajectory[:, -1]
            loss = torch.mean((final_state - z_goal) ** 2)

            # Optimize
            loss.backward()
            optimizer.step()

        return actions.detach()


def build_vjepa2_vitl() -> VJEPA2:
    """Build V-JEPA 2 with ViT-L encoder (300M parameters)."""
    return VJEPA2(
        img_size=224,
        patch_size=16,
        tubelet_size=2,
        num_frames=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        predictor_embed_dim=384,
        predictor_depth=12,
        predictor_num_heads=6,
        use_rope=True
    )


def build_vjepa2_vith() -> VJEPA2:
    """Build V-JEPA 2 with ViT-H encoder (600M parameters)."""
    return VJEPA2(
        img_size=224,
        patch_size=16,
        tubelet_size=2,
        num_frames=16,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        predictor_embed_dim=512,
        predictor_depth=12,
        predictor_num_heads=8,
        use_rope=True
    )


def build_vjepa2_vitg() -> VJEPA2:
    """Build V-JEPA 2 with ViT-g encoder (1B parameters)."""
    return VJEPA2(
        img_size=224,
        patch_size=16,
        tubelet_size=2,
        num_frames=16,
        embed_dim=1408,
        depth=40,
        num_heads=16,
        predictor_embed_dim=512,
        predictor_depth=12,
        predictor_num_heads=8,
        use_rope=True
    )


def build_vjepa2_ac(
    encoder_path: str,
    action_dim: int = 7
) -> VJEPA2_AC:
    """
    Build V-JEPA 2-AC model with pretrained encoder.

    Args:
        encoder_path: Path to pretrained V-JEPA 2 encoder checkpoint
        action_dim: Action dimension

    Returns:
        V-JEPA 2-AC model
    """
    # Load pretrained encoder
    checkpoint = torch.load(encoder_path, map_location='cpu')

    # Build base model and load encoder weights
    base_model = build_vjepa2_vitg()
    base_model.encoder.load_state_dict(checkpoint['encoder'])

    # Build action-conditioned model
    model = VJEPA2_AC(
        vjepa2_encoder=base_model.encoder,
        action_dim=action_dim,
        predictor_depth=24,
        predictor_num_heads=16,
        freeze_encoder=True
    )

    return model
