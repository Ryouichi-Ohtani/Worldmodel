"""
Masking utilities for V-JEPA 2.
Implements multi-block masking strategy for spatiotemporal video tokens.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional


class MultiBlockMaskGenerator:
    """
    Multi-block masking generator for V-JEPA 2.
    Creates random spatiotemporal masks for video tokens.
    """

    def __init__(
        self,
        input_size: Tuple[int, int, int],  # (T, H, W)
        num_masks: int = 4,
        min_aspect_ratio: float = 0.3,
        max_aspect_ratio: float = 3.0,
        min_area: float = 0.15,
        max_area: float = 0.7,
        spatial_only: bool = False
    ):
        """
        Args:
            input_size: Input video size (T, H, W) in patches
            num_masks: Number of mask blocks to generate
            min_aspect_ratio: Minimum aspect ratio for mask blocks
            max_aspect_ratio: Maximum aspect ratio for mask blocks
            min_area: Minimum area ratio for each mask block
            max_area: Maximum area ratio for each mask block
            spatial_only: If True, masks only spatial dimensions (H, W)
        """
        self.T, self.H, self.W = input_size
        self.num_masks = num_masks
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_area = min_area
        self.max_area = max_area
        self.spatial_only = spatial_only

        self.total_patches = self.T * self.H * self.W
        self.spatial_patches = self.H * self.W

    def __call__(self) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Generate multi-block masks.

        Returns:
            encoder_mask: Boolean mask for encoder (True = keep, False = mask)
                         Shape: (T*H*W,)
            predictor_masks: List of masks for predictor targets
                           Each mask shape: (T*H*W,)
        """
        # Initialize mask (all visible)
        encoder_mask = torch.ones(self.T, self.H, self.W, dtype=torch.bool)

        # Generate mask blocks
        predictor_masks = []

        for _ in range(self.num_masks):
            # Sample block dimensions
            if self.spatial_only:
                # Only mask spatial dimensions
                block_t = self.T
                block_h, block_w = self._sample_block_size(self.H, self.W)
                block_t_start = 0
                block_h_start = np.random.randint(0, self.H - block_h + 1)
                block_w_start = np.random.randint(0, self.W - block_w + 1)
            else:
                # Mask spatiotemporal blocks
                # Sample temporal extent
                temporal_ratio = np.random.uniform(0.3, 1.0)
                block_t = max(1, int(self.T * temporal_ratio))

                # Sample spatial extent
                block_h, block_w = self._sample_block_size(self.H, self.W)

                # Random position
                block_t_start = np.random.randint(0, self.T - block_t + 1)
                block_h_start = np.random.randint(0, self.H - block_h + 1)
                block_w_start = np.random.randint(0, self.W - block_w + 1)

            # Create block mask
            block_mask = torch.zeros(self.T, self.H, self.W, dtype=torch.bool)
            block_mask[
                block_t_start:block_t_start + block_t,
                block_h_start:block_h_start + block_h,
                block_w_start:block_w_start + block_w
            ] = True

            # Update encoder mask (remove this block from encoder)
            encoder_mask = encoder_mask & ~block_mask

            # Add to predictor targets
            predictor_masks.append(block_mask.flatten())

        # Flatten encoder mask
        encoder_mask = encoder_mask.flatten()

        return encoder_mask, predictor_masks

    def _sample_block_size(self, max_h: int, max_w: int) -> Tuple[int, int]:
        """
        Sample random block size with area and aspect ratio constraints.

        Args:
            max_h: Maximum height
            max_w: Maximum width

        Returns:
            (block_h, block_w): Block dimensions
        """
        total_area = max_h * max_w
        target_area = np.random.uniform(self.min_area, self.max_area) * total_area

        # Sample aspect ratio
        aspect_ratio = np.random.uniform(self.min_aspect_ratio, self.max_aspect_ratio)

        # Compute dimensions
        block_h = int(np.sqrt(target_area / aspect_ratio))
        block_w = int(aspect_ratio * block_h)

        # Clip to valid range
        block_h = min(max(1, block_h), max_h)
        block_w = min(max(1, block_w), max_w)

        return block_h, block_w


class RandomMaskGenerator:
    """
    Random token masking generator (simpler alternative to multi-block).
    """

    def __init__(
        self,
        input_size: Tuple[int, int, int],
        mask_ratio: float = 0.75,
        num_predict_masks: int = 1
    ):
        """
        Args:
            input_size: Input video size (T, H, W) in patches
            mask_ratio: Ratio of tokens to mask
            num_predict_masks: Number of prediction masks to generate
        """
        self.T, self.H, self.W = input_size
        self.mask_ratio = mask_ratio
        self.num_predict_masks = num_predict_masks
        self.total_patches = self.T * self.H * self.W

    def __call__(self) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Generate random masks.

        Returns:
            encoder_mask: Boolean mask for encoder
            predictor_masks: List of masks for predictor targets
        """
        num_masked = int(self.total_patches * self.mask_ratio)

        # Random permutation
        indices = torch.randperm(self.total_patches)
        masked_indices = indices[:num_masked]

        # Create encoder mask
        encoder_mask = torch.ones(self.total_patches, dtype=torch.bool)
        encoder_mask[masked_indices] = False

        # Create predictor masks
        predictor_masks = []
        tokens_per_mask = num_masked // self.num_predict_masks

        for i in range(self.num_predict_masks):
            start_idx = i * tokens_per_mask
            end_idx = (i + 1) * tokens_per_mask if i < self.num_predict_masks - 1 else num_masked

            pred_mask = torch.zeros(self.total_patches, dtype=torch.bool)
            pred_mask[masked_indices[start_idx:end_idx]] = True
            predictor_masks.append(pred_mask)

        return encoder_mask, predictor_masks


def apply_masks(
    x: torch.Tensor,
    masks: torch.Tensor,
    concat: bool = True
) -> torch.Tensor:
    """
    Apply boolean masks to token sequence.

    Args:
        x: Input tokens (B, N, D) where N is number of tokens
        masks: Boolean mask (N,) or (B, N)
        concat: If True, concatenate masked tokens; if False, return as is with masking

    Returns:
        Masked tokens. If concat=True, shape is (B, M, D) where M = sum(masks)
        If concat=False, shape is (B, N, D) with masked positions zeroed
    """
    if masks.dim() == 1:
        # Broadcast mask to batch dimension
        masks = masks.unsqueeze(0).expand(x.size(0), -1)

    if concat:
        # Select only masked tokens
        masked_x = []
        for i in range(x.size(0)):
            masked_x.append(x[i, masks[i]])
        # Stack with padding if necessary
        max_len = max(m.size(0) for m in masked_x)
        result = torch.zeros(x.size(0), max_len, x.size(2), device=x.device, dtype=x.dtype)
        for i, m in enumerate(masked_x):
            result[i, :m.size(0)] = m
        return result
    else:
        # Zero out masked positions
        return x * masks.unsqueeze(-1).float()


class MaskCollator:
    """
    Collator for generating masks on-the-fly during training.
    """

    def __init__(
        self,
        input_size: Tuple[int, int, int],
        mask_generator_type: str = 'multiblock',
        **mask_kwargs
    ):
        """
        Args:
            input_size: Input video size (T, H, W) in patches
            mask_generator_type: Type of mask generator ('multiblock' or 'random')
            **mask_kwargs: Additional arguments for mask generator
        """
        self.input_size = input_size

        if mask_generator_type == 'multiblock':
            self.mask_generator = MultiBlockMaskGenerator(input_size, **mask_kwargs)
        elif mask_generator_type == 'random':
            self.mask_generator = RandomMaskGenerator(input_size, **mask_kwargs)
        else:
            raise ValueError(f"Unknown mask generator type: {mask_generator_type}")

    def __call__(self, batch: torch.Tensor) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, List[torch.Tensor]]]]:
        """
        Generate masks for a batch.

        Args:
            batch: Input batch (B, C, T, H, W)

        Returns:
            batch: Same input batch
            masks: List of (encoder_mask, predictor_masks) for each sample in batch
        """
        batch_size = batch.size(0)
        masks = []

        for _ in range(batch_size):
            encoder_mask, predictor_masks = self.mask_generator()
            masks.append((encoder_mask, predictor_masks))

        return batch, masks


def create_block_mask(
    shape: Tuple[int, int, int],
    block_shape: Tuple[int, int, int],
    block_position: Tuple[int, int, int]
) -> torch.Tensor:
    """
    Create a block mask at specified position.

    Args:
        shape: Full video shape (T, H, W)
        block_shape: Block shape (block_t, block_h, block_w)
        block_position: Block starting position (t_start, h_start, w_start)

    Returns:
        Boolean mask of shape (T*H*W,)
    """
    T, H, W = shape
    block_t, block_h, block_w = block_shape
    t_start, h_start, w_start = block_position

    mask = torch.zeros(T, H, W, dtype=torch.bool)
    mask[
        t_start:t_start + block_t,
        h_start:h_start + block_h,
        w_start:w_start + block_w
    ] = True

    return mask.flatten()
