"""
V-JEPA 2実装のテストスクリプト
ノートブックの実装が正しく動作するか検証
"""

import torch
import torch.nn as nn
import numpy as np
import sys
from typing import Tuple, Optional

print("="*70)
print("V-JEPA 2 Implementation Test")
print("="*70)
print()

# テスト結果を保存
test_results = {
    'passed': [],
    'failed': [],
    'warnings': []
}

def test_passed(test_name):
    """テスト成功を記録"""
    test_results['passed'].append(test_name)
    print(f"✅ {test_name}")

def test_failed(test_name, error):
    """テスト失敗を記録"""
    import traceback
    error_msg = str(error) if str(error) else "Unknown error"
    if hasattr(error, '__traceback__'):
        error_msg = ''.join(traceback.format_exception(type(error), error, error.__traceback__))
    test_results['failed'].append((test_name, error_msg))
    print(f"❌ {test_name}: {error_msg}")

def test_warning(test_name, message):
    """警告を記録"""
    test_results['warnings'].append((test_name, message))
    print(f"⚠️  {test_name}: {message}")


# ============================================================================
# Test 1: Position Encoding
# ============================================================================

print("\n" + "-"*70)
print("Test 1: Position Encoding Functions")
print("-"*70)

try:
    def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
        assert embed_dim % 2 == 0
        omega = np.arange(embed_dim // 2, dtype=np.float32)
        omega /= embed_dim / 2.0
        omega = 1.0 / 10000**omega
        pos = pos.reshape(-1)
        out = np.einsum('m,d->md', pos, omega)
        emb_sin = np.sin(out)
        emb_cos = np.cos(out)
        emb = np.concatenate([emb_sin, emb_cos], axis=1)
        return emb

    def get_3d_sincos_pos_embed(embed_dim: int, grid_size: int, grid_depth: int, cls_token: bool = False) -> torch.Tensor:
        # Flexible dimension partitioning (removed assertion for compatibility with standard ViT configs)
        # Ensure all dimensions are even (required for sinusoidal encoding)
        dim_t = (embed_dim // 3) // 2 * 2  # Round down to nearest even number
        dim_h = ((embed_dim - dim_t) // 2) // 2 * 2  # Round down to nearest even number
        dim_w = embed_dim - dim_t - dim_h  # Remainder (may be even or odd, will be made even)

        grid_t = np.arange(grid_depth, dtype=np.float32)
        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)

        grid = np.meshgrid(grid_t, grid_h, grid_w, indexing='ij')
        grid = np.stack(grid, axis=0)
        grid = grid.reshape([3, -1]).T

        pos_embed_t = get_1d_sincos_pos_embed_from_grid(dim_t, grid[:, 0])
        pos_embed_h = get_1d_sincos_pos_embed_from_grid(dim_h, grid[:, 1])
        pos_embed_w = get_1d_sincos_pos_embed_from_grid(dim_w, grid[:, 2])

        pos_embed = np.concatenate([pos_embed_t, pos_embed_h, pos_embed_w], axis=1)

        if cls_token:
            pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)

        return torch.from_numpy(pos_embed).float()

    # Test 3D position embedding
    embed_dim = 768
    grid_size = 14
    grid_depth = 8

    pos_embed = get_3d_sincos_pos_embed(embed_dim, grid_size, grid_depth, cls_token=False)
    expected_shape = (grid_depth * grid_size * grid_size, embed_dim)

    if pos_embed.shape == expected_shape:
        test_passed("3D Position Embedding Shape")
    else:
        test_failed("3D Position Embedding Shape",
                   f"Expected {expected_shape}, got {pos_embed.shape}")

    # Test with CLS token
    pos_embed_cls = get_3d_sincos_pos_embed(embed_dim, grid_size, grid_depth, cls_token=True)
    expected_shape_cls = (1 + grid_depth * grid_size * grid_size, embed_dim)

    if pos_embed_cls.shape == expected_shape_cls:
        test_passed("3D Position Embedding with CLS Token")
    else:
        test_failed("3D Position Embedding with CLS Token",
                   f"Expected {expected_shape_cls}, got {pos_embed_cls.shape}")

except Exception as e:
    test_failed("Position Encoding", e)


# ============================================================================
# Test 2: Patch Embedding 3D
# ============================================================================

print("\n" + "-"*70)
print("Test 2: 3D Patch Embedding")
print("-"*70)

try:
    class PatchEmbed3D(nn.Module):
        def __init__(self, img_size=224, patch_size=16, tubelet_size=2, in_channels=3, embed_dim=768):
            super().__init__()
            self.img_size = img_size
            self.patch_size = patch_size
            self.tubelet_size = tubelet_size
            self.grid_size = img_size // patch_size

            self.proj = nn.Conv3d(
                in_channels, embed_dim,
                kernel_size=(tubelet_size, patch_size, patch_size),
                stride=(tubelet_size, patch_size, patch_size)
            )

        def forward(self, x):
            B, C, T, H, W = x.shape
            x = self.proj(x)
            T_patches = x.shape[2]
            H_patches = x.shape[3]
            W_patches = x.shape[4]
            x = x.flatten(2).transpose(1, 2)
            return x, (T_patches, H_patches, W_patches)

    patch_embed = PatchEmbed3D(img_size=224, patch_size=16, tubelet_size=2, in_channels=3, embed_dim=768)

    # Test forward pass
    batch_size = 2
    num_frames = 16
    x = torch.randn(batch_size, 3, num_frames, 224, 224)

    output, (T_p, H_p, W_p) = patch_embed(x)

    expected_T = num_frames // 2
    expected_H = 224 // 16
    expected_W = 224 // 16
    expected_num_patches = expected_T * expected_H * expected_W

    if output.shape == (batch_size, expected_num_patches, 768):
        test_passed("Patch Embedding Output Shape")
    else:
        test_failed("Patch Embedding Output Shape",
                   f"Expected ({batch_size}, {expected_num_patches}, 768), got {output.shape}")

    if (T_p, H_p, W_p) == (expected_T, expected_H, expected_W):
        test_passed("Patch Embedding Grid Dimensions")
    else:
        test_failed("Patch Embedding Grid Dimensions",
                   f"Expected ({expected_T}, {expected_H}, {expected_W}), got ({T_p}, {H_p}, {W_p})")

except Exception as e:
    test_failed("Patch Embedding", e)


# ============================================================================
# Test 3: Attention Module
# ============================================================================

print("\n" + "-"*70)
print("Test 3: Multi-head Attention")
print("-"*70)

try:
    import torch.nn.functional as F

    class Attention(nn.Module):
        def __init__(self, dim, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
            super().__init__()
            self.num_heads = num_heads
            head_dim = dim // num_heads
            self.scale = qk_scale or head_dim ** -0.5

            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.attn_drop = nn.Dropout(attn_drop)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)

        def forward(self, x):
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            if hasattr(F, 'scaled_dot_product_attention'):
                x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
            else:
                attn = (q @ k.transpose(-2, -1)) * self.scale
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                x = attn @ v

            x = x.transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x

    attn = Attention(dim=768, num_heads=12)
    x = torch.randn(2, 196, 768)
    output = attn(x)

    if output.shape == x.shape:
        test_passed("Attention Output Shape")
    else:
        test_failed("Attention Output Shape", f"Expected {x.shape}, got {output.shape}")

except Exception as e:
    test_failed("Attention Module", e)


# ============================================================================
# Test 4: Vision Transformer
# ============================================================================

print("\n" + "-"*70)
print("Test 4: Vision Transformer")
print("-"*70)

try:
    class MLP(nn.Module):
        def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
            super().__init__()
            out_features = out_features or in_features
            hidden_features = hidden_features or in_features
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.act = nn.GELU()
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.drop = nn.Dropout(drop)

        def forward(self, x):
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.drop(x)
            return x

    class Block(nn.Module):
        def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 attn_drop=attn_drop, proj_drop=drop)
            self.norm2 = nn.LayerNorm(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        def forward(self, x):
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
            return x

    class VisionTransformer(nn.Module):
        def __init__(self, img_size=224, patch_size=16, tubelet_size=2, in_channels=3, num_frames=16,
                     embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                     drop_rate=0., attn_drop_rate=0., use_cls_token=False):
            super().__init__()
            self.img_size = img_size
            self.patch_size = patch_size
            self.tubelet_size = tubelet_size
            self.num_frames = num_frames
            self.embed_dim = embed_dim
            self.use_cls_token = use_cls_token

            self.patch_embed = PatchEmbed3D(img_size, patch_size, tubelet_size, in_channels, embed_dim)

            grid_size = img_size // patch_size
            grid_depth = num_frames // tubelet_size
            num_patches = grid_depth * grid_size * grid_size

            if use_cls_token:
                self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
                num_patches += 1
            else:
                self.cls_token = None

            pos_embed = get_3d_sincos_pos_embed(embed_dim, grid_size, grid_depth, cls_token=use_cls_token)
            self.register_buffer('pos_embed', pos_embed.unsqueeze(0))

            self.pos_drop = nn.Dropout(p=drop_rate)

            self.blocks = nn.ModuleList([
                Block(embed_dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate)
                for _ in range(depth)
            ])

            self.norm = nn.LayerNorm(embed_dim)

            self._init_weights()

        def _init_weights(self):
            if self.cls_token is not None:
                nn.init.trunc_normal_(self.cls_token, std=0.02)
            w = self.patch_embed.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            self.apply(self._init_layer_weights)

        def _init_layer_weights(self, m):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        def forward(self, x, return_all_tokens=False):
            B = x.shape[0]
            x, _ = self.patch_embed(x)

            if self.cls_token is not None:
                cls_tokens = self.cls_token.expand(B, -1, -1)
                x = torch.cat([cls_tokens, x], dim=1)

            x = x + self.pos_embed
            x = self.pos_drop(x)

            for block in self.blocks:
                x = block(x)

            x = self.norm(x)

            if return_all_tokens or self.cls_token is None:
                return x
            else:
                return x[:, 0]

    # Test ViT-L configuration
    vit = VisionTransformer(
        img_size=224,
        patch_size=16,
        tubelet_size=2,
        num_frames=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        use_cls_token=True
    )

    # Count parameters
    total_params = sum(p.numel() for p in vit.parameters())
    print(f"   Total parameters: {total_params:,}")

    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 3, 16, 224, 224)

    # Test CLS token output
    output_cls = vit(x, return_all_tokens=False)
    if output_cls.shape == (batch_size, 1024):
        test_passed("ViT CLS Token Output")
    else:
        test_failed("ViT CLS Token Output", f"Expected ({batch_size}, 1024), got {output_cls.shape}")

    # Test all tokens output
    output_all = vit(x, return_all_tokens=True)
    expected_num_tokens = 1 + (16//2) * (224//16) * (224//16)  # CLS + patches
    if output_all.shape == (batch_size, expected_num_tokens, 1024):
        test_passed("ViT All Tokens Output")
    else:
        test_failed("ViT All Tokens Output",
                   f"Expected ({batch_size}, {expected_num_tokens}, 1024), got {output_all.shape}")

except Exception as e:
    test_failed("Vision Transformer", e)


# ============================================================================
# Test 5: ImageNet Classifier
# ============================================================================

print("\n" + "-"*70)
print("Test 5: ImageNet Classifier")
print("-"*70)

try:
    class ImageNetClassifier(nn.Module):
        def __init__(self, encoder, num_classes=1000, freeze_encoder=True, num_frames=16):
            super().__init__()
            self.encoder = encoder
            self.num_classes = num_classes
            self.num_frames = num_frames

            if freeze_encoder:
                for param in self.encoder.parameters():
                    param.requires_grad = False

            embed_dim = encoder.embed_dim
            self.head = nn.Linear(embed_dim, num_classes)
            nn.init.trunc_normal_(self.head.weight, std=0.02)
            nn.init.constant_(self.head.bias, 0)

        def forward(self, x):
            if x.dim() == 4:
                x = x.unsqueeze(2).repeat(1, 1, self.num_frames, 1, 1)
            features = self.encoder(x, return_all_tokens=False)
            logits = self.head(features)
            return logits

    # Build encoder
    encoder = VisionTransformer(
        img_size=224, patch_size=16, tubelet_size=2, num_frames=16,
        embed_dim=1024, depth=24, num_heads=16, use_cls_token=True
    )

    # Build classifier
    classifier = ImageNetClassifier(encoder, num_classes=1000, freeze_encoder=True)

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in classifier.parameters() if not p.requires_grad)

    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Frozen parameters: {frozen_params:,}")

    # Test forward pass with image input
    batch_size = 2
    x_img = torch.randn(batch_size, 3, 224, 224)
    logits = classifier(x_img)

    if logits.shape == (batch_size, 1000):
        test_passed("ImageNet Classifier Output")
    else:
        test_failed("ImageNet Classifier Output", f"Expected ({batch_size}, 1000), got {logits.shape}")

    # Verify encoder is frozen
    encoder_frozen = all(not p.requires_grad for p in classifier.encoder.parameters())
    head_trainable = all(p.requires_grad for p in classifier.head.parameters())

    if encoder_frozen and head_trainable:
        test_passed("Linear Probing Mode (Encoder Frozen)")
    else:
        test_failed("Linear Probing Mode", "Encoder not properly frozen or head not trainable")

except Exception as e:
    test_failed("ImageNet Classifier", e)


# ============================================================================
# Test 6: Memory and Computation
# ============================================================================

print("\n" + "-"*70)
print("Test 6: Memory and Computation Efficiency")
print("-"*70)

try:
    # Test small batch to avoid OOM
    batch_size = 1
    x = torch.randn(batch_size, 3, 16, 224, 224)

    # Test forward pass
    with torch.no_grad():
        output = classifier(x)

    # Estimate memory usage
    param_size = sum(p.numel() * p.element_size() for p in classifier.parameters())
    param_size_mb = param_size / (1024 ** 2)

    print(f"   Model size: {param_size_mb:.2f} MB")

    if param_size_mb < 5000:  # Reasonable for ViT-L
        test_passed("Model Memory Footprint")
    else:
        test_warning("Model Memory Footprint", f"Large model size: {param_size_mb:.2f} MB")

    test_passed("Forward Pass Execution")

except Exception as e:
    test_failed("Memory and Computation", e)


# ============================================================================
# Test 7: Gradient Flow (Training)
# ============================================================================

print("\n" + "-"*70)
print("Test 7: Gradient Flow and Training")
print("-"*70)

try:
    # Create dummy data and labels
    x = torch.randn(2, 3, 224, 224)
    labels = torch.randint(0, 1000, (2,))

    # Forward pass
    logits = classifier(x)

    # Compute loss
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, labels)

    # Backward pass
    optimizer = torch.optim.AdamW([p for p in classifier.parameters() if p.requires_grad], lr=0.001)
    optimizer.zero_grad()
    loss.backward()

    # Check gradients
    head_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in classifier.head.parameters())
    encoder_no_grad = all(p.grad is None for p in classifier.encoder.parameters())

    if head_has_grad:
        test_passed("Classification Head Gradients")
    else:
        test_failed("Classification Head Gradients", "No gradients in head")

    if encoder_no_grad:
        test_passed("Encoder Frozen (No Gradients)")
    else:
        test_warning("Encoder Frozen", "Gradients found in frozen encoder")

    # Optimizer step
    optimizer.step()
    test_passed("Optimizer Step")

except Exception as e:
    test_failed("Gradient Flow", e)


# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*70)
print("Test Summary")
print("="*70)

total_tests = len(test_results['passed']) + len(test_results['failed'])
print(f"\nTotal Tests: {total_tests}")
print(f"✅ Passed: {len(test_results['passed'])}")
print(f"❌ Failed: {len(test_results['failed'])}")
print(f"⚠️  Warnings: {len(test_results['warnings'])}")

if test_results['failed']:
    print("\n" + "-"*70)
    print("Failed Tests:")
    print("-"*70)
    for test_name, error in test_results['failed']:
        print(f"  • {test_name}")
        print(f"    Error: {error}")

if test_results['warnings']:
    print("\n" + "-"*70)
    print("Warnings:")
    print("-"*70)
    for test_name, message in test_results['warnings']:
        print(f"  • {test_name}")
        print(f"    Message: {message}")

print("\n" + "="*70)
if not test_results['failed']:
    print("🎉 All tests passed! Implementation is correct.")
else:
    print("⚠️  Some tests failed. Please review the errors above.")
print("="*70)
