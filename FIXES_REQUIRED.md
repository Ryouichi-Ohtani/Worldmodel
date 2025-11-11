# V-JEPA 2 Implementation - 修正が必要な問題

## 🔴 Critical Issue: Embed Dim と Position Encoding の互換性

### 問題

ノートブック内の`get_3d_sincos_pos_embed`関数は`embed_dim`が3で割り切れることを要求していますが、ViT-Lの標準設定は`embed_dim=1024`で、これは3で割り切れません（1024 % 3 = 1）。

**エラー発生箇所**:
```python
def get_3d_sincos_pos_embed(embed_dim, grid_size, grid_depth, cls_token=False):
    assert embed_dim % 3 == 0  # ←この行でエラー！
```

**使用している設定（cell-8）**:
```python
'vitl': {
    'embed_dim': 1024,  # 3で割り切れない！
    ...
}
```

### 解決策

位置エンコーディング関数を修正して、3で割り切れない`embed_dim`にも対応させます：

```python
def get_3d_sincos_pos_embed(
    embed_dim: int,
    grid_size: int,
    grid_depth: int,
    cls_token: bool = False
) -> torch.Tensor:
    """
    3D sinusoidal position embeddings for video (T x H x W).
    Flexible dimension partitioning for embed_dim not divisible by 3.
    """
    # Flexibly partition embedding dimension (no assertion)
    dim_t = embed_dim // 3
    dim_h = (embed_dim - dim_t) // 2
    dim_w = embed_dim - dim_t - dim_h

    # 以下同じ...
```

### 修正手順

1. ノートブックのcell-6の`get_3d_sincos_pos_embed`関数を更新
2. `assert embed_dim % 3 == 0`の行を削除
3. Python実装ファイルも同様に修正

### 影響範囲

- [vjepa2_imagenet_finetuning.ipynb](vjepa2_imagenet_finetuning.ipynb) - セル6
- [vjepa2/utils/position_encoding.py](vjepa2/utils/position_encoding.py) - 54行目

## ✅ 修正後のテスト

修正後、以下のテストがすべて通過する必要があります：

- ✅ 3D Position Embedding Shape
- ✅ 3D Position Embedding with CLS Token
- ✅ Patch Embedding Output Shape
- ✅ Patch Embedding Grid Dimensions
- ✅ Attention Output Shape
- ✅ Vision Transformer (embed_dim=1024で動作)
- ✅ ImageNet Classifier
- ✅ Memory and Computation
- ✅ Gradient Flow

## 📝 推奨される修正

### オプション1: Assert削除（推奨）

柔軟な次元分割を許可し、任意の`embed_dim`に対応：

```python
# Before
assert embed_dim % 3 == 0

# After
# Removed assertion - flexible partitioning
dim_t = embed_dim // 3
dim_h = (embed_dim - dim_t) // 2
dim_w = embed_dim - dim_t - dim_h
```

### オプション2: Embed Dim調整（非推奨）

標準から外れるため推奨しません：

```python
# ViT-Lの設定を変更
'vitl': {
    'embed_dim': 1023,  # 3の倍数（341 * 3）
    'depth': 24,
    'num_heads': 12,  # 1023 / 12 = 85.25（割り切れない！）
}
```

→ `num_heads`との互換性問題が発生するため不適切

### オプション3: Position Encoding方式の変更

学習可能なposition embeddingを使用：

```python
# 固定sinusoidalの代わりに学習可能に
self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
nn.init.trunc_normal_(self.pos_embed, std=0.02)
```

## 🎯 最終推奨

**オプション1（Assert削除）を実装**してください。

これにより：
- 標準のViT設定（embed_dim=1024）が使用可能
- 柔軟性が向上（任意のembed_dimに対応）
- 論文の意図に最も近い実装

## 修正コード

```python
def get_3d_sincos_pos_embed(
    embed_dim: int,
    grid_size: int,
    grid_depth: int,
    cls_token: bool = False
) -> torch.Tensor:
    """
    3D sinusoidal position embeddings for video (T x H x W).

    Note: Flexibly handles embed_dim not divisible by 3.
    Dimensions are partitioned as: dim_t = embed_dim // 3,
    dim_h = (embed_dim - dim_t) // 2, dim_w = remainder
    """
    # Flexible dimension partitioning (removed assertion)
    dim_t = embed_dim // 3
    dim_h = (embed_dim - dim_t) // 2
    dim_w = embed_dim - dim_t - dim_h

    # Generate 3D grid
    grid_t = np.arange(grid_depth, dtype=np.float32)
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)

    grid = np.meshgrid(grid_t, grid_h, grid_w, indexing='ij')
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([3, -1]).T

    # Generate embeddings
    pos_embed_t = get_1d_sincos_pos_embed_from_grid(dim_t, grid[:, 0])
    pos_embed_h = get_1d_sincos_pos_embed_from_grid(dim_h, grid[:, 1])
    pos_embed_w = get_1d_sincos_pos_embed_from_grid(dim_w, grid[:, 2])

    pos_embed = np.concatenate([pos_embed_t, pos_embed_h, pos_embed_w], axis=1)

    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)

    return torch.from_numpy(pos_embed).float()
```
