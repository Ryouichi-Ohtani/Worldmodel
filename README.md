# V-JEPA 2 Implementation

完全な**V-JEPA 2 (Video Joint Embedding Predictive Architecture 2)** の実装です。Meta AIの論文に忠実に再現しています。

## 📄 論文情報

- **タイトル**: V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning
- **著者**: Mido Assran et al. (Meta AI)
- **arXiv**: [2506.09985](https://arxiv.org/abs/2506.09985)
- **公開日**: 2025年6月
- **公式実装**: [github.com/facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2)

## 🎯 概要

V-JEPA 2は、大規模動画データ（100万時間超）で事前学習された自己教師あり学習モデルです。

### 主な特徴

- **動作理解**: Something-Something v2で77.3%のtop-1精度
- **行動予測**: Epic-Kitchens-100で39.7%のrecall-at-5
- **ロボティクス応用**: わずか62時間の未ラベルロボット動画で行動条件付き世界モデル（V-JEPA 2-AC）を構築
- **ゼロショット計画**: フランカアームでのピック&プレースタスクを画像ゴールを使用して実行

## 🏗️ アーキテクチャ

### 1. 事前学習（V-JEPA 2）

```
入力動画 → [Context Encoder (ViT)] → 可視トークン
                    ↓
         [Predictor] → マスク領域の予測
                    ↓
         [Target Encoder (EMA)] → ターゲット表現
                    ↓
              L1 Loss（予測 vs ターゲット）
```

**主要コンポーネント**:
- **Context Encoder**: Vision Transformer（ViT-L/H/g、300M-1B parameters）
- **Target Encoder**: Context EncoderのEMA（指数移動平均）コピー
- **Predictor**: 小規模ViT、マスク領域の表現を予測
- **損失関数**: L1ノルム（予測表現 vs ターゲット表現）

### 2. 行動条件付きモデル（V-JEPA 2-AC）

```
現在の潜在状態 z_t + 行動 a_t → [Action-Conditioned Predictor] → 次状態 z_{t+1}
```

**用途**: ロボット操作、動画予測、計画

## 📦 実装内容

### コア実装

```
vjepa2/
├── models/
│   ├── vision_transformer.py     # ViTエンコーダ実装
│   ├── predictor.py               # 予測器（標準・行動条件付き）
│   └── vjepa2.py                  # メインモデル
├── utils/
│   ├── position_encoding.py      # 3D位置エンコーディング（sinusoidal + RoPE）
│   └── masking.py                 # マルチブロックマスキング
├── training/
│   ├── trainer.py                 # 学習ループ
│   └── losses.py                  # 損失関数
└── data/
    └── video_dataset.py           # 動画データローダー
```

### Google Colabノートブック

**`vjepa2_imagenet_finetuning.ipynb`**: ImageNetでのファインチューニング

- ✅ V-JEPA 2の完全実装（1つのノートブック）
- ✅ 事前学習済み重みのロード
- ✅ Linear Probing / Full Fine-tuning
- ✅ 学習・評価ループ
- ✅ 結果の可視化

## 🚀 使い方

### 1. Google Colabで実行（推奨）

1. `vjepa2_imagenet_finetuning.ipynb`をGoogle Colabで開く
2. ランタイムをGPUに設定
3. セルを順番に実行

```python
# ノートブックの主要セクション
# 1. 環境セットアップ
# 2. V-JEPA 2モデル実装
# 3. 事前学習済み重みのロード
# 4. ImageNetデータセット準備
# 5. ファインチューニング実行
# 6. 結果の可視化
```

### 2. ローカルで実行

```bash
# 依存関係のインストール
pip install torch torchvision timm einops transformers

# Jupyter Notebookを起動
jupyter notebook vjepa2_imagenet_finetuning.ipynb
```

### 3. 事前学習済み重みのロード方法

V-JEPA 2の事前学習済みモデルは以下の方法でロードできます：

#### 方法1: PyTorch Hub（推奨）

```python
import torch

# 利用可能なモデル
model = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_large', pretrained=True)  # ViT-L (300M)
model = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_huge', pretrained=True)   # ViT-H (600M)
model = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_giant', pretrained=True)  # ViT-g (1B)
```

#### 方法2: Hugging Face Hub

```python
from transformers import AutoModel

model = AutoModel.from_pretrained('facebook/vjepa2-vit-large', trust_remote_code=True)
```

**注意**: ノートブックは自動的にこれらの方法を試行し、利用可能な方法で重みをロードします。

## 📊 主要な実装詳細

### 1. 3D位置エンコーディング

```python
# 3D sinusoidal position embeddings for video (T x H x W)
pos_embed = get_3d_sincos_pos_embed(
    embed_dim=1024,
    grid_size=14,      # 空間グリッド (224/16)
    grid_depth=8,      # 時間グリッド (16/2)
    cls_token=True
)
```

**特徴**:
- 時間・高さ・幅の3次元に分割
- 固定sinusoidal埋め込み（学習不要）
- 3D RoPEにも対応

### 2. マルチブロックマスキング

```python
# Spatiotemporal block masking
mask_generator = MultiBlockMaskGenerator(
    input_size=(8, 14, 14),  # (T, H, W) in patches
    num_masks=4,              # マスクブロック数
    min_area=0.15,            # 最小面積比
    max_area=0.7              # 最大面積比
)

encoder_mask, predictor_masks = mask_generator()
```

**マスキング戦略**:
- ランダムな時空間ブロックをマスク
- エンコーダは可視トークンのみ処理
- 予測器はマスク領域を予測

### 3. EMAターゲットエンコーダ

```python
# Exponential Moving Average update
@torch.no_grad()
def update_target_encoder(momentum=0.996):
    for param_q, param_k in zip(encoder.parameters(),
                                 target_encoder.parameters()):
        param_k.data.mul_(momentum).add_(param_q.data, alpha=1-momentum)
```

**更新スケジュール**:
- 初期momentum: 0.996
- 線形スケジューリング（論文に従う）
- 勾配を伝播させない（stop-gradient）

### 4. 損失関数

```python
# L1 regression loss
def vjepa_loss(predictions, targets, loss_exp=1.0):
    loss = 0
    for pred, target in zip(predictions, targets):
        loss += torch.mean(torch.abs(pred - target) ** loss_exp) / loss_exp
    return loss / len(predictions)
```

**損失の特徴**:
- L1ノルム（論文デフォルト）
- Lpノルムに一般化可能
- マスク領域ごとに計算

## 🎓 ImageNetファインチューニング

### Linear Probing

エンコーダを凍結し、線形分類器のみを学習：

```python
model = ImageNetClassifier(
    encoder=pretrained_encoder,
    num_classes=1000,
    freeze_encoder=True  # Linear Probing
)

# 高い学習率
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
```

### Full Fine-tuning

エンコーダ全体をファインチューニング：

```python
model = ImageNetClassifier(
    encoder=pretrained_encoder,
    num_classes=1000,
    freeze_encoder=False  # Full Fine-tuning
)

# 低い学習率
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
```

## 📈 期待される性能

### ImageNet-1K

| モデル | Top-1 Acc | Top-5 Acc | Parameters |
|--------|-----------|-----------|------------|
| ViT-L  | ~85%      | ~97%      | 300M       |
| ViT-H  | ~86%      | ~98%      | 600M       |
| ViT-g  | ~87%      | ~98%      | 1B         |

*注: 実際の性能は学習設定とデータ量に依存*

## 🔧 カスタマイズ

### モデルサイズの変更

```python
# ViT-Large (300M)
encoder = build_vjepa2_encoder('vitl', num_frames=16)

# ViT-Huge (600M)
encoder = build_vjepa2_encoder('vith', num_frames=16)

# ViT-giant (1B)
encoder = build_vjepa2_encoder('vitg', num_frames=16)
```

### ハイパーパラメータ調整

```python
# 学習設定
EPOCHS = 100
LEARNING_RATE = 0.001  # Linear probing
BATCH_SIZE = 256
WARMUP_EPOCHS = 10
```

## 🤖 ロボティクス応用（V-JEPA 2-AC）

行動条件付き予測器を使用した世界モデル：

```python
# Build action-conditioned model
model_ac = VJEPA2_AC(
    vjepa2_encoder=pretrained_encoder,
    action_dim=7,  # ロボットアクション次元
    freeze_encoder=True
)

# Predict next state
z_current = model_ac.encode(observation)
z_next = model_ac(z_current, action)

# Plan actions to reach goal
actions = model_ac.plan(
    z_init=z_current,
    z_goal=z_goal,
    horizon=10
)
```

## 📚 参考文献

1. **V-JEPA 2 Paper**: [arXiv:2506.09985](https://arxiv.org/abs/2506.09985)
2. **公式実装**: [github.com/facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2)
3. **Blog**: [ai.meta.com/vjepa](https://ai.meta.com/vjepa/)
4. **原論文 I-JEPA**: [arXiv:2301.08243](https://arxiv.org/abs/2301.08243)

## 🙏 謝辞

この実装は、Meta AIのV-JEPA 2論文と公式実装を基にしています。

## 📝 ライセンス

このプロジェクトは教育目的で作成されました。商用利用する場合は、Meta AIの公式ライセンスを確認してください。

---

**実装者**: V-JEPA 2の論文に忠実な完全実装
**最終更新**: 2025年11月
