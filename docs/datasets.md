# データセット

このプロジェクトで使用するデータセットの情報とパス。

---

## サーバー別パス

| サーバー | データルート |
|---------|-------------|
| oasis | `/data/nishida/` |
| fleetwood | `/data2/` |

以下のパスは `{DATA_ROOT}` で記載。環境に応じて読み替えてください。

---

## 1. PSD Dataset (Polarization Specular Dataset)

**用途**: 鏡面反射除去（Specular Highlight Removal）

### パス
```
{DATA_ROOT}/PSD_Dataset/
├── PSD_Train/
│   ├── aligned/           # アラインされた画像
│   │   └── group-XXXX-idx-YY.png
│   └── unaligned/         # アラインされていない画像
├── PSD_val/
│   └── group-XXXX-idx-YY.png  # 直接配置（サブディレクトリなし）
└── PSD_Test/
    └── test-XXXX-idx-YY.png   # プレフィックスが異なる
```

### データ量
| Split | グループ数 | 画像数 (12角度/グループ) |
|-------|-----------|------------------------|
| Train | 642 | 7,704 |
| val | 40 | 480 |
| Test | 206 | 2,472 |
| **合計** | **888** | **10,656** |

### 偏光角度
- 12角度: 30°間隔 (idx-01 ~ idx-12)
- 4角度への変換マッピング:
  - 0° → idx-01
  - 45° → interpolate(idx-02, idx-03)
  - 90° → idx-04
  - 135° → interpolate(idx-05, idx-06)

### 設定ファイル
- `options/train/psd_stage1.yml`
- `options/train/psd_stage2.yml`

### データセットクラス
- `polarfree/data/psd_dataset.py` → `PSDDataset`

---

## 2. PolaRGB Dataset

**用途**: ガラス反射除去（Reflection Removal）- オリジナルPolarFreeタスク

### パス
```
{DATA_ROOT}/PolaRGB/
├── train/
│   ├── easy/
│   └── hard/
└── test/
    ├── easy/
    └── hard/
```

### データ量
| Split | 画像数 |
|-------|--------|
| Train | ~6,312 |
| Test | ~188 |
| **合計** | **~6,500** |

### 設定ファイル
- `options/train/ours_stage1.yml`
- `options/train/ours_stage2.yml`
- `options/test/test.yml`

### データセットクラス
- `polarfree/data/paired_image_polar_dataset.py` → `PairedImagePolarDataset`

---

## 3. PolarSpecular Dataset (予定)

**用途**: 独自の鏡面反射除去データセット（将来構築予定）

### 予定パス
```
PolarSpecular/
├── train/
│   ├── easy/              # 照明色≠物体色
│   └── hard/              # 照明色≈物体色
└── test/
```

### 設定ファイル
- `options/train/specular_stage1.yml`
- `options/train/specular_stage2.yml`
- `options/test/specular_test.yml`

---

## データセット比較

| Dataset | タスク | グループ/画像数 | 偏光角度 | GT |
|---------|--------|----------------|----------|-----|
| PolaRGB | 反射除去 | ~6,500枚 | 4角度 | Transmission |
| PSD | 鏡面除去 | 888グループ | 12角度→4角度 | Diffuse |
| PolarSpecular | 鏡面除去 | (予定) | 4角度 | Diffuse |

---

## 環境設定

### Conda環境
```bash
conda activate polarfree
```

### 環境ファイル
- `environment.yml` - 完全な依存関係リスト（最新の環境と一致確認済み: 2026-01-21）

### 主要パッケージ
| Package | Version |
|---------|---------|
| Python | 3.10.19 |
| PyTorch | 2.11.0.dev (CUDA 12.8) |
| basicsr | 1.4.2 |
| mmcv | 2.1.0 |
| numpy | 2.2.6 |
| opencv-python | 4.13.0.90 |

---

## 学習実行例

### PSD Dataset (Stage 1)
```bash
python train.py -opt options/train/psd_stage1.yml
```

### PSD Dataset (Stage 2)
```bash
python train.py -opt options/train/psd_stage2.yml
```

### PolaRGB Dataset (オリジナル)
```bash
python train.py -opt options/train/ours_stage1.yml
```
