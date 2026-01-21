# PSD Dataset 学習結果

PolarFreeモデルをPSD（Polarization Specular Dataset）で学習した結果。

---

## 学習設定

### Stage 1
| 項目 | 値 |
|------|-----|
| イテレーション | 100,000 |
| バッチサイズ | 8 |
| 学習率 | 1e-4 (CosineAnnealing) |
| 学習時間 | ~8.5時間 |
| GPU | ~89GB使用 |

### Stage 2
| 項目 | 値 |
|------|-----|
| イテレーション | 50,000 (300,000中で停止) |
| バッチサイズ | 8 |
| 学習率 | 2e-4 (CosineAnnealing) |
| 学習時間 | ~15時間 |
| Diffusion timesteps | 8 |

---

## 定量的結果

### Validation Set (40 samples)
| Stage | Iteration | PSNR (dB) |
|-------|-----------|-----------|
| Stage 1 | 95,000 | 37.27 (Best) |
| Stage 2 | 50,000 | **37.58** |

### Test Set (50 samples)
| Stage | Iteration | PSNR (dB) | SSIM |
|-------|-----------|-----------|------|
| Stage 1 | 100,000 | **34.85** | **0.9918** |
| Stage 2 | 50,000 | 34.45 | 0.9914 |

**注**: Test setではStage 1が若干高いPSNRを示す。これは以下の要因が考えられる:
- Stage 2は50kイテレーションで学習を停止（100kで再評価推奨）
- Diffusionモデルの推論時のランダム性
- Test setとValidation setの分布の違い

---

## 定性的結果

比較画像は `results/stage_comparison/` に保存。

### サンプル比較

各画像は左から: Input | Ground Truth | Stage 1 (100k) | Stage 2 (50k)

**観察**:
1. 両ステージとも鏡面反射（ハイライト）の除去に成功
2. Stage 1とStage 2の出力は視覚的にほぼ同等
3. 色調の再現性が高い

---

## 推論時間

画像サイズ: 512 x 768 (PSD Datasetの標準サイズ)

| Stage | Device | Time (ms) | 備考 |
|-------|--------|-----------|------|
| Stage 1 | GPU (H100) | **325** | - |
| Stage 1 | CPU | 30,872 | ~31秒 |
| Stage 2 | GPU (H100) | **271** | Diffusion 8 steps |
| Stage 2 | CPU | 22,397 | ~22秒 |

**注**:
- GPU: NVIDIA H100 (97GB)
- CPU: 測定環境のCPU
- Stage 2はDiffusion推論を含むが、Stage 1より若干速い（ネットワーク構成の違い）

---

## チェックポイント

### Stage 1
```
experiments/psd_stage1/models/
├── net_g_100000.pth      # Generator (72MB)
├── net_g_latest.pth      # → net_g_100000.pth へのコピー
├── net_le_100000.pth     # Latent Encoder (2.5MB)
└── net_le_latest.pth     # → net_le_100000.pth へのコピー
```

| ファイル | サイズ | 説明 |
|----------|--------|------|
| net_g_*.pth | ~72MB | Transformer Generator |
| net_le_*.pth | ~2.5MB | Latent Encoder (Stage 1) |

### Stage 2
```
experiments/psd_stage2/models/
├── net_g_50000.pth       # Generator (72MB)
├── net_le_dm_50000.pth   # Latent Encoder for Diffusion (2.4MB)
└── net_d_50000.pth       # Denoising Network (10MB)
```

| ファイル | サイズ | 説明 |
|----------|--------|------|
| net_g_*.pth | ~72MB | Transformer Generator (Stage 1から継続学習) |
| net_le_dm_*.pth | ~2.4MB | Latent Encoder (Stage 2用, in_chans=9) |
| net_d_*.pth | ~10MB | Denoising Network (Diffusion用) |

### 重みの構造

```python
# 各.pthファイルの構造
{
    'params': OrderedDict,  # モデルのstate_dict
    'epoch': int,           # (training_stateのみ)
    'iter': int,            # (training_stateのみ)
    'optimizers': dict,     # (training_stateのみ)
}

# 使用例
checkpoint = torch.load('net_g_100000.pth')
model.load_state_dict(checkpoint['params'])
```

---

## 使用方法

### テスト実行
```bash
# Stage 1
python scripts/test_psd_checkpoint.py --stage 1 --iter 100000 --save_images

# Stage 2
python scripts/test_psd_checkpoint.py --stage 2 --iter 50000 --save_images

# 比較
python scripts/test_psd_checkpoint.py --stage 1 --compare_iters 25000,50000,75000,100000 --save_images
```

---

## 学習ログ

### Stage 1 Loss推移
| Iteration | l_pix | l_vgg | l_phase |
|-----------|-------|-------|---------|
| 5,000 | 1.08e-02 | 1.54e-03 | 2.02e-01 |
| 50,000 | 6.88e-03 | 8.35e-04 | 1.91e-01 |
| 100,000 | 6.75e-03 | 8.08e-04 | 1.90e-01 |

### Stage 2 Loss推移
| Iteration | l_pix | l_pix_diff | l_vgg | l_phase |
|-----------|-------|------------|-------|---------|
| 5,000 | 8.73e-03 | 0.357 | 1.17e-03 | 0.213 |
| 25,000 | 7.32e-03 | 0.317 | 1.02e-03 | 0.247 |
| 50,000 | 7.79e-03 | 0.309 | 7.71e-04 | 0.181 |

---

## 今後の改善点

1. **Stage 2を100k以上まで学習**
   - 50kは早期停止、完全な収束には不十分

2. **TRS特徴量の活用**
   - 現在はAoLP/DoLPのみ使用
   - IrawD, Ichro, Isvを入力に追加する改良

3. **損失関数の追加**
   - SpecularSparseLoss, PolarConsistencyLoss等（実装済み、未使用）

4. **データ拡張**
   - PSDは888グループと小規模
   - PolaRGBとの混合学習を検討

---

## 参考

- PSD Dataset: `/data2/PSD_Dataset/PSD_Dataset/`
- 設定ファイル: `options/train/psd_stage1.yml`, `options/train/psd_stage2.yml`
- 開発ログ: `logs/Log_2026-01-20.md`, `logs/Log_2026-01-21.md`
