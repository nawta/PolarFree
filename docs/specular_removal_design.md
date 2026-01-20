# PolarFree for Specular Highlight Removal - Design Document

## 1. Overview

This document describes the design of PolarFree adapted for specular highlight removal. The original PolarFree was designed for glass reflection removal; this adaptation leverages the same architecture for removing specular highlights from polarization images.

## 2. Problem Comparison

### 2.1 Glass Reflection (Original PolarFree)

```
M = αt·T + αr·R
```

- **M**: Mixed image (observed)
- **T**: Transmission layer (scene behind glass) - target output
- **R**: Reflection layer (unwanted)
- Both T and R have polarization characteristics
- Goal: Recover T from M

### 2.2 Specular Highlight (This Adaptation)

```
I(φ) = Id + Isc + Isv·cos(2(φ - α))
     = Ic + Isv·cos(2(φ - α))
```

- **Id**: Diffuse reflection (unpolarized, constant across angles)
- **Isc**: Specular unpolarized component (constant)
- **Isv**: Specular polarized component (varies with polarizer angle)
- **Ic = Id + Isc**: Constant component
- **α**: Angle of polarization
- Goal: Recover Id (diffuse component) from polarization images

### 2.3 Key Physical Insight

The polarization-varying component **Isv comes 100% from specular reflection**. This is the foundation for polarization-based specular removal:

- Diffuse reflection: Unpolarized (constant across all polarizer angles)
- Specular reflection: Partially polarized (intensity varies with polarizer angle)

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Stage 1                                  │
│  ┌──────────────┐    ┌─────────────────┐    ┌──────────────┐   │
│  │ Polarization │    │ Latent Encoder  │    │ Transformer  │   │
│  │ Features     │───>│ (network_le)    │───>│ (network_g)  │──>│ Diffuse Prior
│  │ (12 ch)      │    │ 12ch -> 64dim   │    │              │   │
│  └──────────────┘    └─────────────────┘    └──────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                v (frozen weights)
┌─────────────────────────────────────────────────────────────────┐
│                         Stage 2                                  │
│  ┌──────────────┐    ┌─────────────────┐    ┌──────────────┐   │
│  │ Condition    │    │ Latent Encoder  │    │ Diffusion    │   │
│  │ Features     │───>│ (network_le_dm) │───>│ (network_d)  │──>│ Refined Diffuse
│  │ (9 ch)       │    │ 9ch -> 64dim    │    │              │   │
│  └──────────────┘    └─────────────────┘    └──────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## 4. Input Feature Design

### 4.1 Stage 1 Input (12 channels)

| Channel | Content | Purpose |
|---------|---------|---------|
| 1-3 | RGB (observed image) | Original image information |
| 4-6 | IrawD (per channel) | Initial diffuse estimate |
| 7-9 | Ichro (polarization chromaticity) | Illumination-independent color |
| 10 | Isv (grayscale) | Specular intensity map |
| 11 | AOLP | Angle of linear polarization |
| 12 | DOLP | Degree of linear polarization |

### 4.2 Stage 2 Input (9 channels)

| Channel | Content | Purpose |
|---------|---------|---------|
| 1-3 | RGB | Original image |
| 4-6 | IrawD | Diffuse estimate |
| 7 | Isv | Specular intensity |
| 8 | DOLP | Polarization degree |
| 9 | AOLP | Polarization angle |

### 4.3 Feature Computation (TRS Processing)

From 4 polarization images I₀, I₄₅, I₉₀, I₁₃₅:

```python
# Stokes parameters
S0 = (I0 + I45 + I90 + I135) / 2  # Total intensity
S1 = I0 - I90                      # Horizontal vs vertical
S2 = I45 - I135                    # Diagonal difference

# Derived quantities
Ic = S0                            # Constant component
Isv = sqrt(S1² + S2²) / 2          # Varying amplitude (specular)

# DOLP and AOLP
DOLP = sqrt(S1² + S2²) / S0        # Degree of polarization
AOLP = 0.5 * arctan2(S2, S1)       # Angle of polarization

# TRS-based estimates
IrawD = Ic - Isv                   # Raw diffuse estimate (min intensity)
Ichro = IrawD / sum(IrawD)         # Polarization chromaticity
```

## 5. Network Components

### 5.1 Latent Encoder (`network_le`)

```yaml
network_le:
  type: latent_encoder_gelu
  in_chans: 12          # Polarization features
  embed_dim: 64
  block_num: 6
  group: 4
  stage: 1
  patch_expansion: 0.5
  channel_expansion: 4
```

- Encodes polarization features into latent space
- Uses GELU activation for smooth gradients
- Patch-based processing with configurable expansion

### 5.2 Transformer (`network_g`)

```yaml
network_g:
  type: Transformer
  inp_channels: 3
  out_channels: 3
  dim: 48
  num_blocks: [3,4,4,4]
  num_refinement_blocks: 4
  heads: [1,2,4,8]
  ffn_expansion_factor: 2.66
  embed_dim: 64
  group: 4
```

- Multi-scale transformer architecture
- Receives latent codes from encoder
- Outputs diffuse prior image

### 5.3 Diffusion Model (`network_d`)

```yaml
network_d:
  type: denoising
  in_channel: 256       # embed_dim * 4
  out_channel: 256
  inner_channel: 512
  block_num: 4
  group: 4
  patch_expansion: 0.5
  channel_expansion: 2

diffusion_schedule:
  schedule: linear
  timesteps: 8
  linear_start: 0.1
  linear_end: 0.99
```

- Refines Stage 1 output using diffusion process
- 8 timesteps for efficiency
- Linear noise schedule

## 6. Loss Functions

### 6.1 Standard Losses (from Original PolarFree)

| Loss | Weight | Purpose |
|------|--------|---------|
| L1 Loss | 1.0 | Pixel-level reconstruction |
| VGG Loss | 0.02 | Perceptual quality |
| TV Loss | 0.0005 | Smoothness |
| Phase Loss | 0.1 | Color structure preservation |

### 6.2 Specular Removal Losses (New)

#### Specular Sparse Loss
```python
class SpecularSparseLoss(nn.Module):
    """Encourages sparse specular component"""
    def forward(self, specular):
        return torch.mean(torch.abs(specular))
```
- Weight: 0.01
- Promotes sparsity in estimated specular component

#### Polar Consistency Loss
```python
class PolarConsistencyLoss(nn.Module):
    """Ensures consistency between estimated specular and Isv"""
    def forward(self, specular, Isv):
        specular_intensity = specular.mean(dim=1, keepdim=True)
        return F.mse_loss(specular_intensity, Isv)
```
- Weight: 0.1
- Estimated specular should align with measured Isv

### 6.3 Total Loss

```
L_total = L1 + 0.02·L_VGG + 0.0005·L_TV + 0.1·L_phase
        + 0.01·L_specular_sparse + 0.1·L_polar_consistency
```

## 7. Dataset: PSD Adaptation

### 7.1 Dataset Size Comparison

| Item | PolaRGB (Original) | PSD (Specular) |
|------|-------------------|----------------|
| **Task** | Glass reflection removal | Specular highlight removal |
| **Polarization** | 4 angles (0°,45°,90°,135°) | 12 angles (30° intervals) |
| **Camera** | Division-of-focal-plane | Rotating polarizer |
| **Train** | 6,312 images (103 scenes) | 642 groups / 361 pairs |
| **Val** | - | 40 groups / 893 pairs |
| **Test** | 188 images (8 scenes) | 206 groups / 54 pairs |
| **Total Images** | **~6,500** | **~888 groups (~1,308 pairs)** |

**Note**: PSD has significantly fewer polarization groups than PolaRGB. Data augmentation (dataset_enlarge_ratio: 10) is used to compensate.

### 7.2 Angle Mapping

PSD dataset has 12 polarization angles (30° intervals). We map to PolarFree's 4 angles:

| PolarFree | PSD Index | Angle | Method |
|-----------|-----------|-------|--------|
| 0° | idx-01 | 0° | Exact |
| 45° | idx-02, idx-03 | 30°, 60° | Interpolate |
| 90° | idx-04 | 90° | Exact |
| 135° | idx-05, idx-06 | 120°, 150° | Interpolate |

```python
# Interpolation for 45° and 135°
I_45 = (I_30 + I_60) / 2   # (idx-02 + idx-03) / 2
I_135 = (I_120 + I_150) / 2 # (idx-05 + idx-06) / 2
```

### 7.2 Ground Truth

- **Training GT**: IrawD (raw diffuse estimate from polarization)
- **Ideal GT**: PSD diffuse images (when group-to-diffuse mapping available)

### 7.3 Dataset Structure

```
PSD_Dataset/
├── PSD_Train/
│   ├── PSD_Train_diffuse/     # Ground truth diffuse
│   ├── PSD_Train_specular/    # Input with specular
│   └── PSD_Train_group/
│       ├── aligned/           # Aligned polarization groups
│       │   └── group-XXXX-idx-YY.png
│       └── unaligned/
├── PSD_val/
└── PSD_Test/
```

## 8. Training Procedure

### 8.1 Stage 1

```bash
python basicsr/train.py -opt options/train/psd_stage1.yml
```

- Iterations: 100,000
- Batch size: 2
- Learning rate: 1e-4 (cosine annealing)
- Output: Diffuse prior

### 8.2 Stage 2

```bash
python basicsr/train.py -opt options/train/psd_stage2.yml
```

- Iterations: 300,000
- Load Stage 1 weights (frozen)
- Train diffusion model
- Output: Refined diffuse image

## 9. Evaluation Metrics

| Metric | Description | Better |
|--------|-------------|--------|
| PSNR | Peak Signal-to-Noise Ratio | Higher |
| SSIM | Structural Similarity | Higher |
| LPIPS | Learned Perceptual Similarity | Lower |
| SD | Hue histogram standard deviation | Lower |
| CA | Color Accuracy | Higher |

## 10. Key Differences from Original PolarFree

| Aspect | Original PolarFree | Specular Removal |
|--------|-------------------|------------------|
| Task | Glass reflection removal | Specular highlight removal |
| Output | Transmission layer T | Diffuse component Id |
| Physical model | M = αt·T + αr·R | I = Id + Is |
| Key insight | Both layers polarized | Only specular is polarized |
| Input features | 4 polar + AOLP/DOLP + RGB | RGB + IrawD + Ichro + Isv + AOLP/DOLP |
| New losses | - | Specular sparse, Polar consistency |

## 11. File Structure

```
polarfree/
├── archs/
│   ├── Transformer_arch.py      # Main transformer (unchanged)
│   └── latent_encoder_arch.py   # Latent encoder (unchanged)
├── data/
│   ├── paired_image_polar_dataset.py  # Original dataset
│   └── psd_dataset.py                 # PSD dataset loader
├── models/
│   ├── PolarFree_S1_model.py    # Stage 1 model
│   └── PolarFree_S2_model.py    # Stage 2 model
├── utils/
│   ├── losses.py                # Loss functions (extended)
│   └── psd_preprocess.py        # PSD preprocessing
options/
└── train/
    ├── psd_stage1.yml           # Stage 1 config
    └── psd_stage2.yml           # Stage 2 config
docs/
├── PSD_dataset_adaptation.md    # Dataset adaptation details
└── specular_removal_design.md   # This document
```

## 12. Future Improvements

1. **Direct GT matching**: Map PSD groups to diffuse images for better supervision
2. **Multi-scale Isv**: Use Isv at multiple scales for better specular localization
3. **Adaptive loss weighting**: Learn loss weights during training
4. **Cross-polarization features**: Explore using all 12 angles from PSD
5. **Real-world data**: Collect dataset with division-of-focal-plane camera

## References

1. PolarFree: Polarization-based Reflection-Free Imaging (CVPR 2025)
2. Single-Image Specular Highlight Removal via Real-World Dataset Construction (TMM 2021)
3. Polarization Guided Specular Reflection Separation
4. Separation of Reflection Components by Sparse Non-negative Matrix Factorization
