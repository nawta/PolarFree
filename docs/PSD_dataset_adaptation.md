# PSD Dataset Adaptation for Specular Highlight Removal

## Overview

This document describes how to adapt the PSD (Paired Specular-Diffuse) dataset for use with the PolarFree architecture, which was originally designed for glass reflection removal.

## Dataset Comparison

### PolaRGB (Original PolarFree)
- **Purpose**: Glass reflection removal
- **Polarization angles**: 4 angles (0°, 45°, 90°, 135°) captured simultaneously
- **Camera**: Division-of-focal-plane polarization camera
- **Ground truth**: Transmission layer (scene behind glass)

### PSD Dataset (Specular Highlight Removal)
- **Purpose**: Specular highlight removal
- **Polarization angles**: 12 angles (30° intervals: 0°, 30°, 60°, ..., 330°)
- **Capture method**: Rotating polarizer in front of light source
- **Ground truth**: Diffuse component (specular-free image)

## Angle Mapping Strategy

PolarFree expects 4 specific polarization angles. We approximate these from PSD's 12 angles:

| PolarFree Expected | PSD Available | PSD Index | Approximation Error |
|-------------------|---------------|-----------|---------------------|
| 0° | 0° | idx-01 | 0° (exact) |
| 45° | 30° or 60° | idx-02 or idx-03 | ±15° |
| 90° | 90° | idx-04 | 0° (exact) |
| 135° | 120° or 150° | idx-05 or idx-06 | ±15° |

### Recommended Mapping
We use the **closer angles** to minimize error:
- **0°** → idx-01 (0°)
- **45°** → idx-02 (30°) - closer than 60°
- **90°** → idx-04 (90°)
- **135°** → idx-05 (120°) - closer than 150°

### Alternative: Linear Interpolation
For better accuracy, we can interpolate between adjacent angles:
```
I_45° ≈ (I_30° + I_60°) / 2  = (idx-02 + idx-03) / 2
I_135° ≈ (I_120° + I_150°) / 2 = (idx-05 + idx-06) / 2
```

## Physical Model Differences

### Glass Reflection (PolarFree)
```
M = αt·T + αr·R
```
- M: Mixed image
- T: Transmission layer
- R: Reflection layer
- Both T and R have polarization characteristics

### Specular Highlight (This Adaptation)
```
I(φ) = Id + Isc + Isv·cos(2(φ-α))
     = Ic + Isv·cos(2(φ-α))
```
- Id: Diffuse reflection (unpolarized)
- Isc: Specular unpolarized component
- Isv: Specular polarized component (varies with angle)
- **Key insight**: Isv comes 100% from specular reflection

## TRS Processing with Approximated Angles

Using the mapped angles, TRS processing computes:
```python
# Stokes parameters (with approximated angles)
S0 = (I_0 + I_45 + I_90 + I_135) / 2  # Total intensity
S1 = I_0 - I_90                        # Horizontal vs vertical
S2 = I_45 - I_135                      # Diagonal difference

# Derived quantities
Ic = S0                                # Constant component
Isv = sqrt(S1² + S2²) / 2             # Varying amplitude

# Initial estimates
IrawD = Ic - Isv                       # Raw diffuse estimate
IrawS = 2 * Isv                        # Raw specular estimate
```

**Note**: The 15° angle approximation error will introduce some error in S2 calculation, but the overall separation should still be effective.

## PSD Dataset Structure

**Note**: データルートはサーバーにより異なる（oasis: `/data/nishida/`, fleetwood: `/data2/`）

```
{DATA_ROOT}/PSD_Dataset/
├── PSD_Train/
│   ├── PSD_Train_diffuse/          # Ground truth (361 images)
│   │   └── filtered-XXXX.png
│   ├── PSD_Train_specular/         # Input with specular (361 images)
│   │   └── filtered-XXXX.png
│   └── PSD_Train_group/
│       ├── aligned/                 # Aligned polarization groups
│       │   └── group-XXXX-idx-YY.png  (YY: 01-12)
│       └── unaligned/               # Unaligned groups
├── PSD_val/
│   ├── PSD_val_diffuse/            # 893 images
│   ├── PSD_val_specular/           # 893 images
│   └── PSD_val_group/
└── PSD_Test/
    ├── PSD_Test_diffuse/           # 54 images
    ├── PSD_Test_specular/          # 54 images
    └── PSD_Test_group/
```

### Polarization Index Mapping
| Index | Angle (rad) | Angle (deg) |
|-------|-------------|-------------|
| idx-01 | 0 | 0° |
| idx-02 | π/6 | 30° |
| idx-03 | π/3 | 60° |
| idx-04 | π/2 | 90° |
| idx-05 | 2π/3 | 120° |
| idx-06 | 5π/6 | 150° |
| idx-07 | π | 180° |
| idx-08 | 7π/6 | 210° |
| idx-09 | 4π/3 | 240° |
| idx-10 | 3π/2 | 270° |
| idx-11 | 5π/3 | 300° |
| idx-12 | 11π/6 | 330° |

## Implementation Plan

### 1. Data Preprocessing Script
Create `polarfree/utils/psd_preprocess.py`:
- Extract 4 angles from 12-angle groups
- Option for direct mapping or interpolation
- Generate AOLP/DOLP from extracted angles
- Organize into PolarFree-compatible format

### 2. Dataset Loader
Create `polarfree/data/psd_dataset.py`:
- Load specular/diffuse pairs
- Load corresponding polarization images
- Compute TRS features (Ic, Isv, IrawD, Ichro)
- Handle both aligned and unaligned data

### 3. Training Configuration
Update config files:
- Point to PSD dataset paths
- Adjust input channels if needed
- Configure loss weights for specular removal

## Expected Challenges

1. **Angle approximation error**: 15° offset may reduce TRS accuracy
2. **Different physical model**: Specular vs glass reflection behaves differently
3. **Dataset size**: PSD has fewer polarization groups than PolaRGB
4. **Alignment**: Some PSD groups are unaligned

## Evaluation Metrics

Following the SpecularityNet paper:
- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **SD**: Hue histogram standard deviation (lower is better)
- **CA**: Color Accuracy

## References

1. PolarFree: Polarization-based Reflection-Free Imaging (CVPR 2025)
2. Single-Image Specular Highlight Removal via Real-World Dataset Construction (TMM 2021)
3. Polarization Guided Specular Reflection Separation
