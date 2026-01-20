"""
PSD Dataset Loader for Specular Highlight Removal

Converts 12-angle polarization data to 4-angle format compatible with PolarFree.
Follows the original PolarFree dataset output format.
"""

import os
import glob
import cv2
import numpy as np
import torch
from torch.utils import data
from basicsr.utils import FileClient
from basicsr.utils.registry import DATASET_REGISTRY
from polarfree.utils.transforms import augment


@DATASET_REGISTRY.register()
class PSDDataset(data.Dataset):
    """
    PSD Dataset with polarization groups for PolarFree.

    Loads 12-angle polarization data and converts to 4 angles (0°, 45°, 90°, 135°).
    Output format matches the original PairedImagePolarDataset.

    Args:
        opt (dict): Configuration options including:
            dataroot_psd (str): Root path to PSD_Dataset
            split (str): 'Train', 'val', or 'Test'
            use_aligned (bool): Use aligned or unaligned groups
            interpolate (bool): Interpolate for 45° and 135°
            gt_size (int): Crop size for training
            phase (str): 'train' or 'val'/'test'
    """

    # Mapping from PSD 12 angles to PolarFree 4 angles
    # PSD: 30° intervals (idx-01=0°, idx-02=30°, idx-03=60°, idx-04=90°, ...)
    # PolarFree: 0°, 45°, 90°, 135°
    ANGLE_MAP_INTERPOLATE = {
        '000': [1],        # idx-01 (0°) - exact
        '045': [2, 3],     # interpolate idx-02 (30°) and idx-03 (60°)
        '090': [4],        # idx-04 (90°) - exact
        '135': [5, 6],     # interpolate idx-05 (120°) and idx-06 (150°)
    }
    ANGLE_MAP_NEAREST = {
        '000': [1],        # idx-01 (0°)
        '045': [2],        # idx-02 (30°) - nearest to 45°
        '090': [4],        # idx-04 (90°)
        '135': [5],        # idx-05 (120°) - nearest to 135°
    }

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.psd_root = opt['dataroot_psd']
        self.split = opt.get('split', 'Train')
        self.use_aligned = opt.get('use_aligned', True)
        self.interpolate = opt.get('interpolate', True)

        self._build_paths()

    def _build_paths(self):
        """Build list of available groups with GT matching"""
        group_type = 'aligned' if self.use_aligned else 'unaligned'

        # PSD Train has aligned/unaligned subdirectories, but val/Test don't
        group_path_with_subdir = os.path.join(
            self.psd_root, f'PSD_{self.split}',
            f'PSD_{self.split}_group', group_type
        )
        group_path_direct = os.path.join(
            self.psd_root, f'PSD_{self.split}',
            f'PSD_{self.split}_group'
        )

        # Check which path exists and has data
        if os.path.exists(group_path_with_subdir) and len(glob.glob(os.path.join(group_path_with_subdir, 'group-*-idx-01.png'))) > 0:
            self.group_path = group_path_with_subdir
            path_type = group_type
        else:
            self.group_path = group_path_direct
            path_type = 'direct'

        self.diffuse_path = os.path.join(
            self.psd_root, f'PSD_{self.split}', f'PSD_{self.split}_diffuse'
        )

        # Find all groups - handle different naming conventions
        # Train/val use 'group-XXXX-idx-YY.png', Test uses 'test-XXXX-idx-YY.png'
        group_files = glob.glob(os.path.join(self.group_path, 'group-*-idx-01.png'))
        if len(group_files) == 0:
            # Try test- prefix for Test split
            group_files = glob.glob(os.path.join(self.group_path, 'test-*-idx-01.png'))
            self.file_prefix = 'test'
        else:
            self.file_prefix = 'group'

        self.group_ids = sorted(set([
            os.path.basename(f).split('-')[1] for f in group_files
        ]))

        # For training, we need GT. Use IrawD as pseudo-GT since direct matching is complex
        print(f"PSD Dataset: Found {len(self.group_ids)} groups in {self.split} ({path_type})")

    def _load_group_images(self, group_id: str) -> dict:
        """Load all 12 polarization images for a group"""
        images = {}
        for idx in range(1, 13):
            filename = f"{self.file_prefix}-{group_id}-idx-{idx:02d}.png"
            filepath = os.path.join(self.group_path, filename)
            if os.path.exists(filepath):
                img = cv2.imread(filepath)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
                    images[idx] = img
        return images

    def _extract_4_angles(self, images_12: dict) -> dict:
        """Extract 4 polarization angles from 12-angle data"""
        angle_map = self.ANGLE_MAP_INTERPOLATE if self.interpolate else self.ANGLE_MAP_NEAREST
        images_4 = {}

        for angle_name, indices in angle_map.items():
            if len(indices) == 1:
                idx = indices[0]
                if idx in images_12:
                    images_4[angle_name] = images_12[idx]
            else:
                idx1, idx2 = indices
                if idx1 in images_12 and idx2 in images_12:
                    # Linear interpolation
                    images_4[angle_name] = (images_12[idx1] + images_12[idx2]) / 2

        return images_4

    def _compute_polarization_features(self, img0, img45, img90, img135):
        """
        Compute Stokes parameters and derived features.
        Matches the original PolarFree dataset processing.
        """
        eps = 1e-8

        # Convert to grayscale for Stokes computation
        def to_gray(img):
            return np.mean(img, axis=-1)

        I0 = to_gray(img0)
        I45 = to_gray(img45)
        I90 = to_gray(img90)
        I135 = to_gray(img135)

        # Stokes parameters
        S0 = (I0 + I45 + I90 + I135) / 2 + eps
        S1 = I0 - I90
        S2 = I45 - I135

        # Avoid division by zero
        S1 = np.where(S1 == 0, 0.0001, S1)

        # DOLP and AOLP
        dolp = np.sqrt(S1**2 + S2**2) / S0
        dolp = np.clip(dolp, 0, 1)
        aolp = 0.5 * np.arctan2(S2, S1)

        # RGB image from S0 (unpolarized)
        rgb = (img0 + img45 + img90 + img135) / 2

        # Ip and Inp (polarized and non-polarized components)
        Ip = aolp[:, :, np.newaxis] * rgb
        Inp = rgb - Ip

        return {
            'aolp': aolp[np.newaxis, :, :],  # [1, H, W]
            'dolp': dolp[np.newaxis, :, :],  # [1, H, W]
            'rgb': rgb,                       # [H, W, 3]
            'Ip': Ip,                         # [H, W, 3]
            'Inp': Inp,                       # [H, W, 3]
            'I0': I0[np.newaxis, :, :],
            'I45': I45[np.newaxis, :, :],
            'I90': I90[np.newaxis, :, :],
            'I135': I135[np.newaxis, :, :],
        }

    def _compute_trs_features(self, img0_rgb, img45_rgb, img90_rgb, img135_rgb):
        """
        Compute TRS (Transmitted Radiance Sinusoid) features for specular removal.
        """
        eps = 1e-6

        # Per-channel TRS
        Ic = (img0_rgb + img45_rgb + img90_rgb + img135_rgb) / 2
        Q = img0_rgb - img90_rgb
        U = img45_rgb - img135_rgb
        Isv = np.sqrt(Q**2 + U**2) / 2

        # Raw diffuse estimate
        IrawD = np.maximum(Ic - Isv, 0)

        # Polarization chromaticity
        IrawD_sum = np.sum(IrawD, axis=-1, keepdims=True)
        Ichro = IrawD / (IrawD_sum + eps)

        # Grayscale Isv
        Isv_gray = np.mean(Isv, axis=-1)

        return {
            'IrawD': IrawD,                          # [H, W, 3]
            'Ichro': Ichro,                          # [H, W, 3]
            'Isv_gray': Isv_gray[np.newaxis, :, :],  # [1, H, W]
        }

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt
            )

        index = index % len(self.group_ids)
        group_id = self.group_ids[index]

        # Load 12 polarization images
        images_12 = self._load_group_images(group_id)

        # Need at least indices 1-6 for our mapping
        if len(images_12) < 6:
            return self.__getitem__((index + 1) % len(self.group_ids))

        # Extract 4 angles
        images_4 = self._extract_4_angles(images_12)
        if len(images_4) != 4:
            return self.__getitem__((index + 1) % len(self.group_ids))

        img0_rgb = images_4['000']
        img45_rgb = images_4['045']
        img90_rgb = images_4['090']
        img135_rgb = images_4['135']

        # Compute polarization features
        polar_feat = self._compute_polarization_features(
            img0_rgb, img45_rgb, img90_rgb, img135_rgb
        )

        # Compute TRS features
        trs_feat = self._compute_trs_features(
            img0_rgb, img45_rgb, img90_rgb, img135_rgb
        )

        # Use IrawD as pseudo ground truth (diffuse estimate)
        # In real training, this should be the actual diffuse GT
        gt_rgb = trs_feat['IrawD']
        lq_rgb = polar_feat['rgb']

        # Training augmentation
        if self.opt['phase'] == 'train':
            GT_size = self.opt.get('gt_size', 256)
            h, w = lq_rgb.shape[:2]

            if h >= GT_size and w >= GT_size:
                rh = np.random.randint(0, h - GT_size + 1)
                rw = np.random.randint(0, w - GT_size + 1)

                # Crop all images
                img0_rgb = img0_rgb[rh:rh+GT_size, rw:rw+GT_size]
                img45_rgb = img45_rgb[rh:rh+GT_size, rw:rw+GT_size]
                img90_rgb = img90_rgb[rh:rh+GT_size, rw:rw+GT_size]
                img135_rgb = img135_rgb[rh:rh+GT_size, rw:rw+GT_size]
                lq_rgb = lq_rgb[rh:rh+GT_size, rw:rw+GT_size]
                gt_rgb = gt_rgb[rh:rh+GT_size, rw:rw+GT_size]

                # Crop features
                for k in ['aolp', 'dolp', 'I0', 'I45', 'I90', 'I135']:
                    polar_feat[k] = polar_feat[k][:, rh:rh+GT_size, rw:rw+GT_size]
                polar_feat['Ip'] = polar_feat['Ip'][rh:rh+GT_size, rw:rw+GT_size]
                polar_feat['Inp'] = polar_feat['Inp'][rh:rh+GT_size, rw:rw+GT_size]

                for k in ['IrawD', 'Ichro']:
                    trs_feat[k] = trs_feat[k][rh:rh+GT_size, rw:rw+GT_size]
                trs_feat['Isv_gray'] = trs_feat['Isv_gray'][:, rh:rh+GT_size, rw:rw+GT_size]

        # Convert to tensors (CHW format)
        def to_tensor(x):
            if x.ndim == 3 and x.shape[-1] == 3:  # HWC -> CHW
                x = np.transpose(x, (2, 0, 1))
            return torch.from_numpy(np.ascontiguousarray(x.copy())).float()

        # Match original PolarFree output format
        result = {
            # Original polarization features (grayscale)
            'lq_img0': to_tensor(polar_feat['I0']),
            'lq_img45': to_tensor(polar_feat['I45']),
            'lq_img90': to_tensor(polar_feat['I90']),
            'lq_img135': to_tensor(polar_feat['I135']),
            'lq_rgb': to_tensor(lq_rgb),
            'lq_aolp': to_tensor(polar_feat['aolp']),
            'lq_dolp': to_tensor(polar_feat['dolp']),
            'lq_Ip': to_tensor(polar_feat['Ip']),
            'lq_Inp': to_tensor(polar_feat['Inp']),
            # TRS-based features
            'lq_IrawD': to_tensor(trs_feat['IrawD']),
            'lq_Ichro': to_tensor(trs_feat['Ichro']),
            'lq_Isv': to_tensor(trs_feat['Isv_gray']),
            # RGB polarization images
            'lq_img0_rgb': to_tensor(img0_rgb),
            'lq_img45_rgb': to_tensor(img45_rgb),
            'lq_img90_rgb': to_tensor(img90_rgb),
            'lq_img135_rgb': to_tensor(img135_rgb),
            # Ground truth (using IrawD as pseudo-GT)
            'gt_img0': to_tensor(gt_rgb),
            'gt_img45': to_tensor(gt_rgb),
            'gt_img90': to_tensor(gt_rgb),
            'gt_img135': to_tensor(gt_rgb),
            'gt_rgb': to_tensor(gt_rgb),
            'lq_path': f'group-{group_id}'
        }

        return result

    def __len__(self):
        return len(self.group_ids)
