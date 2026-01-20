"""
PSD Dataset Preprocessing for PolarFree

Converts 12-angle polarization data from PSD dataset to 4-angle format
compatible with PolarFree's TRS processing.

Angle Mapping:
    PSD idx-01 (0°)   -> PolarFree 0°
    PSD idx-02 (30°)  -> PolarFree 45° (approximation, or interpolate with idx-03)
    PSD idx-04 (90°)  -> PolarFree 90°
    PSD idx-05 (120°) -> PolarFree 135° (approximation, or interpolate with idx-06)
"""

import os
import cv2
import glob
import argparse
import numpy as np
from tqdm import tqdm


def get_angle_indices(interpolate: bool = True):
    """
    Get the PSD indices to use for each PolarFree angle.

    Args:
        interpolate: If True, return pairs for interpolation.
                     If False, return single closest index.

    Returns:
        Dictionary mapping PolarFree angles to PSD indices
    """
    if interpolate:
        return {
            '000': [1],        # idx-01 (0°) - exact
            '045': [2, 3],     # interpolate idx-02 (30°) and idx-03 (60°)
            '090': [4],        # idx-04 (90°) - exact
            '135': [5, 6],     # interpolate idx-05 (120°) and idx-06 (150°)
        }
    else:
        return {
            '000': [1],        # idx-01 (0°)
            '045': [2],        # idx-02 (30°) - closest to 45°
            '090': [4],        # idx-04 (90°)
            '135': [5],        # idx-05 (120°) - closest to 135°
        }


def load_psd_group(group_path: str, group_id: str) -> dict:
    """
    Load all 12 polarization images for a group.

    Args:
        group_path: Path to aligned or unaligned folder
        group_id: Group identifier (e.g., '0001')

    Returns:
        Dictionary with idx (1-12) as keys and images as values
    """
    images = {}
    for idx in range(1, 13):
        filename = f"group-{group_id}-idx-{idx:02d}.png"
        filepath = os.path.join(group_path, filename)
        if os.path.exists(filepath):
            img = cv2.imread(filepath)
            if img is not None:
                images[idx] = img
    return images


def extract_4_angles(images_12: dict, interpolate: bool = True) -> dict:
    """
    Extract 4 polarization angles from 12-angle data.

    Args:
        images_12: Dictionary with 12 polarization images
        interpolate: Whether to interpolate for 45° and 135°

    Returns:
        Dictionary with 4 polarization images (000, 045, 090, 135)
    """
    angle_map = get_angle_indices(interpolate)
    images_4 = {}

    for angle_name, indices in angle_map.items():
        if len(indices) == 1:
            # Direct mapping
            idx = indices[0]
            if idx in images_12:
                images_4[angle_name] = images_12[idx]
        else:
            # Interpolation
            idx1, idx2 = indices
            if idx1 in images_12 and idx2 in images_12:
                img1 = images_12[idx1].astype(np.float32)
                img2 = images_12[idx2].astype(np.float32)
                interpolated = ((img1 + img2) / 2).astype(np.uint8)
                images_4[angle_name] = interpolated

    return images_4


def compute_rgb_from_polarization(images_4: dict) -> np.ndarray:
    """
    Compute RGB image from 4 polarization angles.

    Using Stokes parameter S0 = (I0 + I45 + I90 + I135) / 2

    Args:
        images_4: Dictionary with 4 polarization images

    Returns:
        RGB image (unpolarized)
    """
    if all(k in images_4 for k in ['000', '045', '090', '135']):
        I0 = images_4['000'].astype(np.float32)
        I45 = images_4['045'].astype(np.float32)
        I90 = images_4['090'].astype(np.float32)
        I135 = images_4['135'].astype(np.float32)

        # S0 = (I0 + I45 + I90 + I135) / 2
        rgb = ((I0 + I45 + I90 + I135) / 2).astype(np.uint8)
        return rgb
    return None


def process_psd_dataset(
    psd_root: str,
    output_root: str,
    split: str = 'Train',
    use_aligned: bool = True,
    interpolate: bool = True
):
    """
    Process PSD dataset and convert to PolarFree format.

    Args:
        psd_root: Root path to PSD_Dataset
        output_root: Output path for converted data
        split: 'Train', 'val', or 'Test'
        use_aligned: Use aligned or unaligned groups
        interpolate: Interpolate for 45° and 135°
    """
    # Input paths
    group_type = 'aligned' if use_aligned else 'unaligned'
    group_path = os.path.join(psd_root, f'PSD_{split}', f'PSD_{split}_group', group_type)
    specular_path = os.path.join(psd_root, f'PSD_{split}', f'PSD_{split}_specular')
    diffuse_path = os.path.join(psd_root, f'PSD_{split}', f'PSD_{split}_diffuse')

    # Check paths exist
    if not os.path.exists(group_path):
        print(f"Warning: Group path does not exist: {group_path}")
        return

    # Get all group IDs
    group_files = glob.glob(os.path.join(group_path, 'group-*-idx-01.png'))
    group_ids = sorted(set([
        os.path.basename(f).split('-')[1]
        for f in group_files
    ]))

    print(f"Found {len(group_ids)} groups in {split} ({group_type})")

    # Output directories
    output_split = 'train' if split == 'Train' else split.lower()
    output_input = os.path.join(output_root, output_split, 'input')
    output_gt = os.path.join(output_root, output_split, 'gt')

    os.makedirs(output_input, exist_ok=True)
    os.makedirs(output_gt, exist_ok=True)

    # Process each group
    for group_id in tqdm(group_ids, desc=f'Processing {split}'):
        # Load 12 polarization images
        images_12 = load_psd_group(group_path, group_id)

        if len(images_12) < 6:  # Need at least idx 1-6 for our mapping
            continue

        # Extract 4 angles
        images_4 = extract_4_angles(images_12, interpolate)

        if len(images_4) != 4:
            continue

        # Compute RGB
        rgb = compute_rgb_from_polarization(images_4)

        # Create scene directory
        scene_dir = os.path.join(output_input, f'scene_{group_id}')
        os.makedirs(scene_dir, exist_ok=True)

        # Save 4 polarization images
        for angle_name, img in images_4.items():
            output_file = os.path.join(scene_dir, f'000_{angle_name}.png')
            cv2.imwrite(output_file, img)

        # Save RGB
        if rgb is not None:
            cv2.imwrite(os.path.join(scene_dir, '000_rgb.png'), rgb)

        # Try to find and copy corresponding diffuse (GT)
        # PSD diffuse files don't have direct group mapping, so we skip for now
        # This needs manual matching based on the dataset structure

    print(f"Processed {len(group_ids)} groups to {output_root}")


def process_specular_diffuse_pairs(
    psd_root: str,
    output_root: str,
    split: str = 'Train'
):
    """
    Process specular-diffuse pairs (without polarization groups).

    This creates a simpler dataset with just input/GT pairs.

    Args:
        psd_root: Root path to PSD_Dataset
        output_root: Output path for converted data
        split: 'Train', 'val', or 'Test'
    """
    specular_path = os.path.join(psd_root, f'PSD_{split}', f'PSD_{split}_specular')
    diffuse_path = os.path.join(psd_root, f'PSD_{split}', f'PSD_{split}_diffuse')

    if not os.path.exists(specular_path) or not os.path.exists(diffuse_path):
        print(f"Warning: Paths do not exist for {split}")
        return

    # Get all specular images
    specular_files = sorted(glob.glob(os.path.join(specular_path, '*.png')))

    print(f"Found {len(specular_files)} specular images in {split}")

    # Output directories
    output_split = 'train' if split == 'Train' else split.lower()
    output_input = os.path.join(output_root, output_split, 'input')
    output_gt = os.path.join(output_root, output_split, 'gt')

    os.makedirs(output_input, exist_ok=True)
    os.makedirs(output_gt, exist_ok=True)

    for spec_file in tqdm(specular_files, desc=f'Processing {split} pairs'):
        basename = os.path.basename(spec_file)
        diff_file = os.path.join(diffuse_path, basename)

        if not os.path.exists(diff_file):
            continue

        # Extract image ID
        img_id = basename.replace('.png', '').replace('filtered-', '')

        # Create scene directory
        scene_dir = os.path.join(output_input, f'scene_{img_id}')
        gt_scene_dir = os.path.join(output_gt, f'scene_{img_id}')
        os.makedirs(scene_dir, exist_ok=True)
        os.makedirs(gt_scene_dir, exist_ok=True)

        # Copy specular as input RGB
        spec_img = cv2.imread(spec_file)
        cv2.imwrite(os.path.join(scene_dir, '000_rgb.png'), spec_img)

        # Copy diffuse as GT
        diff_img = cv2.imread(diff_file)
        cv2.imwrite(os.path.join(gt_scene_dir, '000_rgb.png'), diff_img)

    print(f"Processed pairs to {output_root}")


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess PSD dataset for PolarFree'
    )
    parser.add_argument(
        '--psd_root', type=str,
        default='/data2/PSD_Dataset/PSD_Dataset',
        help='Root path to PSD_Dataset'
    )
    parser.add_argument(
        '--output_root', type=str,
        default='/data2/PSD_PolarFree',
        help='Output path for converted dataset'
    )
    parser.add_argument(
        '--mode', type=str, choices=['polarization', 'pairs', 'both'],
        default='both',
        help='Processing mode: polarization groups, specular-diffuse pairs, or both'
    )
    parser.add_argument(
        '--interpolate', action='store_true', default=True,
        help='Interpolate 45° and 135° from adjacent angles'
    )
    parser.add_argument(
        '--use_aligned', action='store_true', default=True,
        help='Use aligned polarization groups'
    )

    args = parser.parse_args()

    print(f"PSD Root: {args.psd_root}")
    print(f"Output Root: {args.output_root}")
    print(f"Mode: {args.mode}")
    print(f"Interpolate: {args.interpolate}")

    os.makedirs(args.output_root, exist_ok=True)

    for split in ['Train', 'val', 'Test']:
        if args.mode in ['polarization', 'both']:
            process_psd_dataset(
                args.psd_root,
                os.path.join(args.output_root, 'polarization'),
                split=split,
                use_aligned=args.use_aligned,
                interpolate=args.interpolate
            )

        if args.mode in ['pairs', 'both']:
            process_specular_diffuse_pairs(
                args.psd_root,
                os.path.join(args.output_root, 'pairs'),
                split=split
            )


if __name__ == '__main__':
    main()
