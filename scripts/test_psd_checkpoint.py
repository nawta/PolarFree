"""
Test PSD checkpoint at any iteration.
Usage:
    python scripts/test_psd_checkpoint.py --iter 50000
    python scripts/test_psd_checkpoint.py --compare_iters 5000,25000,50000,100000
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import cv2
import numpy as np
from tqdm import tqdm

from polarfree.data.psd_dataset import PSDDataset
from polarfree.archs.Transformer_arch import Transformer
from polarfree.archs.latent_encoder_arch import latent_encoder_gelu
from basicsr.utils.img_util import tensor2img
from basicsr.metrics import calculate_psnr, calculate_ssim


def load_stage1_model(checkpoint_dir, iter_num, device):
    """Load Stage 1 model from checkpoint"""
    net_g = Transformer(
        inp_channels=3, out_channels=3, dim=48,
        num_blocks=[3, 4, 4, 4], num_refinement_blocks=4,
        heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
        bias=False, LayerNorm_type='WithBias',
        dual_pixel_task=False, embed_dim=64, group=4
    ).to(device)

    net_le = latent_encoder_gelu(
        in_chans=12, embed_dim=64, block_num=6, group=4,
        stage=1, patch_expansion=0.5, channel_expansion=4
    ).to(device)

    if iter_num == 'latest':
        g_path = os.path.join(checkpoint_dir, 'net_g_latest.pth')
        le_path = os.path.join(checkpoint_dir, 'net_le_latest.pth')
    else:
        g_path = os.path.join(checkpoint_dir, f'net_g_{iter_num}.pth')
        le_path = os.path.join(checkpoint_dir, f'net_le_{iter_num}.pth')

    if not os.path.exists(g_path):
        return None, None

    net_g.load_state_dict(torch.load(g_path)['params'])
    net_le.load_state_dict(torch.load(le_path)['params'])
    net_g.eval()
    net_le.eval()

    return net_g, net_le


def prepare_input(data, device):
    """Prepare input for latent encoder"""
    lq_rgb = data['lq_rgb'].to(device)
    lq_img0 = data['lq_img0'].to(device)
    lq_img45 = data['lq_img45'].to(device)
    lq_img90 = data['lq_img90'].to(device)
    lq_img135 = data['lq_img135'].to(device)
    lq_aolp = data['lq_aolp'].to(device)
    lq_dolp = data['lq_dolp'].to(device)
    gt_rgb = data['gt_rgb'].to(device)

    input_features = [lq_rgb, lq_img0, lq_img45, lq_img90, lq_img135, lq_aolp, lq_dolp]
    return input_features, lq_rgb, gt_rgb


def test_single_checkpoint(checkpoint_dir, iter_num, dataset, device, num_samples, save_dir=None):
    """Test a single checkpoint and return metrics"""
    net_g, net_le = load_stage1_model(checkpoint_dir, iter_num, device)
    if net_g is None:
        print(f"  Checkpoint {iter_num} not found, skipping...")
        return None, None, []

    psnr_list, ssim_list = [], []
    output_images = []

    with torch.no_grad():
        for idx in range(min(len(dataset), num_samples)):
            data = dataset[idx]
            for key in data:
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].unsqueeze(0)

            input_features, lq_rgb, gt_rgb = prepare_input(data, device)
            latent = net_le(input_features, gt_rgb)
            output = net_g(lq_rgb, latent)

            output_img = tensor2img([output.squeeze(0)])
            gt_img = tensor2img([data['gt_rgb'].squeeze(0)])
            lq_img = tensor2img([lq_rgb.squeeze(0)])

            psnr = calculate_psnr(output_img, gt_img, crop_border=0, test_y_channel=False)
            ssim = calculate_ssim(output_img, gt_img, crop_border=0, test_y_channel=False)
            psnr_list.append(psnr)
            ssim_list.append(ssim)

            if save_dir and idx < 5:  # Save first 5 samples
                output_images.append((idx, lq_img, output_img, gt_img))

    # Save images if requested
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for idx, lq_img, output_img, gt_img in output_images:
            cv2.imwrite(os.path.join(save_dir, f'{idx:04d}_output.png'),
                       cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(save_dir, f'{idx:04d}_input.png'),
                       cv2.cvtColor(lq_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(save_dir, f'{idx:04d}_gt.png'),
                       cv2.cvtColor(gt_img, cv2.COLOR_RGB2BGR))

    return np.mean(psnr_list), np.mean(ssim_list), output_images


def compare_checkpoints(args):
    """Compare multiple checkpoints"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_dir = f'experiments/psd_stage{args.stage}/models'

    # Parse iterations
    iters = [x.strip() for x in args.compare_iters.split(',')]

    # Create dataset
    dataset_opt = {
        'dataroot_psd': '/data2/PSD_Dataset/PSD_Dataset',
        'split': 'Test', 'use_aligned': True, 'interpolate': True,
        'io_backend': {'type': 'disk'}, 'phase': 'val',
    }
    dataset = PSDDataset(dataset_opt)

    print(f"\n{'='*60}")
    print(f"Comparing Stage {args.stage} checkpoints on {len(dataset)} test samples")
    print(f"{'='*60}\n")

    results = []
    for iter_num in iters:
        print(f"Testing iter {iter_num}...")
        save_dir = f'results/psd_stage{args.stage}_iter{iter_num}' if args.save_images else None
        psnr, ssim, _ = test_single_checkpoint(
            checkpoint_dir, iter_num, dataset, device, args.num_samples, save_dir
        )
        if psnr is not None:
            results.append((iter_num, psnr, ssim))
            print(f"  PSNR: {psnr:.4f} dB, SSIM: {ssim:.4f}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"{'Iteration':<15} {'PSNR (dB)':<12} {'SSIM':<10}")
    print(f"{'-'*60}")
    for iter_num, psnr, ssim in results:
        print(f"{iter_num:<15} {psnr:<12.4f} {ssim:<10.4f}")
    print(f"{'='*60}\n")

    # Create comparison image if saving
    if args.save_images and len(results) > 1:
        create_comparison_grid(args.stage, iters, args.num_samples)


def create_comparison_grid(stage, iters, num_samples):
    """Create a grid comparing outputs across iterations"""
    output_dir = f'results/psd_stage{stage}_comparison'
    os.makedirs(output_dir, exist_ok=True)

    for sample_idx in range(min(5, num_samples)):
        images = []
        labels = ['Input', 'GT']

        # Load input and GT from first checkpoint dir
        first_dir = f'results/psd_stage{stage}_iter{iters[0]}'
        if os.path.exists(os.path.join(first_dir, f'{sample_idx:04d}_input.png')):
            input_img = cv2.imread(os.path.join(first_dir, f'{sample_idx:04d}_input.png'))
            gt_img = cv2.imread(os.path.join(first_dir, f'{sample_idx:04d}_gt.png'))
            images.extend([input_img, gt_img])

            # Add outputs from each iteration
            for iter_num in iters:
                iter_dir = f'results/psd_stage{stage}_iter{iter_num}'
                output_path = os.path.join(iter_dir, f'{sample_idx:04d}_output.png')
                if os.path.exists(output_path):
                    images.append(cv2.imread(output_path))
                    labels.append(f'iter {iter_num}')

            if len(images) > 2:
                # Create grid
                h, w = images[0].shape[:2]
                n_cols = len(images)
                grid = np.zeros((h + 30, w * n_cols, 3), dtype=np.uint8)

                for i, (img, label) in enumerate(zip(images, labels)):
                    grid[30:h+30, i*w:(i+1)*w] = img
                    cv2.putText(grid, label, (i*w + 10, 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imwrite(os.path.join(output_dir, f'comparison_{sample_idx:04d}.png'), grid)

    print(f"Comparison images saved to: {output_dir}")


def test_checkpoint(args):
    """Test a single checkpoint"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_dir = f'experiments/psd_stage{args.stage}/models'

    print(f"Loading checkpoint from {checkpoint_dir}, iter={args.iter}")

    dataset_opt = {
        'dataroot_psd': '/data2/PSD_Dataset/PSD_Dataset',
        'split': 'Test', 'use_aligned': True, 'interpolate': True,
        'io_backend': {'type': 'disk'}, 'phase': 'val',
    }
    dataset = PSDDataset(dataset_opt)

    save_dir = f'results/psd_stage{args.stage}_iter{args.iter}' if args.save_images else None
    psnr, ssim, _ = test_single_checkpoint(
        checkpoint_dir, args.iter, dataset, device, args.num_samples, save_dir
    )

    if psnr is not None:
        print(f"\n=== Results (Stage {args.stage}, iter {args.iter}) ===")
        print(f"PSNR: {psnr:.4f} dB")
        print(f"SSIM: {ssim:.4f}")
        if save_dir:
            print(f"Output saved to: {save_dir}")

    return psnr, ssim


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iter', type=str, default='latest', help='Iteration number or "latest"')
    parser.add_argument('--compare_iters', type=str, help='Compare multiple iters (comma-separated)')
    parser.add_argument('--stage', type=int, default=1, help='Stage 1 or 2')
    parser.add_argument('--num_samples', type=int, default=50, help='Number of samples to test')
    parser.add_argument('--save_images', action='store_true', help='Save output images')
    args = parser.parse_args()

    if args.compare_iters:
        compare_checkpoints(args)
    else:
        test_checkpoint(args)
