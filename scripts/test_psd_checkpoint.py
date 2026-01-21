"""
Test PSD checkpoint at any iteration.
Usage:
    python scripts/test_psd_checkpoint.py --iter 50000
    python scripts/test_psd_checkpoint.py --stage 2 --iter 50000
    python scripts/test_psd_checkpoint.py --compare_iters 5000,25000,50000,100000
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from tqdm import tqdm

from polarfree.data.psd_dataset import PSDDataset
from polarfree.archs.Transformer_arch import Transformer
from polarfree.archs.latent_encoder_arch import latent_encoder_gelu
from polarfree.archs.denoising_arch import denoising
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


def load_stage2_model(checkpoint_dir, iter_num, device):
    """Load Stage 2 model from checkpoint"""
    # Generator (same architecture as Stage 1)
    net_g = Transformer(
        inp_channels=3, out_channels=3, dim=48,
        num_blocks=[3, 4, 4, 4], num_refinement_blocks=4,
        heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
        bias=False, LayerNorm_type='WithBias',
        dual_pixel_task=False, embed_dim=64, group=4
    ).to(device)

    # Latent encoder for diffusion (Stage 2)
    net_le_dm = latent_encoder_gelu(
        in_chans=9, embed_dim=64, block_num=6, group=4,
        stage=2, patch_expansion=0.5, channel_expansion=4
    ).to(device)

    # Denoising network
    net_d = denoising(
        in_channel=256, out_channel=256, inner_channel=512,
        block_num=4, group=4, patch_expansion=0.5, channel_expansion=2
    ).to(device)

    if iter_num == 'latest':
        g_path = os.path.join(checkpoint_dir, 'net_g_latest.pth')
        le_dm_path = os.path.join(checkpoint_dir, 'net_le_dm_latest.pth')
        d_path = os.path.join(checkpoint_dir, 'net_d_latest.pth')
    else:
        g_path = os.path.join(checkpoint_dir, f'net_g_{iter_num}.pth')
        le_dm_path = os.path.join(checkpoint_dir, f'net_le_dm_{iter_num}.pth')
        d_path = os.path.join(checkpoint_dir, f'net_d_{iter_num}.pth')

    if not os.path.exists(g_path):
        return None, None, None

    net_g.load_state_dict(torch.load(g_path)['params'])
    net_le_dm.load_state_dict(torch.load(le_dm_path)['params'])
    net_d.load_state_dict(torch.load(d_path)['params'])
    net_g.eval()
    net_le_dm.eval()
    net_d.eval()

    return net_g, net_le_dm, net_d


def setup_diffusion_params(device, timesteps=8, linear_start=0.1, linear_end=0.99):
    """Setup diffusion schedule parameters"""
    betas = np.linspace(linear_start, linear_end, timesteps, dtype=np.float64)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

    sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)
    sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod)
    sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod - 1)

    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    posterior_log_variance_clipped = np.log(np.maximum(posterior_variance, 1e-20))
    posterior_mean_coef1 = betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)

    return {
        'betas': torch.tensor(betas, dtype=torch.float32, device=device),
        'sqrt_recip_alphas_cumprod': torch.tensor(sqrt_recip_alphas_cumprod, dtype=torch.float32, device=device),
        'sqrt_recipm1_alphas_cumprod': torch.tensor(sqrt_recipm1_alphas_cumprod, dtype=torch.float32, device=device),
        'posterior_log_variance_clipped': torch.tensor(posterior_log_variance_clipped, dtype=torch.float32, device=device),
        'posterior_mean_coef1': torch.tensor(posterior_mean_coef1, dtype=torch.float32, device=device),
        'posterior_mean_coef2': torch.tensor(posterior_mean_coef2, dtype=torch.float32, device=device),
        'timesteps': timesteps,
    }


def p_sample(net_d, x, condition_x, t, diffusion_params):
    """Single step of reverse diffusion"""
    # t_tensor should match x shape for denoising network
    t_tensor = torch.full(x.shape, t + 1, device=x.device, dtype=torch.long)
    noise = net_d(x, condition_x, t_tensor)

    sqrt_recip = diffusion_params['sqrt_recip_alphas_cumprod'][t]
    sqrt_recipm1 = diffusion_params['sqrt_recipm1_alphas_cumprod'][t]
    x_recon = sqrt_recip * x - sqrt_recipm1 * noise
    x_recon = x_recon.clamp(-1., 1.)

    coef1 = diffusion_params['posterior_mean_coef1'][t]
    coef2 = diffusion_params['posterior_mean_coef2'][t]
    model_mean = coef1 * x_recon + coef2 * x

    return model_mean


def p_sample_loop(net_d, condition_x, x_noisy, diffusion_params):
    """Full reverse diffusion loop"""
    x = x_noisy
    for t in reversed(range(diffusion_params['timesteps'])):
        x = p_sample(net_d, x, condition_x, t, diffusion_params)
    return x


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


def test_single_checkpoint(checkpoint_dir, iter_num, dataset, device, num_samples, save_dir=None, stage=1):
    """Test a single checkpoint and return metrics"""
    if stage == 1:
        net_g, net_le = load_stage1_model(checkpoint_dir, iter_num, device)
        if net_g is None:
            print(f"  Checkpoint {iter_num} not found, skipping...")
            return None, None, []
        net_le_dm, net_d, diffusion_params = None, None, None
    else:
        result = load_stage2_model(checkpoint_dir, iter_num, device)
        if result[0] is None:
            print(f"  Checkpoint {iter_num} not found, skipping...")
            return None, None, []
        net_g, net_le_dm, net_d = result
        net_le = None
        diffusion_params = setup_diffusion_params(device)

    psnr_list, ssim_list = [], []
    output_images = []

    with torch.no_grad():
        for idx in tqdm(range(min(len(dataset), num_samples)), desc="Testing"):
            data = dataset[idx]
            for key in data:
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].unsqueeze(0)

            input_features, lq_rgb, gt_rgb = prepare_input(data, device)

            if stage == 1:
                latent = net_le(input_features, gt_rgb)
                output = net_g(lq_rgb, latent)
            else:
                # Stage 2: use diffusion model
                prior_c = net_le_dm(input_features)
                prior_noisy = torch.randn_like(prior_c)
                prior = p_sample_loop(net_d, prior_c, prior_noisy, diffusion_params)
                output = net_g(lq_rgb, prior)

            output_img = tensor2img([output.squeeze(0)])
            gt_img = tensor2img([data['gt_rgb'].squeeze(0)])
            lq_img = tensor2img([lq_rgb.squeeze(0)])

            psnr = calculate_psnr(output_img, gt_img, crop_border=0, test_y_channel=False)
            ssim = calculate_ssim(output_img, gt_img, crop_border=0, test_y_channel=False)
            psnr_list.append(psnr)
            ssim_list.append(ssim)

            if save_dir and idx < 10:  # Save first 10 samples
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
            checkpoint_dir, iter_num, dataset, device, args.num_samples, save_dir, stage=args.stage
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
        checkpoint_dir, args.iter, dataset, device, args.num_samples, save_dir, stage=args.stage
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
