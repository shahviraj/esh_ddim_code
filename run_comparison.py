#!/usr/bin/env python3
"""
Comparison script for ESH-cDDIM vs. baseline cDDIM models.

This script runs a series of evaluations to quantitatively and qualitatively
compare the performance of the hardware-aware ESH-cDDIM against a baseline
cDDIM model that only conditions on user location.

Usage:
    python run_comparison.py --esh_model_path <path> --cddim_model_path <path> --dataset_path <path> --mode all
"""

import argparse
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import json
from tqdm import tqdm
from typing import List, Tuple, Dict
from torchvision.transforms import ToTensor

# Add parent directory to path to import from other scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Imports from our project
from esh_cddim_inference import ESHcDDIMInference, load_reference_data
from cddim_comparison_train import ContextUnet as cDDIM_ContextUnet, DDIM as cDDIM_DDIM
from load_deepmimo_datasets import load_deepmimo_datasets, create_ml_dataset
from scipy.stats import wasserstein_distance, ks_2samp
from sklearn.metrics.pairwise import rbf_kernel


def calculate_mmd(x, y, sigma=None):
    """Calculates the Maximum Mean Discrepancy (MMD) between two samples."""
    # Reshape for kernel function
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    
    # Heuristic for sigma: median pairwise distance
    if sigma is None:
        all_samples = np.vstack([x, y])
        distances = np.median(np.abs(all_samples - np.median(all_samples)))
        sigma = distances if distances > 0 else 1.0

    # Calculate kernel matrices
    gamma = 1.0 / (2 * sigma**2)
    k_xx = rbf_kernel(x, x, gamma=gamma)
    k_yy = rbf_kernel(y, y, gamma=gamma)
    k_xy = rbf_kernel(x, y, gamma=gamma)
    
    # Calculate MMD^2
    mmd2 = k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()
    return np.sqrt(mmd2) if mmd2 > 0 else 0


class cDDIMInference:
    """
    A simplified inference class for the baseline cDDIM model.
    """
    def __init__(self, model_path: str, device: str = "cuda:0" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model_path = model_path
        
        # Model parameters for baseline cDDIM
        self.n_feat = 256
        self.n_classes = 3  # Location (x, y, z) only
        self.n_T = 256
        self.betas = (1e-4, 0.02)
        
        # Initialize model from cddim_comparison_train script
        self.model = cDDIM_ContextUnet(in_channels=2, n_feat=self.n_feat, n_classes=self.n_classes)
        
        self.ddim = cDDIM_DDIM(
            nn_model=self.model,
            betas=self.betas,
            n_T=self.n_T,
            device=self.device,
            drop_prob=0.1
        )
        self.load_model()

    def load_model(self):
        """Load the trained model weights."""
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.ddim.load_state_dict(checkpoint)
            self.ddim.to(self.device)
            self.ddim.eval()
            print(f"✓ Baseline cDDIM model loaded successfully from {self.model_path}")
        except Exception as e:
            print(f"✗ Error loading baseline cDDIM model: {e}")
            raise

    def generate_channels(self, context_vectors: torch.Tensor, n_samples: int, guidance_scale: float = 0.0) -> torch.Tensor:
        """Generate channels using the baseline cDDIM model."""
        if context_vectors.shape[1] != 3:
            raise ValueError(f"Baseline cDDIM context vectors must have 3 dimensions (location), got {context_vectors.shape[1]}")
        
        context_vectors = context_vectors.to(dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            generated_channels, _ = self.ddim.sample(
                n_sample=n_samples,
                c_test=context_vectors,
                size=(2, 4, 32),  # Channel dimensions
                device=self.device,
                guide_w=guidance_scale
            )
        return generated_channels


def load_reference_data(dataset_path: str, indices: List[int]) -> Tuple[torch.Tensor, List[Dict]]:
    """
    Loads specific reference samples from the DeepMIMO dataset by index and
    applies the same preprocessing used during training.
    """
    print(f"Loading {len(indices)} specific samples from {dataset_path}...")
    
    # Load the entire dataset structure to access metadata and raw channels
    data = load_deepmimo_datasets(dataset_path, verbose=False)
    X_flat, _, all_metadata = create_ml_dataset(data)
    
    selected_channels_flat = X_flat[indices]
    selected_metadata = [all_metadata[i] for i in indices]
    
    processed_channels = []
    for i in tqdm(range(len(selected_channels_flat)), desc="Processing reference channels"):
        channel_flat = selected_channels_flat[i]
        meta = selected_metadata[i]
        
        # Replicate the channel processing from the training script's BerUMaLDataset
        bs_ant_h, bs_ant_v = meta['bs_ant_h'], meta['bs_ant_v']
        ue_ant_h, ue_ant_v = meta['ue_ant_h'], meta['ue_ant_v']
        
        channel_2d = channel_flat.reshape(ue_ant_h * ue_ant_v, bs_ant_h * bs_ant_v)
        
        # Stack real and imaginary parts
        array1 = np.stack((np.real(channel_2d), np.imag(channel_2d)), axis=0)
        
        # Apply FFT and normalization
        dft_data = np.fft.fft2(array1[0] + 1j * array1[1])
        dft_shifted = np.fft.fftshift(dft_data)
        array1[0] = np.real(dft_shifted)
        array1[1] = np.imag(dft_shifted)
        
        magnitude = np.sqrt(array1[0, :, :]**2 + array1[1, :, :]**2)
        max_magnitude = np.max(magnitude)
        if max_magnitude > 0:
            array1[0, :, :] /= max_magnitude
            array1[1, :, :] /= max_magnitude
            
        processed_channels.append(ToTensor()(array1[:2, :, :]).float())

    return torch.stack(processed_channels), selected_metadata


def run_hardware_specific_fidelity_test(esh_inference, cddim_inference, dataset_path, save_dir):
    """
    Evaluates how well each model matches the distribution of a specific hardware configuration.
    """
    print("\n=== Running Hardware-Specific Fidelity Test ===")
    
    # 1. Load data and select a specific hardware configuration to test against
    print("Loading and filtering dataset for a specific hardware configuration...")
    data = load_deepmimo_datasets(dataset_path, verbose=False)
    _, _, metadata = create_ml_dataset(data)

    # Let's choose the first available hardware configuration as our target
    if not metadata:
        print("Error: No metadata found in the dataset.")
        return

    target_config = {
        'bs_ant_h': metadata[0]['bs_ant_h'], 'bs_ant_v': metadata[0]['bs_ant_v'],
        'ue_ant_h': metadata[0]['ue_ant_h'], 'ue_ant_v': metadata[0]['ue_ant_v'],
        'bs_spacing': metadata[0]['bs_spacing'], 'ue_spacing': metadata[0]['ue_spacing']
    }
    print(f"Target hardware configuration for test: {target_config}")

    # Filter the dataset for this specific configuration
    indices = [
        i for i, m in enumerate(metadata) if
        m['bs_ant_h'] == target_config['bs_ant_h'] and m['bs_ant_v'] == target_config['bs_ant_v'] and
        m['ue_ant_h'] == target_config['ue_ant_h'] and m['ue_ant_v'] == target_config['ue_ant_v'] and
        m['bs_spacing'] == target_config['bs_spacing'] and m['ue_spacing'] == target_config['ue_spacing']
    ]
    
    if len(indices) < 10:
        print(f"Warning: Found only {len(indices)} samples for the target configuration. Results may not be robust.")
        if not indices: return

    # Get ground truth channels and prepare conditioning vectors
    gt_channels, gt_metadata = load_reference_data(dataset_path, indices=indices)
    
    esh_cond = torch.tensor(np.array([
        list(m['user_location'].flatten()) + [m['bs_ant_h'], m['bs_ant_v'], m['ue_ant_h'], m['ue_ant_v'], m['bs_spacing'], m['ue_spacing']]
        for m in gt_metadata
    ]))
    cddim_cond = torch.tensor(np.array([m['user_location'].flatten() for m in gt_metadata]))

    # 2. Generate channels from both models
    print(f"Generating {len(indices)} samples from both models...")
    esh_generated = esh_inference.generate_channels(esh_cond, n_samples=len(indices))
    cddim_generated = cddim_inference.generate_channels(cddim_cond, n_samples=len(indices))

    # 3. Compare distributions
    print("Calculating statistics and comparing distributions...")
    gt_stats = esh_inference.calculate_channel_statistics(gt_channels)
    esh_stats = esh_inference.calculate_channel_statistics(esh_generated)
    cddim_stats = esh_inference.calculate_channel_statistics(cddim_generated)

    results = {}
    metrics_to_compare = ['capacity', 'frobenius_norm']

    fig, axes = plt.subplots(1, len(metrics_to_compare), figsize=(16, 6))
    fig.suptitle('Hardware-Specific Fidelity Comparison', fontsize=16)

    for i, metric in enumerate(metrics_to_compare):
        # Calculate Wasserstein distances
        w_dist_esh = wasserstein_distance(gt_stats[metric], esh_stats[metric])
        w_dist_cddim = wasserstein_distance(gt_stats[metric], cddim_stats[metric])
        results[f'w_dist_{metric}_esh'] = w_dist_esh
        results[f'w_dist_{metric}_cddim'] = w_dist_cddim

        # Calculate MMD
        mmd_esh = calculate_mmd(gt_stats[metric], esh_stats[metric])
        mmd_cddim = calculate_mmd(gt_stats[metric], cddim_stats[metric])
        results[f'mmd_{metric}_esh'] = mmd_esh
        results[f'mmd_{metric}_cddim'] = mmd_cddim

        # Calculate KS Statistic
        ks_esh = ks_2samp(gt_stats[metric], esh_stats[metric])
        ks_cddim = ks_2samp(gt_stats[metric], cddim_stats[metric])
        results[f'ks_stat_{metric}_esh'] = ks_esh.statistic
        results[f'ks_pvalue_{metric}_esh'] = ks_esh.pvalue
        results[f'ks_stat_{metric}_cddim'] = ks_cddim.statistic
        results[f'ks_pvalue_{metric}_cddim'] = ks_cddim.pvalue

        print(f"\n--- Metrics for {metric.replace('_', ' ').title()} ---")
        print(f"  Wasserstein Distance:")
        print(f"    - H-cDDIM: {w_dist_esh:.4f}")
        print(f"    - cDDIM:   {w_dist_cddim:.4f}")
        print(f"  Maximum Mean Discrepancy (MMD):")
        print(f"    - H-cDDIM: {mmd_esh:.4f}")
        print(f"    - cDDIM:   {mmd_cddim:.4f}")
        print(f"  Kolmogorov-Smirnov (KS) Statistic:")
        print(f"    - H-cDDIM: {ks_esh.statistic:.4f} (p-value: {ks_esh.pvalue:.4f})")
        print(f"    - cDDIM:   {ks_cddim.statistic:.4f} (p-value: {ks_cddim.pvalue:.4f})")

        # Plot distributions
        sns.kdeplot(gt_stats[metric], ax=axes[i], label='Ground Truth', fill=True, color='blue')
        sns.kdeplot(esh_stats[metric], ax=axes[i], label=f'H-cDDIM (W-dist: {w_dist_esh:.2f})', fill=True, color='green')
        sns.kdeplot(cddim_stats[metric], ax=axes[i], label=f'cDDIM (W-dist: {w_dist_cddim:.2f})', fill=True, color='red')
        axes[i].set_title(f'Distribution of {metric.replace("_", " ").title()}')
        axes[i].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(save_dir, 'comparison_fidelity_test.png')
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"✓ Fidelity comparison plot saved to {save_path}")

    # Save results
    results_path = os.path.join(save_dir, 'comparison_fidelity_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Fidelity comparison results saved to {results_path}")

    # 4. Save the generated channels for re-plotting
    channel_data_path = os.path.join(save_dir, 'fidelity_test_channels.npz')
    np.savez_compressed(
        channel_data_path,
        gt_channels=gt_channels.cpu().numpy(),
        esh_generated=esh_generated.cpu().numpy(),
        cddim_generated=cddim_generated.cpu().numpy()
    )
    print(f"✓ Channel data for re-plotting saved to {channel_data_path}")


def replot_fidelity_results(data_path: str, save_dir: str):
    """
    Loads saved channel data and regenerates the fidelity comparison plot.
    """
    print(f"\n=== Re-plotting Fidelity Results from {data_path} ===")
    
    try:
        data = np.load(data_path)
        gt_channels = torch.from_numpy(data['gt_channels'])
        esh_generated = torch.from_numpy(data['esh_generated'])
        cddim_generated = torch.from_numpy(data['cddim_generated'])
    except Exception as e:
        print(f"Error: Could not load data from {data_path}. {e}")
        return

    # We need an inference object to access the calculate_channel_statistics method.
    # We can create a dummy one since we don't need a loaded model.
    dummy_inference = ESHcDDIMInference.__new__(ESHcDDIMInference)

    # Calculate statistics
    gt_stats = dummy_inference.calculate_channel_statistics(gt_channels)
    esh_stats = dummy_inference.calculate_channel_statistics(esh_generated)
    cddim_stats = dummy_inference.calculate_channel_statistics(cddim_generated)

    # Re-generate the plot with new labels
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Hardware-Specific Fidelity Comparison', fontsize=16)

    metrics_to_compare = ['capacity', 'frobenius_norm']

    print("\nRecalculating quantitative metrics from saved data:")
    for i, metric in enumerate(metrics_to_compare):
        # Wasserstein distance
        w_dist_esh = wasserstein_distance(gt_stats[metric], esh_stats[metric])
        w_dist_cddim = wasserstein_distance(gt_stats[metric], cddim_stats[metric])
        
        # MMD
        mmd_esh = calculate_mmd(gt_stats[metric], esh_stats[metric])
        mmd_cddim = calculate_mmd(gt_stats[metric], cddim_stats[metric])

        # KS Statistic
        ks_esh = ks_2samp(gt_stats[metric], esh_stats[metric])
        ks_cddim = ks_2samp(gt_stats[metric], cddim_stats[metric])

        print(f"\n--- Metrics for {metric.replace('_', ' ').title()} ---")
        print(f"  Wasserstein Distance:")
        print(f"    - H-cDDIM: {w_dist_esh:.4f}")
        print(f"    - cDDIM:   {w_dist_cddim:.4f}")
        print(f"  Maximum Mean Discrepancy (MMD):")
        print(f"    - H-cDDIM: {mmd_esh:.4f}")
        print(f"    - cDDIM:   {mmd_cddim:.4f}")
        print(f"  Kolmogorov-Smirnov (KS) Statistic:")
        print(f"    - H-cDDIM: {ks_esh.statistic:.4f} (p-value: {ks_esh.pvalue:.4f})")
        print(f"    - cDDIM:   {ks_cddim.statistic:.4f} (p-value: {ks_cddim.pvalue:.4f})")

        # USE THE NEW "H-cDDIM" LABEL HERE
        sns.kdeplot(gt_stats[metric], ax=axes[i], label='Ground Truth', fill=True, color='blue')
        sns.kdeplot(esh_stats[metric], ax=axes[i], label=f'H-cDDIM (W-dist: {w_dist_esh:.2f})', fill=True, color='green')
        sns.kdeplot(cddim_stats[metric], ax=axes[i], label=f'cDDIM (W-dist: {w_dist_cddim:.2f})', fill=True, color='red')
        axes[i].set_title(f'Distribution of {metric.replace("_", " ").title()}')
        axes[i].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(save_dir, 'comparison_fidelity_test_replotted.png')
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"✓ Re-plotted fidelity comparison plot saved to {save_path}")


def run_cross_hardware_test(esh_inference, cddim_inference, dataset_path, save_dir, n_avg_samples=50):
    """
    Visually demonstrates that ESH-cDDIM produces different channels for different
    hardware configs at the same location, while cDDIM does not, comparing against
    the average of ground truth samples.
    """
    print("\n=== Running Cross-Hardware Generation Test ===")

    # 1. Load data and find two different hardware configs
    print("Searching for suitable hardware configurations...")
    data = load_deepmimo_datasets(dataset_path, verbose=False)
    _, _, metadata = create_ml_dataset(data)

    configs = []
    for m in metadata:
        if (m['bs_ant_h'] * m['bs_ant_v'] == 32) and (m['ue_ant_h'] * m['ue_ant_v'] == 4):
            config_tuple = (m['bs_ant_h'], m['bs_ant_v'], m['ue_ant_h'], m['ue_ant_v'], m['bs_spacing'], m['ue_spacing'])
            if config_tuple not in configs:
                configs.append(config_tuple)
    
    if len(configs) < 2:
        print("Error: Could not find at least two different compatible hardware configurations in the dataset.")
        return

    config_a_tuple, config_b_tuple = configs[0], configs[1]
    
    def get_full_config(cfg_tuple):
        return {'bs_ant_h': cfg_tuple[0], 'bs_ant_v': cfg_tuple[1], 'ue_ant_h': cfg_tuple[2], 'ue_ant_v': cfg_tuple[3], 'bs_spacing': cfg_tuple[4], 'ue_spacing': cfg_tuple[5]}

    config_a = get_full_config(config_a_tuple)
    config_b = get_full_config(config_b_tuple)

    # Find a sample for each configuration
    meta_a = next((m for m in metadata if (m['bs_ant_h'], m['bs_ant_v'], m['ue_ant_h'], m['ue_ant_v'], m['bs_spacing'], m['ue_spacing']) == config_a_tuple), None)
    meta_b = next((m for m in metadata if (m['bs_ant_h'], m['bs_ant_v'], m['ue_ant_h'], m['ue_ant_v'], m['bs_spacing'], m['ue_spacing']) == config_b_tuple), None)
    
    if not meta_a or not meta_b:
        print("Error: Could not find samples for both selected configurations.")
        return

    test_location = meta_a['user_location']

    print(f"Using test location: {test_location.flatten()}")
    print(f"Config A: {config_a}")
    print(f"Config B: {config_b}")

    # 2. Prepare conditioning vectors (repeated for averaging)
    esh_cond_a = torch.tensor([list(test_location.flatten()) + list(config_a_tuple)] * n_avg_samples).float()
    esh_cond_b = torch.tensor([list(test_location.flatten()) + list(config_b_tuple)] * n_avg_samples).float()
    cddim_cond = torch.tensor([test_location.flatten()] * n_avg_samples).float()

    # 3. Generate n_avg_samples for each model/config
    print(f"Generating {n_avg_samples} samples for each condition for averaging...")
    esh_gen_a = esh_inference.generate_channels(esh_cond_a, n_samples=n_avg_samples)
    esh_gen_b = esh_inference.generate_channels(esh_cond_b, n_samples=n_avg_samples)
    cddim_gen = cddim_inference.generate_channels(cddim_cond, n_samples=n_avg_samples)

    # 4. Load all ground truth channels for each config
    indices_a = [i for i, m in enumerate(metadata) if (m['bs_ant_h'], m['bs_ant_v'], m['ue_ant_h'], m['ue_ant_v'], m['bs_spacing'], m['ue_spacing']) == config_a_tuple]
    indices_b = [i for i, m in enumerate(metadata) if (m['bs_ant_h'], m['bs_ant_v'], m['ue_ant_h'], m['ue_ant_v'], m['bs_spacing'], m['ue_spacing']) == config_b_tuple]
    
    gt_channels_a, _ = load_reference_data(dataset_path, indices=indices_a)
    gt_channels_b, _ = load_reference_data(dataset_path, indices=indices_b)
    
    # 5. Calculate and average singular values
    def get_avg_singular_values(channels):
        stats = esh_inference.calculate_channel_statistics(channels)
        # Pad singular values to the same length for averaging
        max_len = max(len(s) for s in stats['singular_values'])
        padded_sv = [np.pad(s, (0, max_len - len(s))) for s in stats['singular_values']]
        return np.mean(padded_sv, axis=0)

    avg_sv_esh_a = get_avg_singular_values(esh_gen_a)
    avg_sv_esh_b = get_avg_singular_values(esh_gen_b)
    avg_sv_cddim = get_avg_singular_values(cddim_gen)
    avg_sv_gt_a = get_avg_singular_values(gt_channels_a)
    avg_sv_gt_b = get_avg_singular_values(gt_channels_b)

    # 6. Plot the averaged singular value curves
    plt.figure(figsize=(12, 8))
    plt.plot(avg_sv_gt_a, 'o-', label=f'Ground Truth (Config A)', color='blue', markersize=8)
    plt.plot(avg_sv_esh_a, 's--', label=f'ESH-cDDIM (Config A)', color='green', markersize=6)
    plt.plot(avg_sv_gt_b, 'o-', label=f'Ground Truth (Config B)', color='orange', markersize=8)
    plt.plot(avg_sv_esh_b, 's--', label=f'ESH-cDDIM (Config B)', color='purple', markersize=6)
    plt.plot(avg_sv_cddim, 'x-.', label='Baseline cDDIM (Hardware-Agnostic)', color='red', markersize=8)
    
    plt.title('Average Singular Value Distribution for Different Hardware Configurations', fontsize=16)
    plt.xlabel('Singular Value Index', fontsize=12)
    plt.ylabel('Average Singular Value Magnitude', fontsize=12)
    plt.yscale('log')
    plt.legend(fontsize=10)
    plt.grid(True, which="both", ls="--")
    
    save_path = os.path.join(save_dir, 'comparison_cross_hardware_test_averaged.png')
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"✓ Averaged cross-hardware comparison plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Run comparison between ESH-cDDIM and baseline cDDIM.')
    parser.add_argument('--esh_model_path', type=str, required=True, help='Path to the trained ESH-cDDIM model checkpoint')
    parser.add_argument('--cddim_model_path', type=str, required=True, help='Path to the trained baseline cDDIM model checkpoint')
    parser.add_argument('--dataset_path', type=str, default="../../datasets/DeepMIMO_dataset_full", help='Path to the DeepMIMO test dataset')
    parser.add_argument('--mode', type=str, default='all', choices=['all', 'fidelity', 'cross_hardware', 'replot'], help='Comparison mode to run')
    parser.add_argument('--save_dir', type=str, default='./results/comparison', help='Directory to save comparison results')
    parser.add_argument('--n_avg_samples', type=int, default=50, help='Number of samples to generate for averaging in cross-hardware test')
    parser.add_argument('--replot_data_path', type=str, help='Path to .npz file with channel data for re-plotting')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    if args.mode == 'replot':
        if not args.replot_data_path:
            print("Error: --mode replot requires --replot_data_path to be set.")
            sys.exit(1)
        replot_fidelity_results(args.replot_data_path, args.save_dir)
        print("\n✓ Re-plotting finished.")
        return

    # Initialize inference engines for both models
    esh_inference = ESHcDDIMInference(args.esh_model_path)
    cddim_inference = cDDIMInference(args.cddim_model_path)

    if args.mode in ['all', 'fidelity']:
        run_hardware_specific_fidelity_test(esh_inference, cddim_inference, args.dataset_path, args.save_dir)
    
    if args.mode in ['all', 'cross_hardware']:
        run_cross_hardware_test(esh_inference, cddim_inference, args.dataset_path, args.save_dir, n_avg_samples=args.n_avg_samples)

    print("\n✓ Comparison evaluation finished.")

if __name__ == "__main__":
    main()
