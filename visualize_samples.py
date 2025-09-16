#!/usr/bin/env python3
"""
Side-by-Side Visualization Script for H-cDDIM Paper.

This script generates a visual comparison of channel matrices from:
1. The ground truth (real) dataset.
2. The baseline cDDIM model (location-conditioned).
3. Our H-cDDIM model (hardware-and-location-conditioned).

It selects a number of hardware configurations, samples a real channel for each,
then generates corresponding channels from both models for a side-by-side plot.

Usage:
    python visualize_samples.py \
        --esh_model_path /path/to/H-cDDIM_model.pth \
        --cddim_model_path /path/to/cDDIM_model.pth \
        --dataset_path /path/to/DeepMIMO_dataset
"""

import argparse
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import random

# Add parent directory to path to import from other scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from esh_cddim_inference import ESHcDDIMInference
from run_comparison import cDDIMInference, load_reference_data
from load_deepmimo_datasets import load_deepmimo_datasets, create_ml_dataset

def generate_comparison_figure(esh_inference, cddim_inference, dataset_path, save_dir, n_configs=8):
    """
    Generates and saves the 8x3 comparison figure.
    """
    print("--- Starting Visualization Generation ---")
    
    # 1. Load data and find unique, compatible hardware configurations
    print("Loading dataset to find hardware configurations...")
    data = load_deepmimo_datasets(dataset_path, verbose=False)
    _, _, metadata = create_ml_dataset(data)

    unique_configs = []
    for m in metadata:
        if (m['bs_ant_h'] * m['bs_ant_v'] == 32) and (m['ue_ant_h'] * m['ue_ant_v'] == 4):
            config_tuple = (m['bs_ant_h'], m['bs_ant_v'], m['ue_ant_h'], m['ue_ant_v'], m['bs_spacing'], m['ue_spacing'])
            if config_tuple not in unique_configs:
                unique_configs.append(config_tuple)
    
    if len(unique_configs) < n_configs:
        print(f"Warning: Found only {len(unique_configs)} unique configs, but {n_configs} were requested. Using all available.")
        n_configs = len(unique_configs)

    selected_configs = random.sample(unique_configs, n_configs)
    print(f"Selected {n_configs} hardware configurations for visualization.")

    # 2. For each config, pick one random sample and generate channels
    results = {
        'configs': [],
        'gt_channels': [],
        'cddim_channels': [],
        'hcddim_channels': []
    }

    for config in selected_configs:
        print(f"Processing config: BS {config[0]}x{config[1]}, UE {config[2]}x{config[3]}, Spacing {config[4]}/{config[5]}")
        
        # Find all indices for this config
        indices_for_config = [i for i, m in enumerate(metadata) if 
                              (m['bs_ant_h'], m['bs_ant_v'], m['ue_ant_h'], m['ue_ant_v'], m['bs_spacing'], m['ue_spacing']) == config]
        
        # Randomly select one sample
        sample_index = random.choice(indices_for_config)
        
        # Load the ground truth channel and metadata
        gt_channel, gt_meta = load_reference_data(dataset_path, indices=[sample_index])
        gt_channel = gt_channel[0]
        gt_meta = gt_meta[0]
        
        # Prepare conditioning vectors
        location = gt_meta['user_location'].flatten()
        esh_cond = torch.tensor([list(location) + list(config)]).float()
        cddim_cond = torch.tensor([location]).float()
        
        # Generate channels from both models
        hcddim_channel = esh_inference.generate_channels(esh_cond, n_samples=1)[0]
        cddim_channel = cddim_inference.generate_channels(cddim_cond, n_samples=1)[0]
        
        # Store results
        results['configs'].append(config)
        results['gt_channels'].append(np.transpose(gt_channel.cpu().numpy(), (0, 2,3, 1)))
        results['cddim_channels'].append(cddim_channel.cpu().numpy())
        results['hcddim_channels'].append(hcddim_channel.cpu().numpy())

    # 3. Save the results for re-plotting
    save_path_npz = os.path.join(save_dir, 'visualization_samples.npz')
    np.savez_compressed(save_path_npz, **results)
    print(f"✓ All generated channel samples saved to {save_path_npz}")

    # 4. Create the plot
    plot_from_saved_data(save_path_npz, save_dir)


def plot_from_saved_data(data_path, save_dir):
    """
    Loads data from an .npz file and creates the comparison plot.
    """
    print(f"--- Plotting from saved data at {data_path} ---")
    data = np.load(data_path, allow_pickle=True)
    n_configs = len(data['configs'])

    fig, axes = plt.subplots(n_configs, 3, figsize=(12, 2 * n_configs))
    
    # Configure plot titles
    axes[0, 0].set_title("Ground Truth", fontsize=14)
    axes[0, 1].set_title("cDDIM (Baseline)", fontsize=14)
    axes[0, 2].set_title("H-cDDIM (Ours)", fontsize=14)

    for i in range(n_configs):
        config = data['configs'][i]
        gt_channel = data['gt_channels'][i]
        cddim_channel = data['cddim_channels'][i]
        hcddim_channel = data['hcddim_channels'][i]

        # Calculate channel magnitude
        gt_mag = np.sqrt(gt_channel[0]**2 + gt_channel[1]**2)
        cddim_mag = np.sqrt(cddim_channel[0]**2 + cddim_channel[1]**2)
        hcddim_mag = np.sqrt(hcddim_channel[0]**2 + hcddim_channel[1]**2)
        
        # Common color scale for each row
        vmin = min(gt_mag.min(), cddim_mag.min(), hcddim_mag.min())
        vmax = max(gt_mag.max(), cddim_mag.max(), hcddim_mag.max())

        # Plot Ground Truth
        im = axes[i, 0].imshow(gt_mag, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
        
        # Plot cDDIM
        axes[i, 1].imshow(cddim_mag, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
        
        # Plot H-cDDIM
        axes[i, 2].imshow(hcddim_mag, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
        
        # Row label
        config_label = f"BS:{config[0]}x{config[1]} UE:{config[2]}x{config[3]}\nSpacing:{config[4]}/{config[5]}"
        axes[i, 0].set_ylabel(config_label, fontsize=10, rotation=90, labelpad=20)
        
        # Remove ticks for clarity
        for j in range(3):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path_png = os.path.join(save_dir, 'visualization_comparison.png')
    plt.savefig(save_path_png, dpi=300)
    plt.show()
    print(f"✓ Comparison visualization saved to {save_path_png}")


def main():
    parser = argparse.ArgumentParser(description='Generate side-by-side visualizations of channel samples.')
    parser.add_argument('--esh_model_path', type=str, help='Path to the trained H-cDDIM model checkpoint')
    parser.add_argument('--cddim_model_path', type=str, help='Path to the trained baseline cDDIM model checkpoint')
    parser.add_argument('--dataset_path', type=str, default="../../datasets/DeepMIMO_dataset_full", help='Path to the DeepMIMO test dataset')
    parser.add_argument('--save_dir', type=str, default='./results/visualization', help='Directory to save results')
    parser.add_argument('--n_configs', type=int, default=8, help='Number of hardware configurations to visualize')
    parser.add_argument('--replot_from', type=str, help='Path to an .npz file to regenerate the plot from saved data')
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    if args.replot_from:
        plot_from_saved_data(args.replot_from, args.save_dir)
    else:
        if not args.esh_model_path or not args.cddim_model_path:
            print("Error: Model paths must be provided unless using --replot_from.")
            sys.exit(1)
            
        esh_inference = ESHcDDIMInference(args.esh_model_path)
        cddim_inference = cDDIMInference(args.cddim_model_path)
        generate_comparison_figure(esh_inference, cddim_inference, args.dataset_path, args.save_dir, args.n_configs)

if __name__ == "__main__":
    main()
