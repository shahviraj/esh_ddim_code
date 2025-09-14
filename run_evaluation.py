#!/usr/bin/env python3
"""
Evaluation script for ESH-cDDIM model.
This script runs comprehensive evaluations and generates results for the research paper.

Usage:
    python run_evaluation.py --mode all --model_path ./data/cDDIM_10/model.pth
    python run_evaluation.py --mode spatial --model_path ./data/cDDIM_10/model.pth
    python run_evaluation.py --mode hardware --model_path ./data/cDDIM_10/model.pth
"""

import argparse
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime

# Import our inference class
from esh_cddim_inference import ESHcDDIMInference, create_test_context_vectors, load_reference_data

def run_spatial_generalization_evaluation(inference, save_dir, n_samples=100):
    """
    Evaluate spatial generalization by generating channels for different locations.
    """
    print("=== Spatial Generalization Evaluation ===")
    
    # Create test locations in a grid pattern
    x_coords = np.linspace(-20, 20, 10)
    y_coords = np.linspace(-20, 20, 10)
    z_coords = [1.5]  # Fixed height
    
    results = {
        'locations': [],
        'generated_channels': [],
        'context_vectors': []
    }
    
    for x in x_coords:
        for y in y_coords:
            for z in z_coords:
                # Create context vector for this location
                context = torch.zeros(1, 9, device=inference.device)
                context[0, 0] = x  # x coordinate
                context[0, 1] = y  # y coordinate
                context[0, 2] = z  # z coordinate
                
                # Fixed hardware configuration
                context[0, 3] = 8   # bs_ant_h
                context[0, 4] = 4   # bs_ant_v
                context[0, 5] = 2   # ue_ant_h
                context[0, 6] = 2   # ue_ant_v
                context[0, 7] = 0.5 # bs_spacing
                context[0, 8] = 0.5 # ue_spacing
                
                # Generate channel
                generated = inference.generate_channels(context, n_samples=1, guidance_scale=0.0)
                
                results['locations'].append([x, y, z])
                results['generated_channels'].append(generated[0].cpu().numpy())
                results['context_vectors'].append(context[0].cpu().numpy())
    
    # Analyze spatial patterns
    channels_array = np.array(results['generated_channels'])
    locations_array = np.array(results['locations'])
    
    # Calculate channel power for each location
    channel_powers = []
    for i in range(len(channels_array)):
        H = channels_array[i, 0] + 1j * channels_array[i, 1]
        power = np.sum(np.abs(H)**2)
        channel_powers.append(power)
    
    channel_powers = np.array(channel_powers)
    
    # Create spatial power map
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Reshape for plotting
    x_unique = np.unique(locations_array[:, 0])
    y_unique = np.unique(locations_array[:, 1])
    power_grid = channel_powers.reshape(len(x_unique), len(y_unique))
    
    im = ax.imshow(power_grid, extent=[x_unique.min(), x_unique.max(), 
                                     y_unique.min(), y_unique.max()], 
                   origin='lower', cmap='viridis', aspect='equal')
    ax.set_xlabel('X Coordinate (m)')
    ax.set_ylabel('Y Coordinate (m)')
    ax.set_title('Spatial Channel Power Distribution')
    plt.colorbar(im, ax=ax, label='Channel Power')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'spatial_generalization.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save results
    np.savez(os.path.join(save_dir, 'spatial_evaluation.npz'),
             locations=locations_array,
             channel_powers=channel_powers,
             generated_channels=channels_array)
    
    print(f"✓ Spatial generalization evaluation completed. Results saved to {save_dir}")
    return results

def run_hardware_awareness_evaluation(inference, save_dir, n_samples=50):
    """
    Evaluate hardware awareness by testing different antenna configurations.
    """
    print("=== Hardware Awareness Evaluation ===")
    
    # Define hardware configurations to test
    hardware_configs = [
        {'name': 'Small_BS_Small_UE', 'bs_ant_h': 4, 'bs_ant_v': 2, 'ue_ant_h': 1, 'ue_ant_v': 1, 'bs_spacing': 0.5, 'ue_spacing': 0.5},
        {'name': 'Medium_BS_Small_UE', 'bs_ant_h': 8, 'bs_ant_v': 4, 'ue_ant_h': 1, 'ue_ant_v': 1, 'bs_spacing': 0.5, 'ue_spacing': 0.5},
        {'name': 'Large_BS_Small_UE', 'bs_ant_h': 16, 'bs_ant_v': 8, 'ue_ant_h': 1, 'ue_ant_v': 1, 'bs_spacing': 0.5, 'ue_spacing': 0.5},
        {'name': 'Medium_BS_Medium_UE', 'bs_ant_h': 8, 'bs_ant_v': 4, 'ue_ant_h': 2, 'ue_ant_v': 2, 'bs_spacing': 0.5, 'ue_spacing': 0.5},
        {'name': 'Large_BS_Medium_UE', 'bs_ant_h': 16, 'bs_ant_v': 8, 'ue_ant_h': 2, 'ue_ant_v': 2, 'bs_spacing': 0.5, 'ue_spacing': 0.5},
        {'name': 'Dense_Spacing', 'bs_ant_h': 8, 'bs_ant_v': 4, 'ue_ant_h': 2, 'ue_ant_v': 2, 'bs_spacing': 0.3, 'ue_spacing': 0.3},
        {'name': 'Sparse_Spacing', 'bs_ant_h': 8, 'bs_ant_v': 4, 'ue_ant_h': 2, 'ue_ant_v': 2, 'bs_spacing': 0.7, 'ue_spacing': 0.7},
    ]
    
    results = {}
    
    for config in hardware_configs:
        print(f"Testing configuration: {config['name']}")
        
        # Create context vector
        context = torch.zeros(n_samples, 9, device=inference.device)
        context[:, 0] = torch.randn(n_samples, device=inference.device) * 5  # x
        context[:, 1] = torch.randn(n_samples, device=inference.device) * 5  # y
        context[:, 2] = torch.rand(n_samples, device=inference.device) * 2 + 1  # z
        context[:, 3] = config['bs_ant_h']
        context[:, 4] = config['bs_ant_v']
        context[:, 5] = config['ue_ant_h']
        context[:, 6] = config['ue_ant_v']
        context[:, 7] = config['bs_spacing']
        context[:, 8] = config['ue_spacing']
        
        # Generate channels
        generated = inference.generate_channels(context, n_samples=n_samples, guidance_scale=0.0)
        
        # Analyze channel properties
        channels_np = generated.cpu().numpy()
        channel_properties = []
        
        for i in range(n_samples):
            H = channels_np[i, 0] + 1j * channels_np[i, 1]
            
            # Calculate properties
            U, S, Vh = np.linalg.svd(H)
            rank = np.linalg.matrix_rank(H)
            condition_number = np.max(S) / np.min(S[S > 1e-10]) if np.min(S[S > 1e-10]) > 0 else np.inf
            frobenius_norm = np.linalg.norm(H, 'fro')
            channel_power = np.sum(np.abs(H)**2)
            
            # MIMO capacity (assuming equal power allocation)
            capacity = np.sum(np.log2(1 + S**2))
            
            channel_properties.append({
                'rank': rank,
                'condition_number': condition_number,
                'frobenius_norm': frobenius_norm,
                'channel_power': channel_power,
                'capacity': capacity,
                'singular_values': S
            })
        
        results[config['name']] = {
            'config': config,
            'channel_properties': channel_properties,
            'generated_channels': channels_np
        }
    
    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    config_names = list(results.keys())
    
    # Channel rank comparison
    ranks = [np.mean([prop['rank'] for prop in results[cfg]['channel_properties']]) for cfg in config_names]
    axes[0, 0].bar(config_names, ranks)
    axes[0, 0].set_title('Average Channel Rank')
    axes[0, 0].set_ylabel('Rank')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Condition number comparison
    cond_nums = [np.mean([prop['condition_number'] for prop in results[cfg]['channel_properties']]) for cfg in config_names]
    axes[0, 1].bar(config_names, cond_nums)
    axes[0, 1].set_title('Average Condition Number')
    axes[0, 1].set_ylabel('Condition Number')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].set_yscale('log')
    
    # Channel power comparison
    powers = [np.mean([prop['channel_power'] for prop in results[cfg]['channel_properties']]) for cfg in config_names]
    axes[0, 2].bar(config_names, powers)
    axes[0, 2].set_title('Average Channel Power')
    axes[0, 2].set_ylabel('Power')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # Capacity comparison
    capacities = [np.mean([prop['capacity'] for prop in results[cfg]['channel_properties']]) for cfg in config_names]
    axes[1, 0].bar(config_names, capacities)
    axes[1, 0].set_title('Average MIMO Capacity')
    axes[1, 0].set_ylabel('Capacity (bits/s/Hz)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Frobenius norm comparison
    fro_norms = [np.mean([prop['frobenius_norm'] for prop in results[cfg]['channel_properties']]) for cfg in config_names]
    axes[1, 1].bar(config_names, fro_norms)
    axes[1, 1].set_title('Average Frobenius Norm')
    axes[1, 1].set_ylabel('Frobenius Norm')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # Singular value distributions
    for i, cfg in enumerate(config_names):
        all_singular_values = []
        for prop in results[cfg]['channel_properties']:
            all_singular_values.extend(prop['singular_values'])
        axes[1, 2].hist(all_singular_values, alpha=0.6, label=cfg, bins=20)
    axes[1, 2].set_title('Singular Value Distributions')
    axes[1, 2].set_xlabel('Singular Value')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].legend()
    axes[1, 2].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'hardware_awareness_evaluation.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save detailed results
    np.savez(os.path.join(save_dir, 'hardware_awareness_evaluation.npz'), **results)
    
    print(f"✓ Hardware awareness evaluation completed. Results saved to {save_dir}")
    return results

def run_state_awareness_evaluation(inference, save_dir, n_samples=50):
    """
    Evaluate state awareness by testing LoS/NLoS scenarios.
    Note: This is a simplified version since we don't have actual LoS/NLoS labels in our context.
    We'll simulate this by varying the z-coordinate (height) which affects propagation.
    """
    print("=== State Awareness Evaluation ===")
    
    # Test different heights (simulating different propagation states)
    heights = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]  # Different heights
    
    results = {}
    
    for height in heights:
        print(f"Testing height: {height}m")
        
        # Create context vectors with different heights
        context = torch.zeros(n_samples, 9, device=inference.device)
        context[:, 0] = torch.randn(n_samples, device=inference.device) * 5  # x
        context[:, 1] = torch.randn(n_samples, device=inference.device) * 5  # y
        context[:, 2] = height  # z (height)
        context[:, 3] = 8   # bs_ant_h
        context[:, 4] = 4   # bs_ant_v
        context[:, 5] = 2   # ue_ant_h
        context[:, 6] = 2   # ue_ant_v
        context[:, 7] = 0.5 # bs_spacing
        context[:, 8] = 0.5 # ue_spacing
        
        # Generate channels
        generated = inference.generate_channels(context, n_samples=n_samples, guidance_scale=0.0)
        
        # Analyze channel properties
        channels_np = generated.cpu().numpy()
        channel_properties = []
        
        for i in range(n_samples):
            H = channels_np[i, 0] + 1j * channels_np[i, 1]
            
            # Calculate properties
            U, S, Vh = np.linalg.svd(H)
            rank = np.linalg.matrix_rank(H)
            condition_number = np.max(S) / np.min(S[S > 1e-10]) if np.min(S[S > 1e-10]) > 0 else np.inf
            frobenius_norm = np.linalg.norm(H, 'fro')
            channel_power = np.sum(np.abs(H)**2)
            capacity = np.sum(np.log2(1 + S**2))
            
            channel_properties.append({
                'rank': rank,
                'condition_number': condition_number,
                'frobenius_norm': frobenius_norm,
                'channel_power': channel_power,
                'capacity': capacity
            })
        
        results[f'height_{height}'] = {
            'height': height,
            'channel_properties': channel_properties,
            'generated_channels': channels_np
        }
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    heights_list = [results[f'height_{h}']['height'] for h in heights]
    
    # Channel power vs height
    powers = [np.mean([prop['channel_power'] for prop in results[f'height_{h}']['channel_properties']]) for h in heights]
    axes[0, 0].plot(heights_list, powers, 'o-')
    axes[0, 0].set_title('Channel Power vs Height')
    axes[0, 0].set_xlabel('Height (m)')
    axes[0, 0].set_ylabel('Channel Power')
    
    # Capacity vs height
    capacities = [np.mean([prop['capacity'] for prop in results[f'height_{h}']['channel_properties']]) for h in heights]
    axes[0, 1].plot(heights_list, capacities, 'o-')
    axes[0, 1].set_title('MIMO Capacity vs Height')
    axes[0, 1].set_xlabel('Height (m)')
    axes[0, 1].set_ylabel('Capacity (bits/s/Hz)')
    
    # Condition number vs height
    cond_nums = [np.mean([prop['condition_number'] for prop in results[f'height_{h}']['channel_properties']]) for h in heights]
    axes[1, 0].plot(heights_list, cond_nums, 'o-')
    axes[1, 0].set_title('Condition Number vs Height')
    axes[1, 0].set_xlabel('Height (m)')
    axes[1, 0].set_ylabel('Condition Number')
    axes[1, 0].set_yscale('log')
    
    # Channel rank vs height
    ranks = [np.mean([prop['rank'] for prop in results[f'height_{h}']['channel_properties']]) for h in heights]
    axes[1, 1].plot(heights_list, ranks, 'o-')
    axes[1, 1].set_title('Channel Rank vs Height')
    axes[1, 1].set_xlabel('Height (m)')
    axes[1, 1].set_ylabel('Channel Rank')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'state_awareness_evaluation.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save results
    np.savez(os.path.join(save_dir, 'state_awareness_evaluation.npz'), **results)
    
    print(f"✓ State awareness evaluation completed. Results saved to {save_dir}")
    return results

def run_comprehensive_evaluation(inference, save_dir, n_samples=100):
    """
    Run all evaluations and generate a comprehensive report.
    """
    print("=== Running Comprehensive Evaluation ===")
    
    all_results = {}
    
    # Run all evaluations
    all_results['spatial'] = run_spatial_generalization_evaluation(inference, save_dir, n_samples)
    all_results['hardware'] = run_hardware_awareness_evaluation(inference, save_dir, n_samples)
    all_results['state'] = run_state_awareness_evaluation(inference, save_dir, n_samples)
    
    # Generate summary report
    report = {
        'timestamp': datetime.now().isoformat(),
        'model_path': inference.model_path,
        'n_samples': n_samples,
        'evaluations_completed': list(all_results.keys()),
        'summary': {
            'spatial_evaluation': 'Completed spatial generalization test across grid locations',
            'hardware_evaluation': 'Completed hardware awareness test across antenna configurations',
            'state_evaluation': 'Completed state awareness test across different heights'
        }
    }
    
    # Save report
    with open(os.path.join(save_dir, 'evaluation_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✓ Comprehensive evaluation completed. Report saved to {save_dir}/evaluation_report.json")
    return all_results

def main():
    parser = argparse.ArgumentParser(description='Run ESH-cDDIM evaluation')
    parser.add_argument('--mode', type=str, default='all', 
                       choices=['all', 'spatial', 'hardware', 'state'],
                       help='Evaluation mode to run')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model checkpoint')
    parser.add_argument('--save_dir', type=str, default='./results/evaluation',
                       help='Directory to save evaluation results')
    parser.add_argument('--n_samples', type=int, default=100,
                       help='Number of samples to generate for evaluation')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Initialize inference
    try:
        inference = ESHcDDIMInference(args.model_path)
        print(f"✓ Model loaded successfully from {args.model_path}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return
    
    # Run evaluation based on mode
    if args.mode == 'all':
        run_comprehensive_evaluation(inference, args.save_dir, args.n_samples)
    elif args.mode == 'spatial':
        run_spatial_generalization_evaluation(inference, args.save_dir, args.n_samples)
    elif args.mode == 'hardware':
        run_hardware_awareness_evaluation(inference, args.save_dir, args.n_samples)
    elif args.mode == 'state':
        run_state_awareness_evaluation(inference, args.save_dir, args.n_samples)
    
    print("✓ Evaluation completed successfully!")

if __name__ == "__main__":
    main()
