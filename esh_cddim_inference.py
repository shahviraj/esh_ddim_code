''' 
This script performs inference using the ESH-cDDIM model for generating 
Environment, State, and Hardware-conditioned wireless channels.

This code is adapted from the original ddim_inference.py to work with the 
ESH-cDDIM model trained using esh_cddim_script.py.

Usage:
- python esh_cddim_inference.py generate for generating synthetic channels
- python esh_cddim_inference.py evaluate for evaluating generated channels
- python esh_cddim_inference.py visualize for visualizing channel characteristics

'''

from typing import Dict, Tuple, List
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from torchvision.transforms import ToTensor
import scipy
import matplotlib.cm as cm
from PIL import Image, ImageDraw, ImageFont
import time
from scipy.io import savemat, loadmat
import sys
import os
from sklearn.metrics import mean_squared_error
from scipy.stats import wasserstein_distance
import seaborn as sns

# Import the model classes from the training script
from esh_cddim_script import (
    ResidualConvBlock, UnetDown, UnetUp, EmbedFC, SimpleContextProcessor, 
    ContextUnet, ddim_schedules, DDIM, BerUMaLDataset
)

class ESHcDDIMInference:
    """
    Inference class for the ESH-cDDIM model that handles channel generation,
    evaluation, and visualization.
    """
    
    def __init__(self, model_path: str, device: str = "cuda:0" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the ESH-cDDIM inference class.
        
        Parameters:
        - model_path: Path to the trained model checkpoint
        - device: Device to run inference on
        """
        self.device = device
        self.model_path = model_path
        
        # Model parameters (should match training configuration)
        self.n_feat = 256
        self.n_classes = 9  # [x, y, z, bs_ant_h, bs_ant_v, ue_ant_h, ue_ant_v, bs_spacing, ue_spacing]
        self.n_T = 256
        self.betas = (1e-4, 0.02)
        
        # Initialize model
        self.model = ContextUnet(
            in_channels=2, 
            n_feat=self.n_feat, 
            n_classes=self.n_classes,
            use_variable_context=True
        )
        
        self.ddim = DDIM(
            nn_model=self.model,
            betas=self.betas,
            n_T=self.n_T,
            device=self.device,
            drop_prob=0.1
        )
        
        # Load trained model
        self.load_model()
        
    def load_model(self):
        """Load the trained model weights."""
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.ddim.load_state_dict(checkpoint)
            self.ddim.to(self.device)
            self.ddim.eval()
            print(f"✓ Model loaded successfully from {self.model_path}")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise
    
    def generate_channels(self, 
                         context_vectors: torch.Tensor, 
                         n_samples: int = 100,
                         guidance_scale: float = 0.0,
                         save_path: str = None) -> torch.Tensor:
        """
        Generate synthetic channel matrices given context vectors.
        
        Parameters:
        - context_vectors: Tensor of shape (n_samples, 9) containing context information
        - n_samples: Number of samples to generate
        - guidance_scale: Guidance scale for classifier-free guidance
        - save_path: Optional path to save generated channels
        
        Returns:
        - Generated channel matrices of shape (n_samples, 2, height, width)
        """
        print(f"Generating {n_samples} channels with guidance scale {guidance_scale}")
        
        # Ensure context vectors are on the correct device and have correct shape
        if context_vectors.shape[1] != 9:
            raise ValueError(f"Context vectors must have 9 dimensions, got {context_vectors.shape[1]}")
        
        context_vectors = context_vectors.to(dtype=torch.float32).to(self.device)
        
        # Generate channels
        with torch.no_grad():
            generated_channels, _ = self.ddim.sample(
                n_sample=n_samples,
                c_test=context_vectors,
                size=(2, 4, 32),  # Channel dimensions (real, imag, height, width)
                device=self.device,
                guide_w=guidance_scale
            )
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            savemat(save_path, {'H_generated': generated_channels.cpu().numpy()})
            np.save(save_path.replace('.mat', '.npy'), generated_channels.cpu().numpy())
            print(f"✓ Generated channels saved to {save_path}")
        
        return generated_channels
    
    def evaluate_channels(self, 
                         generated_channels: torch.Tensor,
                         reference_channels: torch.Tensor,
                         context_vectors: torch.Tensor = None) -> Dict[str, float]:
        """
        Evaluate the quality of generated channels against reference channels.
        
        Parameters:
        - generated_channels: Generated channel matrices
        - reference_channels: Reference/ground truth channel matrices
        - context_vectors: Context vectors used for generation
        
        Returns:
        - Dictionary of evaluation metrics
        """
        print("Evaluating generated channels...")
        
        # Convert to numpy for easier computation
        gen_np = generated_channels.cpu().numpy()
        ref_np = reference_channels.cpu().numpy()
        
        metrics = {}
        
        # 1. Mean Squared Error (MSE)
        mse = mean_squared_error(ref_np.flatten(), gen_np.flatten())
        metrics['MSE'] = mse
        
        # 2. Normalized MSE (NMSE)
        ref_power = np.mean(np.sum(np.abs(ref_np)**2, axis=(1, 2, 3)))
        gen_power = np.mean(np.sum(np.abs(gen_np)**2, axis=(1, 2, 3)))
        nmse = mse / ref_power
        metrics['NMSE'] = nmse
        
        # 3. Channel capacity comparison
        def calculate_capacity(channels):
            # Calculate MIMO capacity for each channel
            capacities = []
            for i in range(channels.shape[0]):
                H = channels[i, 0] + 1j * channels[i, 1]  # Convert to complex
                # SVD for capacity calculation
                U, S, Vh = np.linalg.svd(H)
                # Water-filling capacity (assuming equal power allocation for simplicity)
                capacity = np.sum(np.log2(1 + S**2))
                capacities.append(capacity)
            return np.array(capacities)
        
        ref_capacity = calculate_capacity(ref_np)
        gen_capacity = calculate_capacity(gen_np)
        
        metrics['Capacity_MSE'] = mean_squared_error(ref_capacity, gen_capacity)
        metrics['Capacity_Correlation'] = np.corrcoef(ref_capacity, gen_capacity)[0, 1]
        
        # 4. Channel rank comparison
        def calculate_rank(channels):
            ranks = []
            for i in range(channels.shape[0]):
                H = channels[i, 0] + 1j * channels[i, 1]
                rank = np.linalg.matrix_rank(H)
                ranks.append(rank)
            return np.array(ranks)
        
        ref_rank = calculate_rank(ref_np)
        gen_rank = calculate_rank(gen_np)
        
        metrics['Rank_MSE'] = mean_squared_error(ref_rank, gen_rank)
        metrics['Rank_Correlation'] = np.corrcoef(ref_rank, gen_rank)[0, 1]
        
        # 5. Statistical properties
        metrics['Mean_Magnitude_Error'] = np.mean(np.abs(np.abs(ref_np) - np.abs(gen_np)))
        metrics['Std_Magnitude_Error'] = np.std(np.abs(np.abs(ref_np) - np.abs(gen_np)))
        
        print("Evaluation Results:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.6f}")
        
        return metrics
    
    def visualize_channels(self, 
                          channels: torch.Tensor,
                          context_vectors: torch.Tensor = None,
                          save_path: str = None,
                          n_samples: int = 4):
        """
        Visualize generated channel matrices.
        
        Parameters:
        - channels: Channel matrices to visualize
        - context_vectors: Context vectors used for generation
        - save_path: Path to save visualization
        - n_samples: Number of samples to visualize
        """
        print(f"Visualizing {min(n_samples, channels.shape[0])} channel samples...")
        
        channels_np = channels.cpu().numpy()
        n_show = min(n_samples, channels.shape[0])
        
        fig, axes = plt.subplots(2, n_show, figsize=(4*n_show, 8))
        if n_show == 1:
            axes = axes.reshape(2, 1)
        
        for i in range(n_show):
            # Real part
            im1 = axes[0, i].imshow(channels_np[i, 0], cmap='viridis', aspect='auto')
            axes[0, i].set_title(f'Real Part - Sample {i+1}')
            axes[0, i].set_xlabel('BS Antennas')
            axes[0, i].set_ylabel('UE Antennas')
            plt.colorbar(im1, ax=axes[0, i])
            
            # Imaginary part
            im2 = axes[1, i].imshow(channels_np[i, 1], cmap='viridis', aspect='auto')
            axes[1, i].set_title(f'Imaginary Part - Sample {i+1}')
            axes[1, i].set_xlabel('BS Antennas')
            axes[1, i].set_ylabel('UE Antennas')
            plt.colorbar(im2, ax=axes[1, i])
            
            # Add context information if available
            if context_vectors is not None:
                ctx = context_vectors[i].cpu().numpy()
                context_text = f"Loc: ({ctx[0]:.2f}, {ctx[1]:.2f}, {ctx[2]:.2f})\n"
                context_text += f"BS: {int(ctx[3])}x{int(ctx[4])}, UE: {int(ctx[5])}x{int(ctx[6])}\n"
                context_text += f"Spacing: BS={ctx[7]:.2f}, UE={ctx[8]:.2f}"
                axes[0, i].text(0.02, 0.98, context_text, transform=axes[0, i].transAxes,
                               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Visualization saved to {save_path}")
        
        plt.show()
    
    def analyze_hardware_awareness(self, 
                                  base_context: torch.Tensor,
                                  hardware_variations: List[Dict],
                                  save_path: str = None):
        """
        Analyze how the model responds to different hardware configurations.
        
        Parameters:
        - base_context: Base context vector (location fixed)
        - hardware_variations: List of hardware parameter variations
        - save_path: Path to save analysis results
        """
        print("Analyzing hardware awareness...")
        
        results = {}
        
        for i, hw_config in enumerate(hardware_variations):
            # Create context with modified hardware parameters
            context = base_context.clone()
            context[0, 3] = hw_config['bs_ant_h']  # BS horizontal antennas
            context[0, 4] = hw_config['bs_ant_v']  # BS vertical antennas
            context[0, 5] = hw_config['ue_ant_h']  # UE horizontal antennas
            context[0, 6] = hw_config['ue_ant_v']  # UE vertical antennas
            context[0, 7] = hw_config['bs_spacing']  # BS spacing
            context[0, 8] = hw_config['ue_spacing']  # UE spacing
            
            # Generate channels
            generated = self.generate_channels(context, n_samples=1, guidance_scale=0.0)
            
            # Analyze channel properties
            H = generated[0, 0].cpu().numpy() + 1j * generated[0, 1].cpu().numpy()
            U, S, Vh = np.linalg.svd(H)
            
            results[f'config_{i}'] = {
                'hardware': hw_config,
                'channel_rank': np.linalg.matrix_rank(H),
                'condition_number': np.max(S) / np.min(S[S > 1e-10]),
                'frobenius_norm': np.linalg.norm(H, 'fro'),
                'singular_values': S
            }
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Channel rank vs hardware configuration
        configs = list(results.keys())
        ranks = [results[cfg]['channel_rank'] for cfg in configs]
        axes[0, 0].bar(configs, ranks)
        axes[0, 0].set_title('Channel Rank vs Hardware Configuration')
        axes[0, 0].set_ylabel('Channel Rank')
        
        # Condition number vs hardware configuration
        cond_nums = [results[cfg]['condition_number'] for cfg in configs]
        axes[0, 1].bar(configs, cond_nums)
        axes[0, 1].set_title('Condition Number vs Hardware Configuration')
        axes[0, 1].set_ylabel('Condition Number')
        
        # Frobenius norm vs hardware configuration
        fro_norms = [results[cfg]['frobenius_norm'] for cfg in configs]
        axes[1, 0].bar(configs, fro_norms)
        axes[1, 0].set_title('Frobenius Norm vs Hardware Configuration')
        axes[1, 0].set_ylabel('Frobenius Norm')
        
        # Singular value distributions
        for i, cfg in enumerate(configs):
            axes[1, 1].plot(results[cfg]['singular_values'], label=f'Config {i}', marker='o')
        axes[1, 1].set_title('Singular Value Distributions')
        axes[1, 1].set_xlabel('Singular Value Index')
        axes[1, 1].set_ylabel('Singular Value')
        axes[1, 1].legend()
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Hardware awareness analysis saved to {save_path}")
        
        plt.show()
        
        return results

def create_test_context_vectors(n_samples: int = 100, 
                               device: str = "cuda:0" if torch.cuda.is_available() else "cpu") -> torch.Tensor:
    """
    Create test context vectors for inference.
    
    Parameters:
    - n_samples: Number of context vectors to create
    - device: Device to create tensors on
    
    Returns:
    - Context vectors of shape (n_samples, 9)
    """
    # Create diverse context vectors
    context_vectors = torch.zeros(n_samples, 9, device=device)
    
    # Location coordinates (x, y, z) - random within reasonable range
    context_vectors[:, 0] = torch.randn(n_samples, device=device) * 10  # x
    context_vectors[:, 1] = torch.randn(n_samples, device=device) * 10  # y
    context_vectors[:, 2] = torch.rand(n_samples, device=device) * 2 + 1  # z (height)
    
    # BS antenna configuration
    context_vectors[:, 3] = torch.randint(4, 17, (n_samples,), device=device).float()  # bs_ant_h
    context_vectors[:, 4] = torch.randint(2, 9, (n_samples,), device=device).float()   # bs_ant_v
    
    # UE antenna configuration
    context_vectors[:, 5] = torch.randint(1, 5, (n_samples,), device=device).float()   # ue_ant_h
    context_vectors[:, 6] = torch.randint(1, 5, (n_samples,), device=device).float()   # ue_ant_v
    
    # Antenna spacing
    context_vectors[:, 7] = torch.rand(n_samples, device=device) * 0.5 + 0.5  # bs_spacing
    context_vectors[:, 8] = torch.rand(n_samples, device=device) * 0.5 + 0.5  # ue_spacing
    
    return context_vectors

def load_reference_data(dataset_path: str = "DeepMIMO_dataset", 
                       n_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load reference data for evaluation.
    
    Parameters:
    - dataset_path: Path to the DeepMIMO dataset
    - n_samples: Number of samples to load
    
    Returns:
    - Tuple of (channels, context_vectors)
    """
    print(f"Loading reference data from {dataset_path}...")
    
    # Load test dataset
    test_dataset = BerUMaLDataset(dataset_path, 0, n_samples, use_deepmimo=True)
    dataloader = DataLoader(test_dataset, batch_size=n_samples, shuffle=False)
    
    # Get first batch
    channels, context_vectors = next(iter(dataloader))
    
    return channels, context_vectors

def main():
    """
    Main function for ESH-cDDIM inference.
    """
    if len(sys.argv) < 2:
        print("Usage: python esh_cddim_inference.py <mode> [options]")
        print("Modes:")
        print("  generate - Generate synthetic channels")
        print("  evaluate - Evaluate generated channels against reference")
        print("  visualize - Visualize channel characteristics")
        print("  hardware - Analyze hardware awareness")
        sys.exit(1)
    
    mode = sys.argv[1]
    
    # Configuration
    model_path = "./data/cDDIM_10/model.pth"  # Update this path
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    n_samples = 50
    save_dir = "./results/esh_cddim_inference"
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize inference class
    try:
        inference = ESHcDDIMInference(model_path, device)
    except Exception as e:
        print(f"Failed to initialize inference: {e}")
        return
    
    if mode == "generate":
        print("=== Generating Synthetic Channels ===")
        
        # Create test context vectors
        context_vectors = create_test_context_vectors(n_samples, device)
        
        # Generate channels
        generated_channels = inference.generate_channels(
            context_vectors=context_vectors,
            n_samples=n_samples,
            guidance_scale=0.0,
            save_path=os.path.join(save_dir, "generated_channels.mat")
        )
        
        # Visualize some samples
        inference.visualize_channels(
            channels=generated_channels[:4],
            context_vectors=context_vectors[:4],
            save_path=os.path.join(save_dir, "generated_channels_visualization.png")
        )
        
    elif mode == "evaluate":
        print("=== Evaluating Generated Channels ===")
        
        # Load reference data
        try:
            ref_channels, ref_context = load_reference_data(n_samples=n_samples)
        except Exception as e:
            print(f"Failed to load reference data: {e}")
            return
        
        # Generate channels with same context
        generated_channels = inference.generate_channels(
            context_vectors=ref_context,
            n_samples=n_samples,
            guidance_scale=0.0
        )
        
        # Evaluate
        metrics = inference.evaluate_channels(
            generated_channels=generated_channels,
            reference_channels=ref_channels,
            context_vectors=ref_context
        )
        
        # Save evaluation results
        with open(os.path.join(save_dir, "evaluation_metrics.txt"), "w") as f:
            for metric, value in metrics.items():
                f.write(f"{metric}: {value:.6f}\n")
        
    elif mode == "visualize":
        print("=== Visualizing Channel Characteristics ===")
        
        # Create diverse context vectors
        context_vectors = create_test_context_vectors(n_samples, device)
        
        # Generate channels
        generated_channels = inference.generate_channels(
            context_vectors=context_vectors,
            n_samples=n_samples,
            guidance_scale=0.0
        )
        
        # Visualize
        inference.visualize_channels(
            channels=generated_channels,
            context_vectors=context_vectors,
            save_path=os.path.join(save_dir, "channel_visualization.png")
        )
        
    elif mode == "hardware":
        print("=== Analyzing Hardware Awareness ===")
        
        # Create base context
        base_context = create_test_context_vectors(1, device)
        
        # Define hardware variations
        hardware_variations = [
            {'bs_ant_h': 8, 'bs_ant_v': 4, 'ue_ant_h': 2, 'ue_ant_v': 2, 'bs_spacing': 0.5, 'ue_spacing': 0.5},
            {'bs_ant_h': 16, 'bs_ant_v': 8, 'ue_ant_h': 2, 'ue_ant_v': 2, 'bs_spacing': 0.5, 'ue_spacing': 0.5},
            {'bs_ant_h': 32, 'bs_ant_v': 16, 'ue_ant_h': 2, 'ue_ant_v': 2, 'bs_spacing': 0.5, 'ue_spacing': 0.5},
            {'bs_ant_h': 8, 'bs_ant_v': 4, 'ue_ant_h': 4, 'ue_ant_v': 4, 'bs_spacing': 0.5, 'ue_spacing': 0.5},
            {'bs_ant_h': 8, 'bs_ant_v': 4, 'ue_ant_h': 2, 'ue_ant_v': 2, 'bs_spacing': 0.7, 'ue_spacing': 0.7},
        ]
        
        # Analyze hardware awareness
        results = inference.analyze_hardware_awareness(
            base_context=base_context,
            hardware_variations=hardware_variations,
            save_path=os.path.join(save_dir, "hardware_awareness_analysis.png")
        )
        
        # Save results
        np.save(os.path.join(save_dir, "hardware_analysis_results.npy"), results)
        
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)
    
    print("✓ Inference completed successfully!")

if __name__ == "__main__":
    main()
