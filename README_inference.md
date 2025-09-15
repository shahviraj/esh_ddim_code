# ESH-cDDIM Inference and Evaluation

This directory contains scripts for performing inference and evaluation with the ESH-cDDIM (Environment, State, and Hardware-conditioned Diffusion Model) for wireless channel generation.

## Files Overview

- `esh_cddim_inference.py`: Main inference script with comprehensive evaluation capabilities
- `run_evaluation.py`: Automated evaluation script for running experiments
- `esh_cddim_script.py`: Training script (original)
- `load_deepmimo_datasets.py`: Dataset loader for DeepMIMO data

## Quick Start

### 1. Basic Inference

Generate synthetic channels using the trained model:

```bash
python esh_cddim_inference.py generate
```

### 2. Evaluate Against Reference Data

Compare generated channels with reference data:

```bash
python esh_cddim_inference.py evaluate
```

### 3. Visualize Channel Characteristics

Generate visualizations of channel matrices:

```bash
python esh_cddim_inference.py visualize
```

### 4. Analyze Hardware Awareness

Test how the model responds to different hardware configurations:

```bash
python esh_cddim_inference.py hardware
```

## Comprehensive Evaluation

For detailed experiments and paper results, use the evaluation script:

```bash
# Run all evaluations
python run_evaluation.py --mode all --model_path ./data/cDDIM_10/model.pth

# Run specific evaluations
python run_evaluation.py --mode spatial --model_path ./data/cDDIM_10/model.pth
python run_evaluation.py --mode hardware --model_path ./data/cDDIM_10/model.pth
python run_evaluation.py --mode state --model_path ./data/cDDIM_10/model.pth
```

## Model Architecture

The ESH-cDDIM model extends the original cDDIM with:

- **Variable Context Processing**: Handles mixed-type context vectors
- **9-Dimensional Context**: `[x, y, z, bs_ant_h, bs_ant_v, ue_ant_h, ue_ant_v, bs_spacing, ue_spacing]`
- **Hardware Awareness**: Responds to different antenna configurations
- **State Awareness**: Considers propagation conditions (simulated via height variations)

## Context Vector Format

The model expects context vectors with 9 dimensions:

1. **Location (3D)**: `[x, y, z]` - User coordinates
2. **BS Antenna Config (2D)**: `[bs_ant_h, bs_ant_v]` - Base station antenna array dimensions
3. **UE Antenna Config (2D)**: `[ue_ant_h, ue_ant_v]` - User equipment antenna array dimensions
4. **Antenna Spacing (2D)**: `[bs_spacing, ue_spacing]` - Inter-antenna spacing in wavelengths

## Evaluation Metrics

The evaluation includes:

- **Spatial Generalization**: Channel generation across different locations
- **Hardware Awareness**: Response to different antenna configurations
- **State Awareness**: Sensitivity to propagation conditions
- **Statistical Metrics**: MSE, NMSE, channel capacity, rank analysis

## Output Files

Results are saved in the `./results/` directory:

- `generated_channels.mat/.npy`: Generated channel matrices
- `evaluation_metrics.txt`: Quantitative evaluation results
- `*_visualization.png`: Channel visualizations
- `*_evaluation.npz`: Detailed evaluation data
- `evaluation_report.json`: Comprehensive evaluation report

## Requirements

- PyTorch
- NumPy
- SciPy
- Matplotlib
- Seaborn
- scikit-learn
- DeepMIMO dataset (for reference evaluation)

## Usage Examples

### Generate Channels for Specific Hardware Configuration

```python
from esh_cddim_inference import ESHcDDIMInference, create_test_context_vectors

# Initialize inference
inference = ESHcDDIMInference('./data/cDDIM_10/model.pth')

# Create custom context vector
context = torch.zeros(1, 9)
context[0, 0] = 5.0    # x coordinate
context[0, 1] = 3.0    # y coordinate  
context[0, 2] = 2.0    # z coordinate
context[0, 3] = 16     # BS horizontal antennas
context[0, 4] = 8      # BS vertical antennas
context[0, 5] = 2      # UE horizontal antennas
context[0, 6] = 2      # UE vertical antennas
context[0, 7] = 0.5    # BS spacing
context[0, 8] = 0.5    # UE spacing

# Generate channel
generated = inference.generate_channels(context, n_samples=1)
```

### Evaluate Model Performance

```python
# Load reference data
ref_channels, ref_context = load_reference_data(n_samples=100)

# Generate channels
generated = inference.generate_channels(ref_context, n_samples=100)

# Evaluate
metrics = inference.evaluate_channels(generated, ref_channels, ref_context)
print(metrics)
```

## Troubleshooting

1. **Model Loading Error**: Ensure the model path is correct and the model was trained with the same architecture
2. **CUDA Memory Error**: Reduce batch size or use CPU inference
3. **Dataset Loading Error**: Ensure DeepMIMO dataset is in the correct directory structure
4. **Context Dimension Error**: Ensure context vectors have exactly 9 dimensions

## Citation

If you use this code in your research, please cite:

```bibtex
@article{esh_cddim_2024,
  title={Environment, State, and Hardware-Conditioned Diffusion Models for Wireless Channel Generation},
  author={Your Name},
  journal={Agents4Science Conference},
  year={2024}
}
```
