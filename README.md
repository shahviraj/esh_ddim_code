# H-cDDIM: Hardware-Conditioned Generative Channel Modeling

This repository contains the official PyTorch implementation for the paper: "Hardware-Conditioned Generative Channel Modeling: A Diffusion-Based Approach for Location and Hardware-Aware Wireless Dataset Synthesis".

Our work introduces H-cDDIM, a novel conditional diffusion model that generates high-fidelity wireless channel data conditioned not only on user location but also on detailed hardware parameters, including antenna array geometry and inter-element spacing.

## Repository Structure

- `esh_cddim_script.py`: Main script for training the H-cDDIM model.
- `cddim_comparison_train.py`: Script for training the baseline location-only cDDIM model.
- `run_evaluation.py`: Script to run evaluations on a trained H-cDDIM model (statistical fidelity, spatial generalization, etc.).
- `run_comparison.py`: Script to run direct quantitative and qualitative comparisons between H-cDDIM and the baseline cDDIM.
- `visualize_samples.py`: Script to generate side-by-side visualizations of ground truth, cDDIM, and H-cDDIM channel samples.
- `esh_cddim_inference.py`: Contains the core inference logic for H-cDDIM.
- `load_deepmimo_datasets.py`: Helper script to load and process the custom DeepMIMO datasets.
- `environment.yml`: Conda environment file with all necessary dependencies.

## Setup

To set up the environment and install the required dependencies, please use the provided Conda environment file:

```bash
# 1. Clone the repository
git clone https://github.com/your-repo/H-cDDIM.git
cd H-cDDIM

# 2. Create and activate the Conda environment
conda env create -f environment.yml
conda activate cDDIM
```

## Reproducing the Experiments

The experimental pipeline consists of three main stages: Dataset Generation, Model Training, and Evaluation.

### 1. Dataset Generation

The channel datasets used in this work were generated using the [DeepMIMO](https://www.deepmimo.net/) framework, which requires MATLAB.

1.  **Download DeepMIMO:** Obtain the DeepMIMO generator code and the 'O1' outdoor scenario ray-tracing data.
2.  **Generate Datasets:** Use the DeepMIMO MATLAB scripts to generate multiple `.mat` dataset files. Our paper uses 16 distinct hardware configurations by systematically varying the following parameters:
    - **Base Station Arrays:** 4×8, 8×4, 16×2, 32×1
    - **User Equipment Arrays:** 2×2, 4×1
    - **Inter-Antenna Spacing:** 0.4λ, 0.5λ
3.  **Organize Files:** Place all generated `.mat` files into a single directory (e.g., `../datasets/DeepMIMO_dataset_full/`). The data loading scripts will automatically parse the configuration from the filenames.

*Note: Due to the size of the dataset, we recommend generating it locally. We plan to release the full dataset used in the paper upon publication.*

### 2. Model Training

We provide separate scripts for training our proposed H-cDDIM model and the baseline cDDIM.

**Training H-cDDIM (Ours):**

```bash
python esh_cddim_script.py
```
This script will use the dataset located at `../../datasets/DeepMIMO_dataset_full` by default and save model checkpoints to `./data/batch_job_cDDIM_{num_samples}/`. Please adjust the paths inside the script as needed.

**Training the baseline cDDIM:**

```bash
python cddim_comparison_train.py
```
This script also uses the same dataset path and saves checkpoints to `./data/cDDIM_original_comparison_{num_samples}/`.

*We will provide pre-trained model weights for both H-cDDIM and the baseline cDDIM to allow for direct evaluation.*

### 3. Evaluation and Comparison

Once the models are trained, you can reproduce the key results from the paper.

**Quantitative Comparison (Fidelity Test):**
This test generates the quantitative results table and the distribution plots.

```bash
python run_comparison.py \
    --esh_model_path /path/to/your/H-cDDIM_model.pth \
    --cddim_model_path /path/to/your/cDDIM_baseline_model.pth \
    --dataset_path /path/to/your/DeepMIMO_dataset \
    --mode fidelity
```

**Qualitative Comparison (Singular Value Analysis):**
This test generates the plot comparing the averaged singular value distributions.

```bash
python run_comparison.py \
    --esh_model_path /path/to/your/H-cDDIM_model.pth \
    --cddim_model_path /path/to/your/cDDIM_baseline_model.pth \
    --dataset_path /path/to/your/DeepMIMO_dataset \
    --mode cross_hardware
```

**Side-by-Side Visualization:**
This generates the 8x3 figure comparing channel magnitudes.

```bash
python visualize_samples.py \
    --esh_model_path /path/to/your/H-cDDIM_model.pth \
    --cddim_model_path /path/to/your/cDDIM_baseline_model.pth \
    --dataset_path /path/to/your/DeepMIMO_dataset
```

## Citation

If you find this work useful in your research, please consider citing our paper:

```bibtex
@article{YourName2025HcDDIM,
  title={Hardware-Conditioned Generative Channel Modeling: A Diffusion-Based Approach for Location and Hardware-Aware Wireless Dataset Synthesis},
  author={Your Name and Your Advisor},
  journal={Conference on AI for Science},
  year={2025}
}
```
