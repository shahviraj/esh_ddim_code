# cDDIM (Conditional Denoising Diffusion Implicit Model for wireless channel matrix)

This repository contains the implementation for the paper **Generating High Dimensional User-Specific Wireless Channels using Diffusion Models** ([https://www.arxiv.org/abs/2409.03924](https://www.arxiv.org/abs/2409.03924)).

## Generating Initial Channel Dataset from QuaDRiGa

After installing QuaDRiGa ([https://quadriga-channel-model.de/software/](https://quadriga-channel-model.de/software/)), 
place `main_chgen.m` in the `/quadriga_src/` folder.

Then, execute the file in MATLAB. After generation, place the output files into `/data/QuaDRiGa`. Alternatively, you can download the dataset from Google Drive [here](https://drive.google.com/file/d/17ho6jTsPh6HD4IkkYSlB9WM9JXF4xwII/view?usp=drive_link).

## Conda Environment Setup

To create and activate the Conda environment using the provided `environment.yml`, follow these steps:

1. **Create the environment**:

   ```bash
   conda env create -f environment.yml
   ```
   
2. **Activate the environment**:

   ```bash
   conda activate cDDIM
   ```
   
## Training and Inference

First, create `/cDDIM_10000/` folder, and execute `script_channel_ddim.py` to train the model:

```bash
python script_channel_ddim.py
```

For inference, use the following commands:

```bash
python ddim_inference.py generate
```
to generate channel matrices. Then,
```bash
python ddim_inference.py concatenate
```
to concatenate the generated matrices. 
The above description is for the quadriga dataset. A version for the DeepMIMO dataset will be updated.

## References

This repository was inspired by the following codebases:

- The codebase is primarily based on **conditional MNIST**: [https://github.com/cloneofsimo/minDiffusion](https://github.com/cloneofsimo/minDiffusion)
  
Two downstream tasks mentioned in the paper:

- Channel compression - CRNet: [https://github.com/Kylin9511/CRNet](https://github.com/Kylin9511/CRNet)
- Site-specific beamforming - DLGF: [https://github.com/YuqiangHeng/DLGF](https://github.com/YuqiangHeng/DLGF)

Other ideas are referenced in the [paper](https://www.arxiv.org/abs/2409.03924).

If you want to cite this work, please use the following citation:
```bash
@article{lee2024generating,
  title={Generating High Dimensional User-Specific Wireless Channels using Diffusion Models},
  author={Lee, Taekyun and Park, Juseong and Kim, Hyeji and Andrews, Jeffrey G},
  journal={arXiv preprint arXiv:2409.03924},
  year={2024}
}
```
