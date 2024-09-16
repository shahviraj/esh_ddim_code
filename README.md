# cDDIM

This repository contains the implementation for the paper **Generating High Dimensional User-Specific Wireless Channels using Diffusion Models** ([https://www.arxiv.org/abs/2409.03924](https://www.arxiv.org/abs/2409.03924)).

## Setup

First, execute `script_channel_ddim.py` to train the model:

```bash
python script_channel_ddim.py
```

For inference, use the following commands:

```bash
python ddim_inference.py generate
```
to generate channel matrices.Then,
```bash
python ddim_inference.py concatenate
```
to concatenate the generated matrices.

## References

This repository was inspired by the following codebases:

- The codebase is mainly based on **conditional MNIST**: [https://github.com/cloneofsimo/minDiffusion](https://github.com/cloneofsimo/minDiffusion)

Other ideas are referenced in the ([paper](https://www.arxiv.org/abs/2409.03924)).

If you want to cite this work, please use the following citation:
```bash
@article{lee2024generating,
  title={Generating High Dimensional User-Specific Wireless Channels using Diffusion Models},
  author={Lee, Taekyun and Park, Juseong and Kim, Hyeji and Andrews, Jeffrey G},
  journal={arXiv preprint arXiv:2409.03924},
  year={2024}
}
```
