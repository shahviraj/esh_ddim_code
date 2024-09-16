# cDDIM

This repository contains the implementation for the paper **Generating High Dimensional User-Specific Wireless Channels using Diffusion Models** ([https://www.arxiv.org/abs/2409.03924](https://www.arxiv.org/abs/2409.03924)).

## Setup

First, execute `script_channel_ddim.py` to train the model:

```bash
python script_channel_ddim.py
```

For inference, use the following commands:

- `python ddim_inference.py generate` to generate channel matrices.
- `python ddim_inference.py concatenate` to concatenate the generated matrices.

## References

This repository was inspired by the following papers and codebases:

- The codebase is mainly based on **conditional MNIST**: [https://github.com/cloneofsimo/minDiffusion](https://github.com/cloneofsimo/minDiffusion)
- The conditioning idea is taken from **Classifier-Free Diffusion Guidance**: [https://arxiv.org/abs/2207.12598](https://arxiv.org/abs/2207.12598)
