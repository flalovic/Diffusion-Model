# Diffusion Model

A PyTorch implementation of a denoising diffusion probabilistic model (DDPM) for generating 64 x 64 natural landscape images. The forward process gradually adds Gaussian noise over 1,000 steps using a linear beta schedule, while the learned reverse process reconstructs an image by predicting and removing noise at each step.

## Architecture

The noise-prediction network is a three-level U-Net with convolutional encoder and decoder blocks, skip connections, and a 512-channel bottleneck. Sinusoidal 256-dimensional timestep embeddings condition every downsampling and upsampling stage. Self-attention layers are applied throughout the network to capture long-range spatial relationships, and the model is trained with mean squared error between the added noise and predicted noise.

The training pipeline, model definition, sampling procedure, and visualizations are contained in [Diffusion.ipynb](Diffusion.ipynb). Training uses the [Landscape Pictures](https://www.kaggle.com/datasets/arnaud58/landscape-pictures) dataset.

## Setup

```bash
pip install torch torchvision matplotlib tqdm jupyter
jupyter notebook Diffusion.ipynb
```

Extract the dataset into `nature/ls/` before training. A CUDA-capable GPU is recommended.

## Generated Landscapes

<p align="center">
	<img src="images/exampl1.png" alt="First grid of generated landscape images" width="650">
</p>

<p align="center">
	<img src="images/exampl2.png" alt="Second grid of generated landscape images" width="650">
</p>