''' 
This script does conditional image generation on MNIST, using a diffusion model

This code is modified from,
https://github.com/cloneofsimo/minDiffusion

Diffusion model is based on DDPM,
https://arxiv.org/abs/2006.11239
dfo

This technique also features in ImageGen 'Photorealistic Text-to-Image Diffusion Modelswith Deep Language Understanding',
https://arxiv.org/abs/2205.11487

- python ddim_inference.py generate for generating the synthetic channel
- python ddim_inference.py generate for concatenating the generated synthetic channel

'''
from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.datasets import MNIST
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

class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        '''
        standard ResNet style convolutional block
        '''
        self.same_channels = in_channels==out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # this adds on correct residual in case channels have increased
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2 
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        '''
        process and downscale the image feature maps
        '''
        layers = [ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        '''
        process and upscale the image feature maps
        '''
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)


class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat = 256, n_classes=3):
        super(ContextUnet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)

        # self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU()) # XXX
        # Adjust AvgPool2d to match input feature map size
        self.to_vec = nn.Sequential(nn.AvgPool2d((1, 8)), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2*n_feat)
        self.timeembed2 = EmbedFC(1, 1*n_feat)
        self.contextembed1 = EmbedFC(n_classes, 2*n_feat)
        self.contextembed2 = EmbedFC(n_classes, 1*n_feat)

        # Embedding for the 3D coordinates
        self.coord_dim = 3
        self.coord_embed = nn.Linear(self.coord_dim, n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, kernel_size=(1, 8), stride=(1, 8)),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, c, t, context_mask):
        # x is (noisy) image, c is context label, t is timestep, 
        # context_mask says which samples to block the context on

        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)

        # # convert context to one hot embedding
        # c = nn.functional.one_hot(c, num_classes=self.n_classes).type(torch.float)
        
        # mask out context if context_mask == 1
        # context_mask = context_mask[:, None]
        # context_mask = context_mask.repeat(1,self.n_classes)
        # context_mask = (-1*(1-context_mask)) # need to flip 0 <-> 1
        # c = c * context_mask
        
        # # embed context, time step
        c = c.float()
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)


        # # could concatenate the context embedding here instead of adaGN
        # # hiddenvec = torch.cat((hiddenvec, temb1, cemb1), 1)

        # up1 = self.up0(hiddenvec)
        # # up2 = self.up1(up1, down2) # if want to avoid add and multiply embeddings
        # up2 = self.up1(cemb1*up1+ temb1, down2)  # add and multiply embeddings
        # up3 = self.up2(cemb2*up2+ temb2, down1)
        # out = self.out(torch.cat((up3, x), 1))

        # Combine embeddings with upsampling
        up1 = self.up0(hiddenvec)
        up2 = self.up1(cemb1*up1+ temb1, down2)  # add and multiply embeddings
        up3 = self.up2(cemb2*up2+ temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out


def ddim_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDIM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    # mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab
    DDIM_coeff = sqrtmab - torch.sqrt(alpha_t) * torch.sqrt(1 - alphabar_t / alpha_t) #DDIM coef.

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "DDIM_coeff": DDIM_coeff,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


class DDIM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        super(DDIM, self).__init__()
        self.nn_model = nn_model.to(device)

        # register_buffer allows accessing dictionary produced by ddim_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddim_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, c):
        """
        this method is used in training, so samples t and noise randomly
        """
        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.
        # dropout context with some probability
        context_mask = torch.bernoulli(torch.zeros_like(c)+self.drop_prob).to(self.device)
        
        # return MSE between added noise, and our predicted noise
        return self.loss_mse(noise, self.nn_model(x_t, c, _ts / self.n_T, context_mask))

    def sample(self, n_sample, c_test, size, device, guide_w = 0.0):
        # we follow the guidance sampling scheme described in 'Classifier-Free Diffusion Guidance'
        # to make the fwd passes efficient, we concat two versions of the dataset,
        # one with context_mask=0 and the other context_mask=1
        # we then mix the outputs with the guidance scale, w
        # where w>0 means more guidance

        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1), sample initial noise

        # don't drop context at test time
        context_mask = torch.zeros_like(c_test).to(device)

        # double the batch
        c_test = c_test.repeat(2, 1)
        context_mask = context_mask.repeat(2, 1)
        context_mask[n_sample:] = 1. # makes second half of batch context free
        
        x_i_store = [] # keep track of generated steps in case want to plot something 
        print()
        for i in range(self.n_T, 0, -1):
            print(f'sampling timestep {i}',end='\r')
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample,1,1,1)

            # double batch
            x_i = x_i.repeat(2,1,1,1)
            t_is = t_is.repeat(2,1,1,1)

            # DDIM step (deterministic, no random noise added)
            eps = self.nn_model(x_i, c_test, t_is, context_mask)
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1 + guide_w) * eps1 - guide_w * eps2
            x_i = x_i[:n_sample]
            x_i = self.oneover_sqrta[i] * (x_i - eps * self.DDIM_coeff[i])

            if i % 20 == 0 or i == self.n_T or i < 8:
                x_i_store.append(x_i.detach().cpu().numpy())

        x_i_store = np.array(x_i_store)
        return x_i, x_i_store

class BerUMaLDataset(Dataset):
    def __init__(self, data_path, idx_start, idx_end):
        self.data_path = data_path
        self.idx_start = idx_start
        self.idx_end = idx_end
        self.load_dataset()

    def load_dataset(self):
        contents = scipy.io.loadmat(self.data_path)
        array1 = contents['H_set'][self.idx_start:self.idx_end,:,:]
        array1 = np.stack((np.real(array1[:,:,:]), np.imag(array1[:,:,:])), axis=1)
        array2 = array1.copy()
        for i in range(self.idx_end - self.idx_start) :
            dft_data = np.fft.fft2(array2[i,0]+1j*array2[i,1])
            dft_shifted = np.fft.fftshift(dft_data)
            array1[i,0] = np.real(dft_shifted)
            array1[i,1] = np.imag(dft_shifted)

        # Load labels
        self.labels = contents['rx_positions'][self.idx_start:self.idx_end, :]

        self.data = []
        self.labels = []  # Converted to a list
        labels_array = contents['rx_positions'][self.idx_start:self.idx_end,:]  # Assuming these are the labels

        for i in range(array1.shape[0]):
            # Calculating the magnitude
            magnitude = np.sqrt(array1[i, 0, :, :]**2 + array1[i, 1, :, :]**2)

            # Finding the maximum magnitude
            max_magnitude = np.max(magnitude)

            array1[i, 0, :, :] /= max_magnitude
            array1[i, 1, :, :] /= max_magnitude

            self.data.append(array1[i, :2, :, :])  # Appending each slice
            self.labels.append(labels_array[i])  # Appending the corresponding label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = ToTensor()(self.data[idx]).float()
        label = self.labels[idx]
        return data_item, label

def tensor_to_pil(tensor, scale_factor=5, border_size=1, border_color=(0, 0, 0)):
    # Convert a tensor to a PIL Image with a colormap
    array = tensor.cpu().numpy()
    cm_hot = cm.get_cmap('viridis')
    colored_image = cm_hot(array)
    pil_image = Image.fromarray((colored_image[:, :, :3] * 255).astype('uint8'), 'RGB')
    pil_image = pil_image.resize((pil_image.width * scale_factor, pil_image.height * scale_factor), Image.Resampling.LANCZOS)

    # Create a new image with borders
    new_image = Image.new('RGB', (pil_image.width + 2 * border_size, pil_image.height + 2 * border_size), color=border_color)
    new_image.paste(pil_image, (border_size, border_size))
    return new_image

def generate_H_test(ddim, device, save_dir, start_idx=0, end_idx=20, batch_size=5000, n_sample=5000, ws_test=[0.0]):
    """
    Generate H_test matrices and save them to disk.

    Parameters:
    - ddim: The diffusion model instance.
    - device: The computation device ('cuda' or 'cpu').
    - save_dir: Directory to save the generated data.
    - start_idx: Starting index for data loading.
    - end_idx: Ending index for data loading.
    - batch_size: Batch size for data loading.
    - n_sample: Number of samples to generate.
    - ws_test: List of guidance weights.
    """
    for j in range(start_idx, end_idx):
        # Load BerUMaL dataset
        data_path = "./data/QuaDRiGa/NumUEs_100000_num_BerUMaL_ULA.mat"  # Replace with your actual path
        berumal_dataset_test = BerUMaLDataset(data_path, 5000 * j, 5000 * (j + 1))

        # Create a DataLoader for the BerUMaL dataset
        dataloader_test = DataLoader(berumal_dataset_test, batch_size=batch_size, shuffle=False)

        # Select a fixed subset of test samples and their corresponding coordinates
        fixed_test_samples, fixed_test_coords = next(iter(dataloader_test))
        fixed_test_samples = fixed_test_samples[:n_sample].to(device)
        fixed_test_coords = fixed_test_coords[:n_sample].to(device)

        ddim.eval()
        with torch.no_grad():
            for w in ws_test:
                start_time = time.time()
                x_gen, _ = ddim.sample(n_sample, fixed_test_coords, (2, 4, 32), device, guide_w=w)
                end_time = time.time()
                inference_time = end_time - start_time
                print(f"Inference time for guidance scale {w}: {inference_time} seconds")

                x_gen_ifft = x_gen.clone()
                # Process the generated data if necessary
                # You can uncomment and modify the process_signal function as needed

                # Save the generated data
                os.makedirs(save_dir, exist_ok=True)
                savemat(os.path.join(save_dir, f'H_test_{j}.mat'), {'H': x_gen_ifft.cpu().numpy()})
                np.save(os.path.join(save_dir, f'H_test_{j}.npy'), x_gen_ifft.cpu().numpy())
                print(f'Saved H_test_{j} to {save_dir}')

def concatenate_H_tests(save_dir, output_filename):
    """
    Concatenate multiple H_test files into a single file.

    Parameters:
    - save_dir: Directory where the H_test files are saved.
    - output_filename: The filename for the concatenated output.
    """
    mat_files = [os.path.join(save_dir, f'H_test_{i}.mat') for i in range(20)]
    arrays = []
    for filename in mat_files:
        data = loadmat(filename)
        arrays.append(data['H'])

    concatenated_array = np.concatenate(arrays, axis=0)
    savemat(os.path.join(save_dir, output_filename), {'H': concatenated_array})
    print(f'Saved concatenated H_test to {os.path.join(save_dir, output_filename)}')

def main(mode):
    """
    Main function to control the flow of the script.

    Parameters:
    - mode: 'generate' to generate H_test matrices, 'concatenate' to concatenate them.
    """
    # Set up parameters
    batch_size = 5000
    n_T = 256
    device = "cuda:0"  # Change as per your setup
    n_classes = 3
    n_feat = 256
    lrate = 1e-4
    save_dir = './data/cDDIM_10000'  # Change as needed
    ws_test = [0.0]
    n_sample = 5000

    # Initialize the model
    ddim = DDIM(nn_model=ContextUnet(in_channels=2, n_feat=n_feat, n_classes=n_classes),
                betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
    ddim.to(device)

    # Load the trained model
    ddim.load_state_dict(torch.load(os.path.join(save_dir, "model.pth")))

    if mode == 'generate':
        generate_H_test(ddim, device, save_dir, start_idx=0, end_idx=20,
                        batch_size=batch_size, n_sample=n_sample, ws_test=ws_test)
    elif mode == 'concatenate':
        concatenate_H_tests(save_dir, 'H_test_concat.mat')
    else:
        print("Invalid mode selected. Choose 'generate' or 'concatenate'.")

def tensor_to_pil(tensor, scale_factor=5, border_size=1, border_color=(0, 0, 0)):
    # Convert a tensor to a PIL Image with a colormap
    array = tensor.cpu().numpy()
    cm_hot = cm.get_cmap('viridis')
    colored_image = cm_hot(array)
    pil_image = Image.fromarray((colored_image[:, :, :3] * 255).astype('uint8'), 'RGB')
    pil_image = pil_image.resize((pil_image.width * scale_factor, pil_image.height * scale_factor), Image.Resampling.LANCZOS)

    # Create a new image with borders
    new_image = Image.new('RGB', (pil_image.width + 2 * border_size, pil_image.height + 2 * border_size), color=border_color)
    new_image.paste(pil_image, (border_size, border_size))
    return new_image

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <mode>")
        print("Modes: 'generate' or 'concatenate'")
        sys.exit(1)

    mode = sys.argv[1]
    main(mode)
