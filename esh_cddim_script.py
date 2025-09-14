''' 
This script is a code of 
"Generating High Dimensional User-Specific Wireless Channels using Diffusion Models",
https://arxiv.org/abs/2409.03924.


This code is modified from,
https://github.com/cloneofsimo/minDiffusion

Diffusion model is based on DDIM,
https://arxiv.org/abs/2010.02502

The conditioning idea is taken from 'Classifier-Free Diffusion Guidance',
https://arxiv.org/abs/2207.12598

'''

from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import ToTensor
import scipy
import matplotlib.cm as cm
from PIL import Image, ImageDraw, ImageFont
from scipy.io import savemat
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


class SimpleContextProcessor(nn.Module):
    """
    Processes sequences of tuples by projecting each tuple to a learned embedding
    and then concatenating them. It can use a dedicated embedder for the first element.
    """
    def __init__(self, output_dim=256, emb_dim_per_tuple=128, first_element_embedder=None):
        super(SimpleContextProcessor, self).__init__()
        self.output_dim = output_dim
        self.emb_dim_per_tuple = emb_dim_per_tuple
        self.first_element_embedder = first_element_embedder
        self.context_embedders = nn.ModuleDict()
        self.final_projection = None

    def forward(self, context_sequence):
        if isinstance(context_sequence, torch.Tensor):
            context_sequence = [(context_sequence,)]
        batch_size = context_sequence[0][0].shape[0] if isinstance(context_sequence[0], tuple) else context_sequence[0].shape[0]
        device = context_sequence[0][0].device if isinstance(context_sequence[0], tuple) else context_sequence[0].device

        

        embedded_contexts = []
        for i, context_tuple in enumerate(context_sequence):
            #print(context_tuple)
            context_tensor = torch.cat([t.view(batch_size, -1) for t in context_tuple], dim=1) if isinstance(context_tuple, tuple) else context_tuple.view(batch_size, -1)
            #print(context_tensor.shape)
            if i == 0 and self.first_element_embedder is not None:
                embedded = self.first_element_embedder(context_tensor)
                print(embedded.shape)
                print(context_tensor.shape[1])
            else:
                context_dim = context_tensor.shape[1]
                print(context_dim)
                
                embedder_key = f"embedder_{i}_dim_{context_dim}"

                if embedder_key not in self.context_embedders:
                    self.context_embedders[embedder_key] = nn.Sequential(
                        nn.Linear(context_dim, self.emb_dim_per_tuple),
                        nn.GELU(),
                        nn.Linear(self.emb_dim_per_tuple, self.emb_dim_per_tuple)
                    ).to(device)
                
                embedded = self.context_embedders[embedder_key](context_tensor)
                print(embedded.shape)
                

            embedded_contexts.append(embedded)

        
        concatenated_contexts = torch.cat(embedded_contexts, dim=1)
        
        concatenated_dim = concatenated_contexts.shape[1]
        if self.final_projection is None or not isinstance(self.final_projection, nn.Identity) and self.final_projection[0].in_features != concatenated_dim:
            self.final_projection = nn.Sequential(
                nn.Linear(concatenated_dim, self.output_dim),
                nn.GELU(),
                nn.Linear(self.output_dim, self.output_dim)
            ).to(device)
        
        return self.final_projection(concatenated_contexts)


class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat=256, n_classes=6, use_variable_context=True):
        super(ContextUnet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes
        self.use_variable_context = use_variable_context

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d((1, 8)), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2 * n_feat)
        self.timeembed2 = EmbedFC(1, 1 * n_feat)

        # Embedder for fixed context or the first element of variable context
        # The first context (user location) is 3-dimensional
        self.contextembed1 = EmbedFC(3, 2 * n_feat)
        self.contextembed2 = EmbedFC(3, 1 * n_feat)
        
        if use_variable_context:
            self.context_processor = SimpleContextProcessor(output_dim=2 * n_feat, first_element_embedder=self.contextembed1)
            self.context_processor2 = SimpleContextProcessor(output_dim=1 * n_feat, first_element_embedder=self.contextembed2)
        else:
            # For clarity, set processors to None when not used
            self.context_processor = None
            self.context_processor2 = None

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

        # embed time step
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        # Process context based on type
        if self.use_variable_context:
            # c is expected to be a list of tuples with varying dimensions
            # e.g., [(x_coords, y_coords, z_coords), (los_status,), (antenna_count, freq, power)]
            if isinstance(c, torch.Tensor) and c.ndim == 2 and c.shape[1] == 9:
                # Split 9D context into 5 groups: [x, y, z, bs_ant_h, bs_ant_v, ue_ant_h, ue_ant_v, bs_spacing, ue_spacing]
                # Group 1: [x, y, z] - user location (3D)
                # Group 2: [bs_ant_h, bs_ant_v] - BS antenna config (2D)
                # Group 3: [ue_ant_h, ue_ant_v] - UE antenna config (2D)
                # Group 4: [bs_spacing] - BS spacing (1D)
                # Group 5: [ue_spacing] - UE spacing (1D)
                c = [c[:, :3], c[:, 3:5], c[:, 5:7], c[:, 7:8], c[:, 8:9]]
            
            cemb1 = self.context_processor(c).view(-1, self.n_feat * 2, 1, 1)
            cemb2 = self.context_processor2(c).view(-1, self.n_feat, 1, 1)
        else:
            # Original fixed-size context processing
            c = c.float()
            cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
            cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)

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
    # mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab # DDPM coef.
    DDIM_coeff = sqrtmab - torch.sqrt(alpha_t) * torch.sqrt(1 - alphabar_t / alpha_t) # DDIM coef.

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

        # register_buffer allows accessing dictionary produced by ddpm_schedules
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
    def __init__(self, dataset_folder="../datasets/DeepMIMO_dataset", idx_start=0, idx_end=None, use_deepmimo=True):
        self.dataset_folder = dataset_folder
        self.idx_start = idx_start
        self.idx_end = idx_end
        self.use_deepmimo = use_deepmimo
        self.load_dataset()
    
    def load_dataset(self):
        if self.use_deepmimo:
            self.load_deepmimo_dataset()
        else:
            self.load_original_dataset()
    
    def load_deepmimo_dataset(self):
        """Load DeepMIMO dataset using our custom loader"""
        # Import our DeepMIMO loader
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        from load_deepmimo_datasets import load_deepmimo_datasets, create_ml_dataset
        
        # Load all DeepMIMO datasets
        print("Loading DeepMIMO datasets...")
        # Use absolute path to ensure we find the dataset folder
        #dataset_path = os.path.join(os.path.dirname(__file__), '..', self.dataset_folder)
        dataset_path = os.path.join(self.dataset_folder)
        data = load_deepmimo_datasets(dataset_path, verbose=True)
        
        # Create ML-ready dataset
        X, y, metadata = create_ml_dataset(data, include_metadata=True)
        
        print(f"Loaded DeepMIMO data: {X.shape[0]} samples")
        
        # Convert to the format expected by the diffusion model
        self.data = []
        self.labels = []
        
        # Determine end index
        if self.idx_end is None:
            self.idx_end = X.shape[0]
        
        # Ensure indices are within bounds
        self.idx_start = max(0, min(self.idx_start, X.shape[0]))
        self.idx_end = max(self.idx_start, min(self.idx_end, X.shape[0]))
        
        print(f"Processing samples {self.idx_start} to {self.idx_end}")
        
        for i in range(self.idx_start, self.idx_end):
            # Reshape flattened channel data back to 2D
            # Assuming the channel was flattened as (num_ant_BS * num_ant_UE,)
            # We need to determine the antenna dimensions from metadata
            if i < len(metadata):
                meta = metadata[i]
                bs_ant_h = meta.get('bs_ant_h', 8)  # Default values
                bs_ant_v = meta.get('bs_ant_v', 4)
                ue_ant_h = meta.get('ue_ant_h', 2)
                ue_ant_v = meta.get('ue_ant_v', 2)
                
                # Reshape to (num_ant_BS, num_ant_UE) format
                channel_2d = X[i].reshape(ue_ant_h * ue_ant_v, bs_ant_h * bs_ant_v )
                
                # Convert to complex and then to real/imaginary parts
                # For now, we'll use the real part as channel data
                # You may need to adjust this based on your specific channel representation
                channel_complex = channel_2d  # Assuming X[i] is already complex
                
                # Stack real and imaginary parts
                array1 = np.stack((np.real(channel_complex), np.imag(channel_complex)), axis=0)
                
                # Apply FFT and normalization similar to original code
                dft_data = np.fft.fft2(array1[0] + 1j * array1[1])
                dft_shifted = np.fft.fftshift(dft_data)
                array1[0] = np.real(dft_shifted)
                array1[1] = np.imag(dft_shifted)
                
                # Normalize by maximum magnitude
                magnitude = np.sqrt(array1[0, :, :]**2 + array1[1, :, :]**2)
                max_magnitude = np.max(magnitude)
                if max_magnitude > 0:
                    array1[0, :, :] /= max_magnitude
                    array1[1, :, :] /= max_magnitude
                
                self.data.append(array1[:2, :, :])
                
                # Create combined label with user location, antenna configuration, and spacing
                if 'user_location' in meta and meta['user_location'] is not None:
                    # Flatten user location and combine with other parameters
                    location = meta['user_location'].flatten() if hasattr(meta['user_location'], 'flatten') else meta['user_location']
                    
                    # Get additional parameters and ensure they are scalars
                    bs_spacing = float(meta.get('bs_spacing', 0.5))  # Default spacing
                    ue_spacing = float(meta.get('ue_spacing', 0.5))  # Default spacing
                    
                    # Combine: [x, y, z, bs_ant_h, bs_ant_v, ue_ant_h, ue_ant_v, bs_spacing, ue_spacing]
                    combined_label = np.concatenate([
                        location, 
                        [bs_ant_h, bs_ant_v, ue_ant_h, ue_ant_v],
                        [bs_spacing, ue_spacing]
                    ])
                    self.labels.append(combined_label)
                else:
                    # Fallback to default values
                    # Use default location [0, 0, 0] + antenna config + default spacing
                    self.labels.append(np.array([0, 0, 0, bs_ant_h, bs_ant_v, ue_ant_h, ue_ant_v, 0.5, 0.5]))
            else:
                # Fallback for missing metadata
                channel_2d = X[i].reshape(8, 4)  # Default shape
                array1 = np.stack((np.real(channel_2d), np.imag(channel_2d)), axis=0)
                self.data.append(array1[:2, :, :])
                # Use default values for combined label: [0, 0, 0, 8, 4, 2, 2, 0.5, 0.5]
                self.labels.append(np.array([0, 0, 0, 8, 4, 2, 2, 0.5, 0.5]))  # Default location + antenna config + spacing
    
    def load_original_dataset(self):
        """Load original BerUMaL dataset format"""
        contents = scipy.io.loadmat(self.data_path)
        array1 = contents['H_set'][self.idx_start:self.idx_end,:,:] # (10000, 4, 32)
        array1 = np.stack((np.real(array1[:,:,:]), np.imag(array1[:,:,:])), axis=1)
        array2 = array1.copy()
        for i in range(self.idx_end - self.idx_start) : 
            dft_data = np.fft.fft2(array2[i,0]+1j*array2[i,1])
            dft_shifted = np.fft.fftshift(dft_data)
            array1[i,0] = np.real(dft_shifted)
            array1[i,1] = np.imag(dft_shifted)

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

def train():
    n_epoch = 1000 #50000
    batch_size = 128
    n_T = 256
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    n_classes = 9  # Updated to match our combined label format: [x, y, z, bs_ant_h, bs_ant_v, ue_ant_h, ue_ant_v, bs_spacing, ue_spacing]
    n_feat = 256
    lrate = 1e-4
    save_model = True

    num_samples = 10# 1000 #10000

    # save_dir = f'../../outputs/esh_ddim/cDDIM_{num_samples}/'

    save_dir = f'./data/cDDIM_{num_samples}/'
    os.makedirs(save_dir, exist_ok=True)

    ws_test = [0.0] # strength of generative guidance
    n_sample = 10

    ddim = DDIM(
    nn_model=ContextUnet(in_channels=2, n_feat=n_feat, n_classes=n_classes),
    betas=(1e-4, 0.02), 
    n_T=n_T, 
    device=device, 
    drop_prob=0.1
    )
    ddim.to(device)

    # Load DeepMIMO dataset (or original BerUMaL dataset)
    use_deepmimo = True  # Set to False to use original BerUMaL dataset
    
    if use_deepmimo:
        # Use DeepMIMO dataset
        berumal_dataset_train = BerUMaLDataset("../../datasets/DeepMIMO_dataset", 0, num_samples, use_deepmimo=True)
        berumal_dataset_test = BerUMaLDataset("../../datasets/DeepMIMO_dataset", num_samples, num_samples + 1000, use_deepmimo=True)
    else:
        # Use original BerUMaL dataset
        berumal_dataset_train = BerUMaLDataset("./data/QuaDRiGa/NumUEs_100000_num_BerUMaL_ULA.mat", 0, num_samples, use_deepmimo=False)
        berumal_dataset_test = BerUMaLDataset("./data/QuaDRiGa/NumUEs_100000_num_BerUMaL_ULA.mat", 90000, 100000, use_deepmimo=False)

    # Create a DataLoader for the BerUMaL dataset
    print(berumal_dataset_train.labels)
    dataloader_train = DataLoader(berumal_dataset_train, batch_size=batch_size, shuffle=True)
    data_labels = dataloader_train.dataset.labels
    #print(data_labels[10])
    dataloader_test = DataLoader(berumal_dataset_test, batch_size=batch_size, shuffle=False)

    # Select a fixed subset of test samples and their corresponding coordinates\
    # fixed_test_samples, fixed_test_coords = next(iter(dataloader_test))
    # fixed_test_samples = fixed_test_samples[:n_sample].to(device)
    # fixed_test_coords = fixed_test_coords[:n_sample].to(device)

    optim = torch.optim.Adam(ddim.parameters(), lr=lrate)

    nmse_values_test = []  # List to store NMSE values
    os.makedirs(save_dir, exist_ok=True)
    for ep in range(n_epoch):
        print(f'epoch {ep}')
        ddim.train()

        # linear lrate decay
        optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)

        pbar = tqdm(dataloader_train)
        loss_ema = None
        for x, c in pbar:
            optim.zero_grad()
            x = np.transpose(x, (0, 2, 3, 1))
            x = x.to(device)
            c = c.to(device)
            c = torch.tensor(c, dtype=torch.float32).to(device)
            loss = ddim(x, c)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

        # optionally save model
        if ep % 500 == 0 : 
            torch.save(ddim.state_dict(), save_dir + f"model_{ep}.pth")
            print('saved model at ' + save_dir + f"model_{ep}.pth")

def demo_variable_context():
    """
    Demonstration of how to use the new variable context functionality.
    Shows how to create and use sequences of tuples with varying dimensions.
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Create model with variable context enabled
    model = ContextUnet(
        in_channels=2, 
        n_feat=256, 
        n_classes=3, 
        use_variable_context=True
    ).to(device)
    
    # Example input data
    batch_size = 4
    x = torch.randn(batch_size, 2, 4, 32).to(device)  # Channel data
    t = torch.rand(batch_size, 1, 1, 1).to(device)    # Time step
    context_mask = torch.zeros(batch_size, 1).to(device)
    
    # Example variable context: sequence of tuples with different dimensions
    # Each tuple represents different types of context information
    context_sequence = [
        # Tuple 1: 3D coordinates (x, y, z)
        (torch.randn(batch_size, 1).to(device),  # x coordinate
         torch.randn(batch_size, 1).to(device),  # y coordinate  
         torch.randn(batch_size, 1).to(device)), # z coordinate
        
        # Tuple 2: Line-of-sight status (binary)
        (torch.randint(0, 2, (batch_size, 1)).float().to(device),),  # LOS/NLOS status
        
        # Tuple 3: Hardware parameters (antenna count, frequency, power)
        (torch.randint(1, 17, (batch_size, 1)).float().to(device),   # antenna count
         torch.randn(batch_size, 1).to(device),                      # frequency
         torch.randn(batch_size, 1).to(device))                      # power
    ]
    
    print("Variable Context Example:")
    print(f"Context sequence length: {len(context_sequence)}")
    for i, ctx_tuple in enumerate(context_sequence):
        print(f"  Tuple {i+1}: {[tensor.shape for tensor in ctx_tuple]}")
    
    # Forward pass
    with torch.no_grad():
        output = model(x, context_sequence, t, context_mask)
        print(f"Model output shape: {output.shape}")
        print("✓ Variable context processing successful!")


if __name__ == "__main__":
    train()
    #demo_variable_context()
