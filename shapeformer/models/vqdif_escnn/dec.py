# import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import ResnetBlockFC
from .common import normalize_3d_coordinate
# from .common import normalize_coordinate, map2local

from .unet3d import UNet3D
from .updown import Upsampler

class LocalDecoder(nn.Module):
    ''' Decoder.
        Instead of conditioning on global features, on plane/volume local features.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        unet3d (bool): weather to use 3D U-Net
        unet3d_kwargs (dict): 3D U-Net parameters
        upsampler (bool): weather to use upsampler
        upsampler_kwargs (dict): upsampler parameters
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    '''

    def __init__(
        self,
        dim = 3,
        c_dim = 128,
        unet3d = False,
        unet3d_kwargs = None,
        upsampler = False,
        upsampler_kwargs = None,
        hidden_size = 256,
        n_blocks = 5,
        leaky = False,
        sample_mode = 'bilinear',
        padding = 0.1,
    ):
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        if unet3d:
            self.unet3d = UNet3D(**unet3d_kwargs)
        else:
            self.unet3d = None

        if upsampler:
            self.upsampler = Upsampler(**upsampler_kwargs)
        else:
            self.upsampler = None

        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])


        self.fc_p = nn.Linear(dim, hidden_size)

        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)
        ])

        self.fc_out = nn.Linear(hidden_size, 1)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode
        self.padding = padding
    

    def sample_grid_feature(self, p, c):
        # p: probe positions, shape (B, N, 3)
        # c: grid features, shape (B, C, D_in, H_in, W_in)
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding) # normalize to the range of (0, 1)
        p_nor = p_nor[:, :, None, None].float() # Shape (B, N, 1, 1, 3)
        vgrid = 2.0 * p_nor - 1.0 # normalize to (-1, 1)
        # acutally trilinear interpolation if mode = 'bilinear'
        # F.grid_sample() in general performs resampling of the voxel data in the first argument (shape: (B, C, D_in, H_in, W_in)) on a new 3D grid determined by the second argument (shape: (B, D_out, H_out, W_out, 3)).
        # Each position in this grid holds a 3D-vector acting as a position for lookup from the voxel data. These positions are normalized to the [-1, 1] range in every dimension.
        # In this particular case, we are not interested in maintaining any grid structure of the probe points. Thus, the second argument, vgrid, has 2 singleton dimensions, and can be regarded as 1-dimensional.
        # Specifically, D_out=N, H_out=W_out=1.
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1).squeeze(-1) # Output shape (B, C, D_out, H_out, W_out) = (B, C, N, 1, 1), squeezed to (B, C, N).
        return c


    def forward(self, c_grid, p, **kwargs):
        # Inputs:
        # - c_grid: dense 16x16x16 feature grid. Shape (B, C, 16, 16, 16)
        # - p: "probes" at which to evaluate implicit decoder. Shape (B, N, 3)

        # unet & upsample
        # in & out: (B, k*C, res/k, res/k, res/k), k=2**downsample_steps

        # UNet: 16x16x16 -> 2x2x2 -> 16x16x16.
        if self.unet3d is not None:
            uneted    = self.unet3d(c_grid)
        else:
            uneted    = c_grid
        # (B, C, res,res,res)
        # Upsampling: 16x16x16 -> 64x64x64.
        if self.upsampler is not None:
            upsampled = self.upsampler(uneted)
        else:
            upsampled = uneted
        # 64x64x64 by now.

        # implicit decoder
        # Sample features at "probes" via trilinear interpolation:
        c = self.sample_grid_feature(p, upsampled) # Shape (B, C, N)
        c = c.transpose(1, 2) # Shape (B, N, C)

        # All FC layers are applied individually per each pair of "probe" and the corresponding sampled feature from the grid.

        # "probes" p are first fed as inputs to an initial FC layer.
        p = p.float()
        net = self.fc_p(p)

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                # In every block we mix in the upsampled output from the 3D Unet, sampled at the probes:
                net = net + self.fc_c[i](c)

            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        out = out

        return out
