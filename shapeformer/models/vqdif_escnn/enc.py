import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import ResnetBlockFC
from torch_scatter import scatter_mean, scatter_max
from .common import coordinate2index, normalize_3d_coordinate
# from .common import normalize_coordinate, map2local
from .common import nearest_voxel_center_coordinates
#from .encoder.unet import UNet
# from .unet3d import UNet3D
from .updown import Downsampler

class LocalPoolPointnet(nn.Module):
    ''' PointNet-based encoder network with ResNet blocks for each point.
        Number of input points are fixed.
    
    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
        scatter_type (str): feature aggregation when doing local pooling
        downsampler (bool): weather to use downsampler
        downsampler_kwargs (dict): downsampler parameters
        grid_resolution (int): defined resolution for grid feature 
        plane_type (str): feature type, 'xz' - 1-plane, ['xz', 'xy', 'yz'] - 3-plane, ['grid'] - 3D grid volume
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        n_blocks (int): number of blocks ResNetBlockFC layers
    '''

    def __init__(
        self,
        c_dim = 128,
        dim = 3,
        hidden_dim = 128,
        scatter_type = 'max',
        downsampler = False,
        downsampler_kwargs = None,
        c2i_order = "original",
        grid_resolution = None,
        plane_type = 'grid',
        padding = 0.1,
        n_blocks = 5,
        voxel_relative_point_residuals = False,
    ):
        super().__init__()
        self.c_dim = c_dim
        self.c2i_order = c2i_order
        self.voxel_relative_point_residuals = voxel_relative_point_residuals
        assert self.voxel_relative_point_residuals, 'Can not use absolute positions as point cloud input features in an equivariant setting.'

        self.fc_pos = nn.Linear(dim, 2*hidden_dim)
        self.blocks = nn.ModuleList([
            ResnetBlockFC(2*hidden_dim, hidden_dim) for i in range(n_blocks)
        ])
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.hidden_dim = hidden_dim

        # if unet3d:
        #     self.unet3d = UNet3D(**unet3d_kwargs)
        # else:
        #     self.unet3d = None
        if downsampler:
            self.downsampler = Downsampler(**downsampler_kwargs)
        else:
            self.downsampler = None

        #self.reso_plane = plane_resolution
        self.reso_grid = grid_resolution
        self.plane_type = plane_type
        self.padding = padding

        if scatter_type == 'max':
            self.scatter = scatter_max
        elif scatter_type == 'mean':
            self.scatter = scatter_mean
        else:
            raise ValueError('incorrect scatter type')

    def generate_grid_features(self, p_nor, c):
        # p_nor: point positions, shape (B, N, 3)
        # c: point features, shape (B, N, C)
        index = coordinate2index(p_nor, self.reso_grid, coord_type='3d', c2i_order=self.c2i_order)
        # scatter grid features from points
        fea_grid = c.new_zeros(p_nor.size(0), self.c_dim, self.reso_grid**3)
        c = c.permute(0, 2, 1)
        # Go from sparse points to dense 64x64x64 voxels via mean-pooling:
        fea_grid = scatter_mean(c, index, out=fea_grid) # B x C x reso^3
        fea_grid = fea_grid.reshape(p_nor.size(0), self.c_dim, self.reso_grid, self.reso_grid, self.reso_grid) # (B, C, res, res, res)
        self.pc_feature_grid = fea_grid.detach().clone()

        #if self.unet3d is not None:
        #    fea_grid = self.unet3d(fea_grid)
        if self.downsampler is not None:
            # produce (B, k*C, res/k, res/k, res/k), k=2**downsample_steps
            # Downsample feature grid 4X, using 1x1x1 stride 1 kernels & 2x2x2 stride 2 kernels:
            fea_grid = self.downsampler(fea_grid)
           
        out_reso_grid = fea_grid.shape[-1]
        assert torch.all(fea_grid.shape[-3:] == out_reso_grid), "Expected cubic grid! (equal resolution in all spatial dimensions)"
        
        # Extract x/y/z voxel indices for all points: (the purpose is to create a mask which is true at these voxels and false elsewhere)
        mask_ind      = (p_nor * out_reso_grid).long() #coordinate2index(p_nor, out_reso_grid, coord_type='3d', c2i_order=self.c2i_order)
        #print(mask_ind.shape, index.shape, self.reso_grid, out_reso_grid)
        #print( len(torch.unique(mask_ind, axis=-1)) )
        inds_flat     = mask_ind.view(-1, mask_ind.shape[-1]) # Reshape (B, N, 3) -> (B*N, 3)
        binds = torch.repeat_interleave(torch.arange(mask_ind.shape[0]).type_as(mask_ind).long(), mask_ind.shape[1]) # Flat indices for which "sample index" in the batch each point belongs to. Shape: (B*N, 3)
        mask = torch.zeros(p_nor.shape[0], out_reso_grid, out_reso_grid, out_reso_grid).bool() # (B, res, res, res)
        # Using the flat indices along each dimension, set all non-empty entries in the mask to true (non-empty meaning there are points):
        mask[binds, inds_flat[:,2], inds_flat[:,1], inds_flat[:,0]] = True

        # Dense, downsampled, feature grid, together with a mask to determine where there are any points present:
        return fea_grid, mask

    def pool_local(self, xy, index, c):
        # xy['grid']:      [0, 1] coordinates, shape (B, N, 3).   (NOTE: APPEARS UNUSED)
        # index['grid']:   Linear voxel indices, shape (B, 1, N)
        # c:               Point features, shape (B, N, C)
        bs, fea_dim = c.size(0), c.size(2)
        keys = xy.keys()

        c_out = 0
        for key in keys:
            # scatter plane features from points
            if key == 'grid':
                # [scatter]. 1st arg: point features c are permuted from (B, N, C) to (B, C, N)
                # [scatter]. 2nd arg: Voxel indices, shape (B, 1, N). (These indices will be broadcasted to (B, C, N) such that the linear voxel index is shared across all channels for a particular point in a particular batch.)
                # Default arg dim=-1 is assumed, i.e. the scatter operation is across N (the points).
                fea = self.scatter(c.permute(0, 2, 1), index[key], dim_size=self.reso_grid**3) # Outputs per-voxel mean/max, ordered according to linear voxel index. Shape: (B, C, res^3)
            else:
                raise NotImplementedError()
                #fea = self.scatter(c.permute(0, 2, 1), index[key], dim_size=self.reso_plane**2)
            if self.scatter == scatter_max:
                fea = fea[0]
            # gather feature back to points
            # [torch.gather()] The index arg consists of the linear voxel indices, expanded (with replicated values) from (B, 1, N) to (B, C, N).
            fea = fea.gather(dim=2, index=index[key].expand(-1, fea_dim, -1)) # Input shape (B, C, res^3). Output shape: (B, C, N)
            c_out += fea
        # Finally: reshape (B, C, N) back to (B, N, C).
        return c_out.permute(0, 2, 1)


    def forward(self, p):
        """
        p: Input point cloud, shape (B, N, 3)
        """
        assert len(p.shape) == 3
        assert p.shape[2] == 3
        # batch_size, T, D = p.size()

        # acquire the index for each point
        assert self.plane_type == 'grid'
        # NOTE: coord['grid'] should be identical to "p_nor" in self.generate_grid_features().
        # Basically, normalize_3d_coordinate() maps [-0.55, 0.55] linearly to [0, 1], and saturates coordinates beyond the limits.
        coord = {
            'grid': normalize_3d_coordinate(p.clone(), padding=self.padding), # In & out: (B, N, 3)
        }
        # coordinate2index() determines linear voxel indices from the [0, 1]**3 coordinates.
        index = {
            'grid': coordinate2index(coord['grid'], self.reso_grid, coord_type='3d', c2i_order=self.c2i_order), # In: (B, N, 3), out: (B, 1, N)
        }

        # Build residual vectors, relative to each voxel center
        if self.voxel_relative_point_residuals:
            coord['grid'] -= nearest_voxel_center_coordinates(coord['grid'], self.reso_grid)

        # Apply per-point MLP on point cloud
        if self.voxel_relative_point_residuals:
            # Voxel-center-to-point residual vectors are used as input features
            net = self.fc_pos(coord['grid']) # Simple linear layer acting on (B, N, 3) will regard first 2 dimensions as batch dimensions. Output: (B, N, C).
        else:
            # Absolute positions are used as input features
            net = self.fc_pos(p) # Simple linear layer acting on (B, N, 3) will regard first 2 dimensions as batch dimensions. Output: (B, N, C).

        # Point-wise residual blocks, interleaved by per-voxel mean/max-pooling.
        # Each block is a 2-layer MLP with residual connection, and has 2*hidden_dim input channels and hidden_dim output channels.
        # The blocks are applied point-wise.
        net = self.blocks[0](net)
        for block in self.blocks[1:]:
            pooled = self.pool_local(coord, index, net)
            net = torch.cat([net, pooled], dim=2)
            net = block(net)

        # A final point-wise linear layer (projecting hidden_dim -> c_dim=128).
        c = self.fc_c(net) # In / out: (B, N, C)

        # Acquire dense voxel features by per-voxel mean pooling of point features (never max pooling here!), followed by 3D-CNN downsampling:
        # The returned tensor fea is a dense, 4X downsampled, feature grid, together with a mask to determine where there are any points present:
        fea, mask = self.generate_grid_features(coord['grid'], c)
        # Intermediate resolution of resulting grid: 64x64x64 (corresponding to all local point-pooling operations). Resulting downsampled resolution: 16x16x16.
        
        #sparse_fea = self.dense2sparse(p, fea)

        # return: (B, k*C, res/k, res/k, res/k), k=2**downsample_steps
        return fea, mask
