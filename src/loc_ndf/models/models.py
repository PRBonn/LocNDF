from math import pi
import torch
import torch.nn as nn

from pytorch_lightning.core.module import LightningModule
from loc_ndf.models import loss
from loc_ndf.utils import vis, utils
import open3d as o3d
from easydict import EasyDict
import tqdm
from loc_ndf.utils import pytimer


##################################################################
################ Lightning Module ################################
##################################################################
class LocNDF(LightningModule):
    def __init__(self, hparams: dict):
        super().__init__()
       # name you hyperparameter hparams, then it will be saved automagically.
        self.save_hyperparameters(hparams)
        self.update_map_params()  # self.T_local and self.occupancy_mask

        self.model = get_network(hparams['model'])
        self.loss = loss.ProjectedDistanceLoss(**hparams['loss']['params'])

    def update_map_params(self, points=None):
        T_local = utils.compute_T_from_bounding_box(
            self.hparams['bounding_box'])
        self.register_buffer('T_local', T_local)

        mask = utils.get_occ_mask(
            points=points, **self.hparams['occupancy_mask'])
        self.register_buffer('occupancy_mask', mask)

    def forward(self, x: torch.Tensor, in_global_frame=True):
        if in_global_frame:
            x = utils.transform(x, self.T_local)
        distance = self.model(x)
        return distance

    def training_step(self, batch: dict, batch_idx):
        points = self.forward(batch['points'])

        inter_grad, inter_val = self.compute_gradient_dists(batch['inter'])

        rand_grad, _ = self.compute_gradient_dists(batch['random'])

        loss, losses = self.loss(
            points_distance=points,
            points=batch['points'],
            inter_val=inter_val,
            inter_grad=inter_grad,
            inter_pos=batch['inter'],
            ray_dists=batch['dists'],
            rand_grad=rand_grad)

        for k, v in losses.items():
            self.log(f'train/{k}', v)
        return loss

    def compute_gradient_dists(self, position: torch.Tensor):
        position.requires_grad_(True)
        val = self.forward(position)
        grad = utils.compute_gradient(val, position)
        return grad, val

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams['train']['lr'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams['train']['max_epoch'], eta_min=self.hparams['train']['lr']/1e3)
        return [optimizer], [scheduler]

    def get_grid(self, nv=[500, 500, 50]):
        mins = self.hparams['bounding_box'][0]
        maxs = self.hparams['bounding_box'][1]
        x = torch.linspace(mins[0], maxs[0], nv[0],
                           device=self.device, dtype=self.dtype)
        y = torch.linspace(mins[1], maxs[1], nv[1],
                           device=self.device, dtype=self.dtype)
        z = torch.linspace(mins[2], maxs[2], nv[2],
                           device=self.device, dtype=self.dtype)
        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        # make homogenous
        grid = torch.stack([xx, yy, zz, torch.ones_like(zz)], -1)
        return grid

    def get_occupancy_mask(self, nv=[500, 500, 50]):
        """ Returns the occupancy mask for evaluating the mesh for a given voxelsize.
        Be aware that the stored occupancy might be stored in a lower resolution thus the values are interpolated. 
        """
        return torch.nn.functional.interpolate(self.occupancy_mask[None, None, ...].float(), size=nv).squeeze().bool()

    def get_mesh(self, nv=[500, 500, 50], tau=None, file=None, verbose=False, mask=None):
        tau = self.hparams['data']['isosurface'] if tau is None else tau
        pytimer.tic()
        if verbose:
            print('Start meshing')
        grid = self.get_grid(nv)
        pytimer.tocTic('Created grid', verbose=verbose)
        rows = []
        with torch.no_grad():
            for row in tqdm.tqdm(grid, leave=False, desc='Eval Grid'):
                rows.append(self.forward(row))
        pytimer.tocTic('Evaluated model', verbose=verbose)
        dist = torch.stack(rows).squeeze().cpu().numpy()
        mesh = vis.grid_to_mesh(
            dist, tau=tau,
            ascent=self.hparams['data']['gradient_ascent'],
            mins=self.hparams['bounding_box'][0],
            maxs=self.hparams['bounding_box'][1],
            mask=mask)
        pytimer.tocTic('finished meshing', verbose=verbose)

        if file is not None:
            o3d.io.write_triangle_mesh(file, mesh)
        pytimer.tocTic(f'stored to disk: {file}', verbose=verbose)
        return mesh

##################################################################
################ Network Architecture ############################
##################################################################


class LidarNerf(nn.Module):
    def __init__(self,
                 inter_fdim: int,
                 pos_encoding: dict,
                 sigmoid=True):
        super().__init__()
        self.pos_encoding = PositionalEncoder(**pos_encoding['params'])
        in_dim = self.pos_encoding.featureSize()
        self.net1 = nn.Sequential(
            MLP_Block(in_dim, inter_fdim),
            MLP_Block(inter_fdim,
                      inter_fdim),
            MLP_Block(inter_fdim,
                      inter_fdim),
            MLP_Block(inter_fdim,
                      inter_fdim),
            MLP_Block(inter_fdim,
                      inter_fdim))
        self.net2 = nn.Sequential(
            MLP_Block(inter_fdim+in_dim,
                      inter_fdim),
            MLP_Block(inter_fdim,
                      inter_fdim),
            MLP_Block(inter_fdim,
                      inter_fdim),
            nn.Linear(inter_fdim, 1),
            nn.Sigmoid() if sigmoid else nn.Identity()
        )

    def forward(self, x):
        pos = self.pos_encoding(x)

        x = self.net1(pos)
        x = torch.cat([x, pos], -1)
        x = self.net2(x)
        return x


class Siren(nn.Module):
    def __init__(self,
                 inter_fdim: int,
                 pos_encoding: dict,
                 sigmoid=True,
                 sin=True):
        super().__init__()
        self.pos_encoding = PositionalEncoder(**pos_encoding['params'])
        in_dim = self.pos_encoding.featureSize()
        self.net = nn.Sequential(
            MLP_Block(in_dim, inter_fdim, sin=sin),
            MLP_Block(inter_fdim,
                      inter_fdim, sin=sin),
            MLP_Block(inter_fdim,
                      inter_fdim, sin=sin),
            MLP_Block(inter_fdim,
                      inter_fdim, sin=sin),
            MLP_Block(inter_fdim,
                      inter_fdim, sin=sin),
            nn.Linear(inter_fdim, 1),
            nn.Sigmoid() if sigmoid else nn.Identity()
        )

    def forward(self, x):
        pos = self.pos_encoding(x)
        x = self.net(pos)
        return x


class Networks(EasyDict):
    LidarNerf = LidarNerf
    Siren = Siren


def get_network(type_params: dict) -> nn.Module:
    """ dict type_params {
        type: str, specifies which loss
        params: dict, kwargs of the loss
    }
    """
    return Networks()[type_params['type']](**type_params['params'])


################################################################
###################### Blocks ##################################
################################################################

class SinLayer(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class MLP_Block(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, sin=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=out_dim),
            nn.LayerNorm(out_dim),
            SinLayer() if sin else nn.LeakyReLU())

    def forward(self, x):
        return self.net(x)


################################################################
################ Positional Encoding ###########################
################################################################


class PositionalEncoder(nn.Module):
    def __init__(self, freq, num_bands=5, dimensionality=3, base=2):
        super().__init__()
        self.freq, self.num_bands = torch.tensor(freq), num_bands
        self.dimensionality, self.base = dimensionality, torch.tensor(base)

    def forward(self, x):
        x = x[..., :self.dimensionality, None]
        device, dtype, orig_x = x.device, x.dtype, x

        scales = torch.logspace(0., torch.log(
            self.freq / 2) / torch.log(self.base), self.num_bands, base=self.base, device=device, dtype=dtype)
        # Fancy reshaping
        scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

        x = x * scales * pi
        x = torch.cat([x.sin(), x.cos()], dim=-1)
        x = torch.cat((x, orig_x), dim=-1)
        x = x.flatten(-2, -1)
        return x

    def featureSize(self):
        return self.dimensionality*(self.num_bands*2+1)
################################################################
