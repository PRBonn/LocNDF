import torch
from loc_ndf.utils import vis
from pytorch_lightning.callbacks import Callback

################################################################
# Point to point dist
################################################################


def chamfer_point(points: torch.Tensor, mesh, num_samples=1e7, scale=1):
    p_o3d = vis.torch2o3d(points)
    map = mesh.sample_points_uniformly(number_of_points=int(num_samples))
    p2m = torch.tensor(p_o3d.compute_point_cloud_distance(map)) * scale
    m2p = torch.tensor(map.compute_point_cloud_distance(p_o3d)) * scale
    return p2m, m2p


class ReconsructionCallback(Callback):
    def __init__(self, dataset, num_samples, num_voxels=1000) -> None:
        super().__init__()
        self.data = dataset
        self.points = dataset.points
        self.num_samples = num_samples
        self.num_voxels = num_voxels

        self.chamfer = 1e5

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        nv = [self.num_voxels, self.num_voxels, self.num_voxels//10]
        p2m, m2p = chamfer_point(points=self.points,
                                 mesh=pl_module.get_mesh(
                                     nv=nv,
                                     mask=pl_module.get_occupancy_mask(
                                         nv).cpu().numpy()
                                 ),
                                 num_samples=self.num_samples)
        self.log(f"reconstruction/p2m", p2m.mean())
        self.log(f"reconstruction/m2p", m2p.mean())
        self.log(f"reconstruction/chamfer", (p2m.mean()+m2p.mean())/2)
        self.chamfer = (p2m.mean()+m2p.mean())/2
        pass
