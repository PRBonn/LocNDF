from loc_ndf.utils import registration
from copy import deepcopy
import click
from os.path import join
import torch
import loc_ndf.datasets.datasets as datasets
import loc_ndf.models.models as models
import numpy as np
from loc_ndf.utils import vis


class Registrator:
    def __init__(self, checkpoint,
                 scan_idx_offset=4,
                 num_voxels=400,
                 threshold=0.01) -> None:

        cfg = torch.load(checkpoint)['hyper_parameters']

        # Load data and model
        self.model = models.LocNDF.load_from_checkpoint(
            checkpoint, hparams=cfg).cuda()
        self.model.requires_grad_(False)

        data = datasets.DataModule(cfg).get_train_set()
        self.bb_size = data.bb_size
        folder = data.folder
        ids, poses, _ = data.get_registration_scans(folder)

        # get the next one
        i = int(np.argwhere(ids == (data.scan_idx+scan_idx_offset+1)).squeeze())
        scan_id, self.scan_pose = int(ids[i]), poses[i]

        self.scan_gt = datasets.parse_scan(
            join(folder, f'pcds/{str(scan_id)}.pcd'))
        self.scan_raw = deepcopy(self.scan_gt)
        self.scan_orig = deepcopy(self.scan_gt)
        self.scan_orig.paint_uniform_color([1, 0, 0])

        self.scan_gt.transform(self.scan_pose)
        self.scan_gt.paint_uniform_color([0, 1, 0])

        nv = [num_voxels, num_voxels, num_voxels//10]
        self.map = self.model.get_mesh(nv, threshold, mask=self.model.get_occupancy_mask(
            nv).cpu().numpy())

        # Registration
        self.points = torch.tensor(np.asarray(self.scan_orig.points).astype(
            np.float32), device=self.model.device)
        self.points = torch.cat(
            [self.points, torch.ones_like(self.points[..., :1])], -1)

        self.T = torch.eye(4, 4, device=self.points.device)
        print("Red: Initial Guess")
        print("Cyan: Current Transformation")
        print("Green: GT-Transformation")

    def getGeometries(self, index):
        points = (self.T @ self.points.T).T
        points = points.detach()

        T = deepcopy(self.T)
        scan_t = deepcopy(self.scan_orig)

        est_pose = T.cpu().detach().numpy()

        scan_t.transform(est_pose)
        scan_t.paint_uniform_color([0, 1, 1])

        bb = torch.tensor(
            self.model.hparams['bounding_box'], device=points.device)
        within = ((points[..., :3] >= bb[0]) & (
            points[..., :3] < bb[1])).all(-1)
        DT = registration.registration_step(
            points[within], self.model).detach().float()
        self.T = DT@self.T

        return [self.map, self.scan_gt, scan_t, self.scan_orig]


@click.command()
@click.option('--checkpoint',
              '-c',
              type=str,
              help='path to checkpoint file (.ckpt)',
              required=True)
@click.option('--num_voxels',
              '-v',
              type=int,
              default=400,
              required=True)
@click.option('--threshold',
              '-t',
              type=float,
              default=0.01)
@click.option('--scan_offset',
              '-s',
              type=int,
              default=1)
def main(checkpoint, num_voxels, threshold, scan_offset):

    visualizer = vis.Visualizer(
        Registrator(checkpoint, scan_offset,
                    threshold=threshold, num_voxels=num_voxels)
    )
    visualizer.run()


if __name__ == "__main__":
    main()
