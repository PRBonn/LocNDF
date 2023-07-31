import click
from os.path import join
import torch

import loc_ndf.models.models as models

import open3d as o3d


@click.command()
# Add your options here
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
              default=0.01,
              required=False)
def main(checkpoint, num_voxels, threshold):
    cfg = torch.load(checkpoint)['hyper_parameters']
    print(cfg.keys())

    # Load data and model
    model = models.LocNDF.load_from_checkpoint(checkpoint, hparams=cfg).cuda()
    model.requires_grad_(False)

    nv = [num_voxels, num_voxels, num_voxels//10]

    file = join(*checkpoint.split("/")[:-2],
                'mesh', f"size_{nv[0]}_{nv[1]}_{nv[2]}_t_{threshold:.2f}.ply")
    print(file)
    mesh = model.get_mesh(nv, threshold, mask=model.get_occupancy_mask(
        nv).cpu().numpy())  # ,file=file)

    o3d.visualization.draw_geometries(
        [mesh], mesh_show_back_face=True)


if __name__ == "__main__":
    main()
