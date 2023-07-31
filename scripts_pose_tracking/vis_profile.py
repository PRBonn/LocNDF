
import numpy as np
import matplotlib.pyplot as plt
import click
import torch

import loc_ndf.datasets.datasets as datasets
import loc_ndf.models.models as models
from loc_ndf.utils import vis

import matplotlib.pyplot as plt

from tqdm import tqdm

from loc_ndf.datasets import datasets
import open3d as o3d
import matplotlib.cm as cm


def get_points(cfg):
    data = datasets.StatDataModule(cfg).train_dataloader()
    for batch in data:
        break

    points = batch['points'].numpy()
    return points


class Handler():
    def __init__(self, grid, points, axis=2, max_dist=None) -> None:
        self.fig, (self.ax, self.ax_cloud) = plt.subplots(1, 2)
        self.fig.canvas.mpl_connect('key_press_event', self.on_press)
        self.axis = axis
        # print('axis',axis)

        self.ax.set_title('Press n for next and b for back')
        self.ax.set_xlabel('y')
        self.ax.set_ylabel('x')

        self.grid = grid
        self.min = points.min(0)
        self.max = points.max(0)
        print(self.min, self.max)
        self.d = (self.max[axis]-self.min[axis])/self.grid.shape[axis]
        self.i = self.grid.shape[axis]//2

        self.ax_cloud = self.fig.add_subplot(122, projection='3d',)

        self.grid = np.swapaxes(grid, self.axis, 2)
        self.max_dist = np.max(grid) if max_dist is None else max_dist
        self.points = points
        self.draw()

    def get_surface(self):
        if self.axis == 0:
            y = np.linspace(self.min[1], self.max[1], 20)
            z = np.linspace(self.min[2], self.max[2], 20)
            Y, Z = np.meshgrid(y, z)
            X = np.zeros_like(Y)+self.i * self.d + self.min[self.axis]
            return X, Y, Z
        if self.axis == 1:
            x = np.linspace(self.min[0], self.max[0], 20)
            z = np.linspace(self.min[2], self.max[2], 20)
            X, Z = np.meshgrid(x, z)
            Y = np.zeros_like(X)+self.i * self.d + self.min[self.axis]
            return X, Y, Z
        if self.axis == 2:
            x = np.linspace(self.min[0], self.max[0], 20)
            y = np.linspace(self.min[1], self.max[1], 20)
            X, Y = np.meshgrid(x, y)
            Z = np.zeros_like(X)+self.i * self.d + self.min[self.axis]
            return X, Y, Z

    def aspect_ratio(self):
        max_range = (self.points.max(0)-self.points.min(0)).max() / 2.0
        mid = (self.points.max(0)+self.points.min(0))/2
        self.ax_cloud.set_xlim(mid[0] - max_range, mid[0] + max_range)
        self.ax_cloud.set_ylim(mid[1] - max_range, mid[1] + max_range)
        self.ax_cloud.set_zlim(mid[2] - max_range, mid[2] + max_range)

    def on_press(self, event):
        if event.key == 'n':
            self.i += 1
            self.draw()
        if event.key == 'b':
            self.i -= 1
            self.draw()

    def draw(self):
        self.ax.clear()
        self.ax.imshow(self.grid[:, :, self.i], origin='lower')

        self.ax_cloud.clear()

        X, Y, Z = self.get_surface()
        self.ax_cloud.plot_surface(X, Y, Z, color=[1, 0, 0, 1])
        self.ax_cloud.set_xlabel('x')
        self.ax_cloud.set_ylabel('y')
        self.fig.canvas.draw()

    def run(self):
        plt.show()

    def get_image(self, i, name):
        img = self.grid[:, :, i]
        img = (img-np.min(img)) / (self.max_dist-np.min(img))
        img = (cm.get_cmap()(img)[::-1, :, :3]*255).astype(np.uint8)

        # img = np.flipud(img)
        img = o3d.geometry.Image(img)
        y = i * self.d + self.min[self.axis]
        minz, maxz = self.min[2], self.max[2]
        w = 25
        corners = np.array([
            [y, -w, minz],
            [y, w, minz],
            [y, w, maxz],
            [y, -w, maxz],
        ])

        triangles = np.array([[2, 1, 0], [2, 0, 3]])
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(corners),
                                         o3d.utility.Vector3iVector(triangles))
        uvs = np.array([(1, 1), (1, 0), (0, 0), (1, 1), (0, 0), (0, 1)])
        mesh.triangle_uvs = o3d.utility.Vector2dVector(uvs)

        mesh.compute_vertex_normals()

        mat = o3d.visualization.rendering.MaterialRecord()
        mat.albedo_img = img
        return {'name': name, 'geometry': mesh, 'material': mat}


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
@click.option('--axis',
              '-a',
              type=int,
              default=0,
              required=True)
@click.option('--index',
              '-i',
              type=int,
              default=10,
              required=True)
def main(checkpoint, num_voxels, axis, index):
    start_idx = num_voxels//2
    cfg = torch.load(checkpoint)['hyper_parameters']

    # Load data and model
    model = models.LocNDF.load_from_checkpoint(checkpoint, hparams=cfg).cuda()

    res = 1/100
    data_set = datasets.get_dataset(cfg['data']['train'])
    points = data_set.get_points()
    points = np.unique((points/res).astype(np.int64),
                       axis=0).astype(np.float32)*res

    nv = [num_voxels, num_voxels, num_voxels//10]
    grid = model.get_grid(nv).cuda()

    rows = []
    for row in tqdm(grid):
        rows.append(model(row))
    dists = torch.stack(rows).squeeze().cpu().numpy()

    h = Handler(dists,
                points=points,
                axis=axis)
    pts = vis.torch2o3d(data_set.get_points(), estimate_normals=True)
    pts.orient_normals_to_align_with_direction()
    pts.paint_uniform_color([0.5, 0.5, 0.5])

    geoms = [pts]
    geoms.append(h.get_image(start_idx+index, name=f"Distance Field"))
    o3d.visualization.draw(geoms)


if __name__ == "__main__":
    with torch.no_grad():
        main()
