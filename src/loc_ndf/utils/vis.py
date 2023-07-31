import torch
import open3d as o3d
import numpy as np
from skimage import measure
import time

BACKGROUND = [0, 0, 0]


def torch2o3d(pcd, colors=None, normals=None, estimate_normals=False):
    pcd = pcd.detach().cpu().squeeze().numpy() if isinstance(pcd, torch.Tensor) else pcd

    assert len(pcd.shape) <= 2, "Batching not implemented"
    colors = colors.detach().cpu().squeeze().numpy() if isinstance(
        colors, torch.Tensor) else colors
    normals = normals.detach().cpu().squeeze().numpy() if isinstance(
        normals, torch.Tensor) else normals

    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(pcd[:, :3])
    if estimate_normals:
        pcl.estimate_normals()
    if colors is not None:
        pcl.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        pcl.normals = o3d.utility.Vector3dVector(normals)
    return pcl


def grid_to_mesh(grid, tau=0.5, translate=None, ascent=False, mins=None, maxs=None, grid_size=None, mask=None):
    maxs = np.array(maxs)
    mins = np.array(mins)
    direction = 'ascent' if ascent else 'descent'

    min_v = grid[mask].min()
    max_v = grid[mask].max()
    tau = tau if ((tau > min_v) and (tau < max_v)) else (max_v+min_v)/2

    verts, faces, normals, values = measure.marching_cubes(
        grid, tau, gradient_direction=direction, mask=mask)
    if grid_size is None:
        grid_size = grid.shape

    # Marching cubes returns in a weird coordinate system...
    verts += 0.5
    verts /= np.array(grid_size)
    if mins is not None:
        verts *= (maxs-mins)
        verts += mins
    if translate is not None:
        verts += translate

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.vertex_colors = o3d.utility.Vector3dVector(-normals/3+0.5)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    return mesh


def get_occupancy_grid(file, nv,):
    p = o3d.io.read_point_cloud(file)
    # o3d.visualization.draw_geometries([p])

    x = np.asarray(p.points)
    mi, ma = x.min(0), x.max(0)
    x = (x-mi)/(ma-mi+1e-5).max()
    mi, ma = x.min(0), x.max(0)

    grid = np.zeros([nv, nv, nv])
    coords = (x*nv).astype('int')
    for p in coords:
        grid[p[0], p[1], p[2]] = 1
    return grid


class Visualizer():
    def __init__(self, point_cloud_provider, width=1920, height=1080):
        self.i = 0
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.register_key_callback(ord("N"), self.next)
        self.vis.register_key_callback(
            ord("K"),  self.change_background_to_black)
        self.vis.register_key_callback(ord("S"), self.start_prev)
        self.vis.register_key_callback(ord("X"), self.stop_prev)
        self.vis.register_key_callback(ord("Z"), self.save)

        self.stop = False
        self.clouds = point_cloud_provider
        self.vis.create_window(width=width, height=height)

        self.render = False
        self.image_files = []

    def save(self, vis):
        for i, g in enumerate(self.geoms):
            o3d.io.write_point_cloud(f"{i}.ply", g)

    def updatePoints():
        pass

    def change_background_to_black(self, vis):
        opt = vis.get_render_option()
        opt.background_color = BACKGROUND
        return False
        # self.

    def next(self, vis=None):
        self.vis.clear_geometries()
        self.geoms = self.clouds.getGeometries(self.i)
        for g in self.geoms:
            self.vis.add_geometry(g, reset_bounding_box=(self.i == 0))
        self.i += 1

    def start_prev(self, vis=None):
        self.stop = False
        while not self.stop:
            self.next()
            self.vis.poll_events()
            time.sleep(0.1)

    def stop_prev(self, vis=None):
        self.stop = True

    def run(self):
        print('N: next')
        print('S: start')
        print('X: stop')
        print('R: render video')
        self.next()
        self.vis.run()
