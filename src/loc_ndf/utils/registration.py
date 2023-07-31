from loc_ndf.utils import pytimer
from tqdm import tqdm
import numpy as np
import torch
from loc_ndf.utils import utils
from loc_ndf.models import models
from os.path import join
from loc_ndf.datasets import datasets
import os


class PoseTracker:
    def __init__(self, checkpoints: list, test_folder, start_idx, max_dist=50, num_points=20000, GM_k=None, nv=200, num_iter=20, threshold=0.01) -> None:
        self.device = 'cuda'

        self.models = [models.LocNDF.load_from_checkpoint(
            ckpt, hparams=torch.load(ckpt)['hyper_parameters']).to(device=self.device) for ckpt in checkpoints]
        self.test_folder = test_folder
        # self.cache = cache.get_cache(directory=utils.DATA_DIR)

        self.max_dist = max_dist
        self.num_points = num_points
        self.GM_k = GM_k
        self.num_iter = num_iter

        self.rel_pose = None
        self.poses = []
        self.meshes = []
        self.times = []
        for model in self.models:
            pose = torch.tensor(
                model.hparams['data']['pose'], device=self.device, dtype=torch.double).reshape(4, 4)
            if self.rel_pose is None:
                self.rel_pose = torch.linalg.inv(pose)
            self.poses.append((self.rel_pose @ pose).to(device=self.device))

        init_pose_file = join(test_folder, "poses/init_poses.txt")
        if os.path.isfile(init_pose_file):
            self.init_scan_poses = self.rel_pose @ torch.tensor(datasets.readPoses(
                init_pose_file)[0], device=self.device, dtype=torch.double)
        else:
            self.init_scan_poses = self.rel_pose @ torch.tensor(datasets.readPoses(
                join(test_folder, "poses/gt_poses.txt"))[0], device=self.device, dtype=torch.double)
        self.gt_scan_poses = self.rel_pose @ torch.tensor(datasets.readPoses(
            join(test_folder, "poses/gt_poses.txt"))[0], device=self.device, dtype=torch.double)

        self.gt_scan_poses = self.gt_scan_poses
        self.init_scan_poses = self.init_scan_poses

        self.pose = self.init_scan_poses[start_idx]
        self.constant_velocity = torch.eye(
            4, device=self.device, dtype=torch.double)

        self.running_idx = start_idx
        self.get_meshes(nv=[nv, nv, nv//10], tau=threshold)
        self.est_poses = []

    def get_meshes(self, nv=[500, 500, 50], tau=0.01):
        if len(self.meshes) == 0:
            for i, model in enumerate(tqdm(self.models)):
                mesh = model.get_mesh(nv,
                                      mask=model.get_occupancy_mask(
                                          nv).cpu().numpy(), tau=tau
                                      )  # ,file=file)
                mesh.transform(self.poses[i].cpu().numpy())
                self.meshes.append(mesh)
        return self.meshes

    def getGeometries(self, i):
        current_pose, gt_pose, scan = self.register_next()

        scan.transform(current_pose.cpu().numpy())
        scan.paint_uniform_color([154/256, 0, 0])

        return self.get_meshes() + [scan]

    def register_next(self):
        scan = datasets.parse_scan(
            join(self.test_folder, f"pcds/{self.running_idx+1}.pcd"))

        points = np.asarray(scan.points)
        points = torch.tensor(
            points, dtype=self.models[0].dtype, device=self.models[0].device)
        points = points[points.norm(dim=-1) < self.max_dist]  # max distance
        points = torch.cat([points, torch.ones_like(points[:, :1])], dim=-1)

        pytimer.tic()
        current_pose = self.register_scan(
            points, num_iter=self.num_iter, initial_guess=self.constant_velocity @ self.pose)
        self.times.append(pytimer.toc('registration', verbose=False))

        self.constant_velocity = current_pose @ torch.linalg.inv(
            self.pose) if len(self.est_poses) > 0 else torch.eye(4).to(current_pose)
        self.pose = current_pose
        print("avg icp time:", np.mean(self.times))

        dt, dr = pose_error(current_pose, self.gt_scan_poses[self.running_idx])
        print(f'dt: {dt:0.3}m, dr: {dr:0.3}deg')

        self.est_poses.append(current_pose.detach().cpu().numpy())
        self.running_idx += 1
        return current_pose, self.gt_scan_poses[self.running_idx-1], scan

    def forward(self, points):
        """ returns distance for each point
        Args:
            points (n,3): xyz
        """
        poses = torch.stack(self.poses)
        dists = torch.sum(
            (points[:, None, :3] - poses[None, :, :3, -1])**2, dim=-1)
        closest_pose_idx = dists.argmin(-1)

        valid_dists = []
        valid_points = []
        gradients = []
        for i in range(len(self.poses)):
            model = self.models[i]
            idx_i = closest_pose_idx == i

            points_i = points[idx_i]
            points_local = (torch.linalg.inv(poses[i]).to(
                dtype=points_i.dtype) @ points_i.T).T
            bb = torch.tensor(
                model.hparams['bounding_box'], device=points.device)
            within = ((points_local[..., :3] >= bb[0]) & (
                points_local[..., :3] < bb[1])).all(-1)
            p = points_i
            d = model(points_local)

            grads = utils.compute_gradient(d, p)[..., :3]
            valid_points.append(p[within])
            valid_dists.append(d[within])
            gradients.append(grads[within])

        valid_dists = torch.cat(valid_dists).detach()
        valid_points = torch.cat(valid_points).detach()
        gradients = torch.cat(gradients).detach()
        return valid_points, valid_dists, gradients

    def register_scan(self, points, num_iter, initial_guess=None):
        if initial_guess is None:
            T = torch.eye(
                4, device=points.device, dtype=torch.float64)
        else:
            T = initial_guess  # to local frame
        points = points.detach().T
        for _ in range(num_iter):
            points_t = (T.to(dtype=points.dtype) @ points).T
            DT = self.registration_step(points_t, GM_k=self.GM_k).detach()
            T = DT@T

            change = torch.acos(
                (torch.trace(DT[:3, :3])-1)/2) + DT[:3, -1].norm()
            if change < 1e-4:
                break
        return T

    def registration_step(self, points: torch.Tensor, GM_k=None):
        if points.shape[0] < 10:
            return torch.eye(4, device=points.device, dtype=torch.float64)
        points.requires_grad_(True)

        points, distances, gradients = self.forward(points)
        grad_norm = gradients.norm(dim=-1, keepdim=True)
        gradients = gradients/grad_norm
        distances = distances/grad_norm

        T = df_icp(points[..., :3], gradients, distances, GM_k=GM_k)
        return T

    def get_memory(self):
        param_size = 0

        for model in self.models:
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
            buffer_size = 0
            for buffer in model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()
        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb
##############################################################################
########################### Functions ########################################
##############################################################################


def df_icp(points, gradients, distances, GM_k=None):
    w = 1 if GM_k is None else GM_k/(GM_k+distances**2)**2

    cross = torch.cross(points, gradients)
    J = torch.cat([gradients, cross], -1)
    N = J.T @ (w*J)
    g = -(J*w).T @ distances
    t = torch.linalg.inv(N.to(dtype=torch.float64)
                         ) @ g.to(dtype=torch.float64).squeeze()

    T = torch.eye(4, device=points.device, dtype=torch.float64)
    T[:3, :3] = expmap(t[3:])
    T[:3, -1] = t[:3]
    return T


def registration_step(points: torch.Tensor, distf):
    if points.shape[0] < 10:
        return torch.eye(4, device=points.device, dtype=points.dtype)
    points.requires_grad_(True)

    distances = distf(points)

    gradients = utils.compute_gradient(distances, points).detach()[..., :3]
    grad_norm = gradients.norm(dim=-1, keepdim=True)

    gradients = gradients/grad_norm
    distances = distances/grad_norm

    T = df_icp(points[..., :3], gradients, distances)
    return T


def skew(v):
    S = torch.zeros(3, 3, device=v.device, dtype=v.dtype)
    S[0, 1] = -v[2]
    S[0, 2] = v[1]
    S[1, 2] = -v[0]
    return S - S.T


def expmap(axis_angle: torch.Tensor):
    angle = axis_angle.norm()
    axis = axis_angle/angle
    eye = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    S = skew(axis)
    R = eye + angle.sin()*S + (1-angle.cos())*(S@S)
    return R


def arccos(x):
    if torch.is_tensor(x):
        return x.arccos()
    else:
        return np.arccos(x)


def batch_trace(x):
    return x.diagonal(0, -1, -2).sum(-1)


def pose_error(T1, T2, scale=1, dim=3, reduce=True):
    dt = ((T1[..., :dim, -1] - T2[..., :dim, -1])**2).sum(-1)**0.5 * scale
    dr = arccos(((batch_trace(
        T1[..., :dim, :dim].transpose(-1, -2) @ T2[..., :dim, :dim]) - 1)/2).clip(-1, 1))/np.pi*180
    if reduce:
        return dt.mean(), dr.mean()
    else:
        return dt, dr


def registration(points, model, num_iter, initial_guess=None):
    T = torch.eye(
        4, device=points.device) if initial_guess is None else initial_guess

    for _ in range(num_iter):
        points_t = (T @ points.detach().T).T
        bb = torch.tensor(model.hparams['bounding_box'], device=points.device)
        within = ((points[..., :3] >= bb[0]) & (
            points[..., :3] < bb[1])).all(-1)
        DT = registration_step(points_t[within], model).detach()
        T = DT@T
    return T
