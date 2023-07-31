from easydict import EasyDict
from loc_ndf.utils import pytimer
from scipy.spatial.transform import Rotation as R
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import open3d as o3d
import numpy as np
from os.path import join
from loc_ndf.utils import utils


class DataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_set = None

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        data_set = self.get_train_set()
        loader = DataLoader(
            data_set,
            batch_size=self.cfg['train']['batch_size'],
            num_workers=self.cfg['train']['num_workers'],
            shuffle=True)
        return loader

    def get_train_set(self):
        if self.train_set is None:
            self.train_set = get_dataset(self.cfg['data']['train'])
        return self.train_set

#################################################
################## Data loader ##################
#################################################


def interpolate_points(points, center, num=20, log=False, truncation_d=None):
    """points between center and enpoints

    Args:
        points [3]: 
        center [3]: 
        num (int, optional): num_points. Defaults to 20.

    Returns:
        intermediate_points [n x 3]: intermediate_points
    """
    if log:
        alpha = 1 - np.logspace(-1, 0, num, dtype=np.float32)[::-1, None]
        alpha = alpha/alpha.max()
    else:
        alpha = np.linspace(0, 1, num, dtype=np.float32)[:, None]
    if truncation_d:
        dist = np.linalg.norm(points-center)
        if truncation_d < dist:
            alpha = 1-alpha[::-1] * truncation_d / dist

    dists = np.linalg.norm(points-center) * (1-alpha)
    return alpha * points[None, :] + (1-alpha) * center[None, :], dists


def interpolate_points_batch(points, center, num=20, log=False, truncation_d=None):
    """points between center and enpoints

    Args:
        points [Nx3]: 
        center [3]: 
        num (int, optional): num_points. Defaults to 20.

    Returns:
        intermediate_points [n x 3]: intermediate_points
    """
    if log:
        alpha = 1 - np.logspace(-1, 0, num, dtype=np.float32)[::-1, None, None]
        alpha = alpha/alpha.max()
    else:
        alpha = np.linspace(0, 1, num, dtype=np.float32)[:, None, None]
    if truncation_d:
        dist = np.linalg.norm(points-center[None, :], axis=-1)
        if truncation_d < dist:
            alpha = 1-alpha[::-1] * truncation_d / dist[None, :, None]

    dists = np.linalg.norm(
        points-center[None, :], axis=-1)[None, :, None] * (1-alpha)
    return alpha * points[None, :] + (1-alpha) * center[None, None, :], dists


########################################################
############ Aopollo ###################################
########################################################


def readPoses(file):
    data = np.loadtxt(file)
    id, time, t, q = np.split(data, [1, 2, 5], axis=1)
    r = R.from_quat(q).as_matrix()
    pose = np.zeros([r.shape[0], 4, 4])
    pose[:, :3, -1] = t
    pose[:, :3, :3] = r
    pose[:, -1, -1] = 1
    return pose, np.squeeze(id), np.squeeze(time)


def parse_scan(file):
    pcd = o3d.io.read_point_cloud(file)
    return pcd


class ApolloTorch(Dataset):
    def __init__(self, folder, scan_idx,
                 num_scans=10,
                 bb_size=[40, 40, 40],
                 num_inter=200,
                 log=False,
                 truncation_d=None,
                 close_sample_d=0,
                 batch_size=1000) -> None:
        super().__init__()
        self.num_inter = num_inter
        self.log = log
        self.truncation_d = truncation_d
        self.close_sample_d = close_sample_d
        self.bb_size = np.array(bb_size)
        self.scale = self.bb_size.max()
        self.scan_idx = scan_idx
        self.folder = join(utils.DATA_DIR, folder)

        self.points, self.poses, self.idx, self.out_points, self.out_idx = self.load_points(
            self.folder, scan_idx, self.bb_size, num_scans)

        self.min = np.min(self.points, axis=0)
        self.max = np.max(self.points, axis=0)
        self.bounding_box = [self.min.tolist(), self.max.tolist()]
        self.points = np.hstack([self.points, np.ones_like(
            self.points[..., :1])])  # homogenous poop

        # torchify:
        self.batch_size = batch_size
        self.points = torch.tensor(self.points)
        self.poses = torch.tensor(self.poses)

    def __getitem__(self, index):
        ind = torch.tensor(np.random.choice(
            len(self.points), self.batch_size, replace=False))
        point, center = self.get_point_center(ind)

        inter, dists = self.interpolate_points_batch(
            point, center)

        random = torch.rand((self.num_inter//2, 3)) * \
            (self.max-self.min)+self.min
        dr = (torch.rand((self.num_inter//2, 3)) - 0.5) * \
            2 * self.close_sample_d
        random = torch.stack([random, random + dr], dim=-2)
        random = torch.cat(
            [random, torch.ones_like(random[..., :1])], dim=-1)

        out = {'points': point,
               'random': random,
               'inter': inter,
               'center': center,
               'dists': dists}
        return out

    def interpolate_points_batch(self, points, center):
        """points between center and enpoints

        Args:
            points [Nx3]: 
            center [3]: 
            num (int, optional): num_points. Defaults to 20.

        Returns:
            intermediate_points [n x 3]: intermediate_points
        """
        if self.log:
            alpha = 1 - torch.logspace(0, -1, self.num_inter)[:, None, None]
            alpha = alpha/alpha.max()
        else:
            alpha = torch.linspace(0, 1, self.num_inter)[:, None, None]

        dists = torch.norm(
            points-center, dim=-1)[None, :, None] * (1-alpha)
        return alpha * points + (1-alpha) * center[None, :], dists

    def get_min_max(self):
        return self.minz, self.maxz

    def get_point_center(self, index):
        return self.points[index], self.poses[self.idx[index], :, -1]

    def get_outsider(self):
        index = np.random.randint(0, self.out_points.shape[0])
        return self.out_points[index], self.center[self.out_idx[index]]

    def get_points(self):
        return self.points

    def __len__(self):
        return len(self.points)//self.batch_size

    def get_registration_scans(self, scan_folder):
        poses, id, time = readPoses(join(self.folder, 'poses/gt_poses.txt'))
        sposes, sid, stime = readPoses(join(scan_folder, 'poses/gt_poses.txt'))
        ref_pose = poses[self.scan_idx]
        rel_pose = np.linalg.inv(poses[self.scan_idx])

        diff = np.abs((sposes[:, :3, -1] - ref_pose[None, :3, -1]))
        within = (diff < (self.bb_size[None, :]/2)).all(1)

        wposes = sposes[within, :, :]
        wid = sid[within, ...]
        wposes = rel_pose @ wposes
        return wid, wposes, rel_pose

    def get_scans(self, distances):
        distances = np.array(distances)

        poses, id, time = readPoses(join(self.folder, 'poses/gt_poses.txt'))
        wposes, indices = self.get_nearby_poses_evolutional(
            poses, self.scan_idx, self.bb_size)
        rel_pose = np.linalg.inv(poses[self.scan_idx])
        wposes = rel_pose @ wposes
        pose_dist = np.linalg.norm(wposes[:, :3, -1], axis=-1)

        rel_d = np.abs(pose_dist[None, :] - distances[:, None])
        idx = np.argmin(rel_d, axis=-1)

        wposes = wposes[idx]
        indices = indices[idx]

        wposes = wposes.astype(np.float32)

        pcds = []
        for index in indices:
            # index+1 since apollo starts with index 1...
            s = parse_scan(join(self.folder, f'pcds/{str(index+1)}.pcd'))
            p = np.asarray(s.points)
            p = p/self.bb_size + 0.5
            p = p.astype(np.float32)
            pcds.append(p)
        return pcds, wposes, self.bb_size.max()

    @staticmethod
    def get_nearby_poses_evolutional(poses, scan_idx, bb_size):
        wposes = []
        indices = []
        i = scan_idx
        while(True):
            diff = np.abs((poses[i] - poses[scan_idx])[:3, -1])
            within = (diff < (bb_size/2)).all()
            if within:
                wposes.append(poses[i])
                indices.append(i)
                i = i-1
            else:
                break
        i = scan_idx+1
        while(True):
            diff = np.abs((poses[i] - poses[scan_idx])[:3, -1])
            within = (diff < (bb_size/2)).all()
            if within:
                wposes.append(poses[i])
                indices.append(i)
                i = i+1
            else:
                break
        wposes = np.stack(wposes, 0)
        indices = np.array(indices)
        return wposes, indices

    def load_points(self, folder, scan_idx, bb_size, num_scans=10):
        pytimer.tic()
        poses, id, time = readPoses(join(folder, 'poses/gt_poses.txt'))

        # evolutional
        wposes, indices = self.get_nearby_poses_evolutional(
            poses, scan_idx, bb_size)
        # Furthest Pose Sampling
        fps = poses[scan_idx][None, ...]
        fpi = [scan_idx]
        for i in range(num_scans-1):
            diff = ((wposes[None, :, :3, -1] -
                    fps[:, None, :3, -1])**2).sum(-1)
            diff = diff.min(0)
            idx = diff.argmax(0)
            fps = np.concatenate((fps, wposes[None, idx, :, :]))
            fpi.append(indices[idx])

        bb_points = []
        bb_indices = []
        bb_poses = []
        outside_points = []
        outside_indices = []

        rel_pose = np.linalg.inv(poses[scan_idx])
        for idx, i, pose in zip(range(num_scans), fpi, fps):
            s = parse_scan(join(folder, f'pcds/{str(i+1)}.pcd'))
            pose = rel_pose @ pose
            s.transform(pose)
            p = np.asarray(s.points)
            w = (np.abs(p) < (bb_size[None, :]/2)).all(1)  # within bb

            o = p[~w]  # outside points

            p = p[w]

            bb_poses.append(pose)
            bb_points.append(p)
            bb_indices.append(np.full_like(p[:, 0], idx))

            outside_points.append(o)
            outside_indices.append(np.full_like(o[:, 0], idx))

        bb_points = np.concatenate(bb_points, 0).astype(np.float32)
        bb_indices = np.concatenate(bb_indices, 0).astype(np.int64)
        bb_poses = np.stack(bb_poses, 0).astype(np.float32)

        outside_points = np.concatenate(outside_points, 0).astype(np.float32)
        outside_indices = np.concatenate(outside_indices, 0).astype(np.int64)

        # print(bb_points.shape,bb_poses.shape,bb_indices.shape)
        pytimer.tocTic()

        return bb_points, bb_poses, bb_indices, outside_points, outside_indices

#################################################################################################
# Incremental Mapping
#################################################################################################


def get_key_poses(
        num_maps,
        folder,
        start_scan_idx,
        dist,
        overlap_pct):

    poses, id, time = readPoses(join(folder, 'poses/gt_poses.txt'))
    wposes = num_maps*[poses[start_scan_idx]]
    indices = num_maps*[start_scan_idx]

    j = start_scan_idx+1
    for i in range(1, num_maps):
        within = True
        while(within):
            diff = np.linalg.norm((poses[j, :2, -1] - wposes[i-1][:2, -1]))
            within = (diff < (dist*(1-overlap_pct/100)))
            if within:
                wposes[i] = poses[j]
                indices[i] = j
                j += 1
    return wposes, indices


def get_apollo_maps(
        num_maps,
        folder,
        scan_idx,
        num_scans=10,
        bb_size=[50, 50, 50],
        num_inter=200,
        log=False,
        truncation_d=None,
        close_sample_d=0,
        overlap_pct=10):

    wposes, indices = get_key_poses(num_maps=num_maps, folder=folder,
                                    start_scan_idx=scan_idx, dist=max(*bb_size), overlap_pct=overlap_pct)
    datasets = [ApolloTorch(folder=folder,
                            scan_idx=i,
                            num_scans=num_scans,
                            bb_size=bb_size,
                            num_inter=num_inter,
                            log=log,
                            truncation_d=truncation_d,
                            close_sample_d=close_sample_d) for i in indices]
    return datasets, wposes, indices


class Datasets(EasyDict):
    apollo_torch = ApolloTorch


def get_dataset(type_params: dict) -> Dataset:
    """ dict type_params {
        type: str, specifies which loss
        params: dict, kwargs of the loss
    }
    """
    return Datasets()[type_params['type']](**type_params['params'])
