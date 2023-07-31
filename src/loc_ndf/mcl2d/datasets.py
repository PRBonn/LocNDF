from loc_ndf.utils import utils
import tqdm
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import numpy as np
from os.path import join


class DataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_set = None

    def train_dataloader(self):
        data_set = self.get_train_set()
        loader = DataLoader(
            data_set,
            batch_size=self.cfg['train']['batch_size'],
            num_workers=self.cfg['train']['num_workers'],
            shuffle=True)
        return loader

    def val_dataloader(self):
        data_set = MCLDataset(self.cfg['data']['val'])
        loader = DataLoader(
            data_set,
            num_workers=self.cfg['train']['num_workers'],
            batch_size=1)
        return loader

    def test_dataloader(self):
        data_set = MCLDataset(self.cfg['data']['val'])
        loader = DataLoader(data_set,
                            num_workers=self.cfg['train']['num_workers'],
                            batch_size=1)
        return loader

    def get_train_set(self):
        if self.train_set is None:
            self.train_set = MCLDataset(**self.cfg['data']['train']['params'])
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


#######################################################################
#################### Abstract Class ###################################
#######################################################################


class MCLDataset(Dataset):
    def __init__(self, folder, num_inter, log=False, truncation_d=None, close_sample_d=0) -> None:
        super().__init__()
        self.num_inter = num_inter
        self.log = log
        self.truncation_d = truncation_d
        self.close_sample_d = close_sample_d
        self.folder = join(utils.DATA_DIR, folder)

        self.points, self.poses, self.indices = self.load_points(self.folder)
        self.min = np.min(self.points, axis=0)
        self.max = np.max(self.points, axis=0)
        self.bounding_box = [self.min.tolist(), self.max.tolist()]
        self.points = np.hstack([self.points, np.ones_like(
            self.points[..., :1])])  # homogenous poop

    def load_points(self, folder):

        poses_ = np.loadtxt(join(folder, 'poses.txt'))
        poses = R.from_euler('z', poses_[:, -1], degrees=False).as_matrix()
        poses[:, :2, -1] = poses_[:, :2]

        points = []
        indices = []
        for idx, pose in enumerate(tqdm.tqdm(poses)):
            if (idx % 10):
                continue
            scan = np.load(f'{folder}/scans/{str(idx).zfill(6)}.npy')[:2, :].T
            scan = (pose[:2, :2] @ scan.T).T+pose[:2, -1]
            points.append(scan)
            indices.append(np.full_like(scan[:, 0], idx))
        points = np.concatenate(points, 0).astype(np.float32)
        indices = np.concatenate(indices, 0).astype(np.int64)
        poses = np.stack(poses, 0).astype(np.float32)
        return points, poses, indices

    def get_point_center(self, index):
        point = self.points[index]
        center = self.poses[self.indices[index], :, -1]
        return point, center

    def __len__(self):
        return len(self.points)

    def get_points(self):
        return self.points

    def __getitem__(self, index):
        point, center = self.get_point_center(index)
        inter, dists = interpolate_points(
            point, center, self.num_inter, self.log, self.truncation_d)

        random = np.random.rand(self.num_inter//2, 2) * \
            (self.max-self.min)+self.min
        dr = (np.random.rand(self.num_inter//2, 2) - 0.5) * \
            2 * self.close_sample_d
        random = np.stack([random, random + dr], axis=-2)
        random = random.astype(np.float32)
        random = np.concatenate(
            [random, np.ones_like(random[..., :1])], axis=-1)

        out = {'points': point,
               'random': random,
               'inter': inter,
               'center': center,
               'dists': dists}
        return out
