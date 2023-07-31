import os
import torch

CONFIG_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../../../config/'))
DATA_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../../../data/'))
EXPERIMENT_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../../../experiments/'))


def compute_gradient(y: torch.Tensor, x: torch.Tensor, grad_outputs=None):
    """computes dy/dx

    Args:
        y (torch.Tensor): functional
        x (torch.Tensor): positions
        grad_outputs (torch.Tensor): stuff to write output to.

    Returns:
        grad (torch.Tensor): gradient of y wrt x
    """
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(
        y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def get_occ_mask(points, kernel_size=3, nv=[100, 100, 10], device='cuda:0'):
    if points is None:
        return torch.full(nv, True, dtype=torch.bool)
    D = len(nv)
    points = points[..., :D]
    mask = torch.zeros(nv, device=device)
    nv = torch.tensor(nv, device=device)

    idx = torch.tensor(points, device=device)
    idx = idx-idx.min(dim=-2)[0]
    idx /= idx.max(dim=-2)[0]
    idx *= nv
    idx = idx.clamp_max(nv-1).long()

    mask[idx[:, 0], idx[:, 1], idx[:, 2]] = 1
    mask = torch.nn.functional.max_pool3d(
        mask[None, ...], stride=1, kernel_size=kernel_size, padding=kernel_size//2).squeeze()
    return mask.bool()


def compute_T_from_bounding_box(bb):
    """bb, bb[0] = min, bb[1] = max

    Return
        T : Homogenous transformation matrix to local frame
    """
    mins = torch.tensor(bb[0])
    max = torch.tensor(bb[1])
    D = len(mins)  # dimensionality

    T_transl = torch.eye(D+1)
    T_transl[:D, -1] = -mins
    T_scale = torch.eye(D+1)
    T_scale /= (max-mins).max()
    T_scale[-1, -1] = 1
    T_local = T_scale @ T_transl

    return T_local


def transform(p, pose):
    return (pose @ p.transpose(-2, -1)).transpose(-2, -1)
