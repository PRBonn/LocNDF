import torch
from pytorch_lightning.core.module import LightningModule
from loc_ndf.models import loss, models
from loc_ndf.utils import utils
import math


class MCLNet(LightningModule):
    def __init__(self, hparams: dict):
        super().__init__()
        # name you hyperparameter hparams, then it will be saved automagically.
        self.save_hyperparameters(hparams)

        self.model = models.get_network(hparams["model"])
        self.loss = loss.ProjectedDistanceLoss(**hparams['loss']['params'])
        self.register_buffer(
            "occupancy_mask",
            torch.zeros(hparams["occupancy_mask"]["nv"], dtype=torch.bool),
        )

        T_local = utils.compute_T_from_bounding_box(hparams["bounding_box"])
        self.register_buffer("T_local", T_local)

    def forward(self, x: torch.Tensor, in_global_frame=True):
        if in_global_frame:
            x = utils.transform(x, self.T_local)
        distance = self.model(x)
        return distance

    def is_inside(self, x: torch.Tensor, in_global_frame=True):
        if in_global_frame:
            x = utils.transform(x, self.T_local)
        inside = torch.all((x >= 0) & (x <= 1.0), dim=-1)
        rc = x[..., :2] * \
            torch.tensor(self.occupancy_mask.shape, device=x.device)
        rc = rc.long().clamp(
            min=torch.tensor([0, 0], device=x.device, dtype=torch.long),
            max=torch.tensor(
                self.occupancy_mask.shape, device=x.device, dtype=torch.long
            )
            - 1,
        )
        valid = self.occupancy_mask[rc[..., 0], rc[..., 1]]
        valid = valid & inside
        return valid

    def compute_distance(self, x: torch.Tensor, batch_size=None, in_global_frame=True):
        if batch_size is None:
            return self.forward(x, in_global_frame=in_global_frame), self.is_inside(
                x, in_global_frame=in_global_frame
            )
        N = x.shape[0]
        B = batch_size
        distance = torch.cat(
            [
                self.forward(
                    x[i * B: min((i + 1) * B, N)
                      ], in_global_frame=in_global_frame
                )
                for i in range(math.ceil(N / B))
            ],
            dim=0,
        )
        return distance, self.is_inside(x, in_global_frame=in_global_frame)

    def on_train_start(self) -> None:
        self.occupancy_mask = torch.zeros(
            self.hparams["occupancy_mask"]["nv"], dtype=torch.bool
        )

    def training_step(self, batch: dict, batch_idx):
        points = self.forward(batch["points"])

        inter_grad, inter_val = self.compute_gradient(batch["inter"])
        rand_grad, rand_val = self.compute_gradient(batch["random"])
        loss, losses = self.loss(
            points_distance=points,
            points=batch["points"],
            inter_val=inter_val,
            inter_grad=inter_grad,
            inter_pos=batch["inter"],
            ray_dists=batch["dists"],
            rand_grad=rand_grad,
        )

        # update occupancy grid
        inter_local = utils.transform(batch["inter"], self.T_local)
        r = (
            (inter_local[..., 0] * self.occupancy_mask.shape[0])
            .long()
            .clamp(min=0, max=self.occupancy_mask.shape[0] - 1)
            .flatten()
        )
        c = (
            (inter_local[..., 1] * self.occupancy_mask.shape[1])
            .long()
            .clamp(min=0, max=self.occupancy_mask.shape[1] - 1)
            .flatten()
        )
        self.occupancy_mask[r, c] = True

        for k, v in losses.items():
            self.log(f"train/{k}", v)
        return loss

    def compute_gradient(self, position: torch.Tensor):
        position.requires_grad_(True)
        val = self.forward(position)
        grad = utils.compute_gradient(val, position)
        return grad, val

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams["train"]["lr"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams["train"]["max_epoch"],
            eta_min=self.hparams["train"]["lr"] / 1e3,
        )
        return [optimizer], [scheduler]

    def get_grid(self, nv=[400, 400]):
        x = torch.linspace(0, 1.0, nv[0], device=self.device)
        y = torch.linspace(0, 1.0, nv[1], device=self.device)
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        grid = torch.stack([xx, yy], -1)
        return grid

    def get_memory(self):
        param_size = 0

        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb
