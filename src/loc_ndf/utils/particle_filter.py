import torch


def exp(vec):
    T = torch.zeros(vec.shape[0], 3, 3, device=vec.device)
    T[:, 0, 0] = torch.cos(vec[:, -1])
    T[:, 0, 1] = -torch.sin(vec[:, -1])
    T[:, 1, 0] = torch.sin(vec[:, -1])
    T[:, 1, 1] = torch.cos(vec[:, -1])
    T[:, :2, -1] = vec[:, :2]
    T[:, 2, 2] = torch.ones(vec.shape[0], device=vec.device)
    return T


def log(T):
    v = torch.zeros(T.shape[0], 3, device=T.device)
    v[:, :2] = T[:, :2, -1]
    v[:, -1] = torch.atan2(T[:, 1, 0], T[:, 0, 0])
    return v


class ParticleFilter:
    def __init__(
        self,
        num_particles: int,
        bounding_box: torch.Tensor,
        output_file: str,
        extrinsic: torch.Tensor,
        device=torch.device("cuda"),
        min_weight=1e-8,
        min_rotation_update=0.1,
        min_translation_update=0.05,
        likelihood_gain=100
    ):
        self.device = device
        self.inverse_extrinsic = torch.linalg.inv(extrinsic)
        self.particles = torch.ones(num_particles, 4, device=self.device)
        self.particles[:, :2] = (
            torch.rand(num_particles, 2, device=self.device)
            * (bounding_box[1] - bounding_box[0])
            + bounding_box[0]
        )
        self.particles[:, 2] = (
            2 * torch.pi * torch.rand(num_particles, device=self.device)
        )
        self.num_particles = num_particles
        self.noise_coefficients = (
            torch.tensor(
                [
                    [0.01, 0.0005, 0.0002],
                    [0.0005, 0.0001, 0.0001],
                    [0.001, 0.00001, 0.05],
                ],
                device=self.device,
            )
            * 50
        )
        self.min_weight = min_weight
        self.min_rotation_update = min_rotation_update
        self.min_translation_update = min_translation_update
        self.likelihood_gain = likelihood_gain
        self.cumulative_translation = 0.0
        self.cumulative_rotation = 0.0
        self.min_valid_points = 30
        self.num_particles_tracking = 10000

        self.pose_tracking_std = 0.3
        self.traj_file = open(output_file, "w")

    def resample_outside_points(self, model, bb_box):
        num_invalids = 1e8
        while num_invalids > 1000:
            p_local = self.particles[:, :3].clone()
            p_local[..., -1] = 1
            valid = model.is_inside(p_local)
            num_invalids = (~valid).sum()
            self.particles[~valid, :2] = (
                torch.rand(num_invalids, 2, device=model.device) *
                (bb_box[1] - bb_box[0])
                + bb_box[0]
            )

    def is_in_pose_tracking(self):
        return self.num_particles == self.num_particles_tracking

    def mean(self):
        xy = self.particles[:, :2]
        theta = self.particles[:, 2]
        mu = torch.zeros(1, 3, device=self.device)
        weights = self.particles[:, -1]
        mu[:, :2] = torch.sum(xy * weights[:, None], dim=0) / weights.sum()
        mu[:, 2] = torch.atan2(
            (torch.sin(theta) * weights).sum(), (torch.cos(theta) * weights).sum()
        )
        return exp(mu).squeeze()

    def apply_motion_model(self, odometry: torch.tensor):
        scales = torch.abs(odometry)
        std_deviations = self.noise_coefficients @ scales
        control = odometry + std_deviations * torch.randn(
            self.num_particles, 3, device=self.device
        )
        assert control.shape[0] == self.num_particles
        new_poses = exp(self.particles[:, :3]) @ exp(control)
        self.particles[:, :3] = log(new_poses)

        self.cumulative_translation += torch.linalg.norm(odometry[:2])
        self.cumulative_rotation += torch.abs(odometry[-1])

    def get_pose(self):
        if self.is_in_pose_tracking():
            return self.mean()
        else:
            most_likely = torch.argmax(self.particles[:, -1])
            return exp(self.particles[None, most_likely, :3]).squeeze()

    def resample(self, weights):
        std = torch.std(self.particles[:, :2], dim=0).norm()
        if std < self.pose_tracking_std:
            self.num_particles = self.num_particles_tracking
        indices = list(
            torch.utils.data.WeightedRandomSampler(weights, self.num_particles)
        )
        self.particles = self.particles[indices, :]

    def write_pose(self):
        """
        write the current pose estimate with position standard deviation
        we write everything in the base_link of the robot
        """
        std = torch.std(self.particles[:, :2], dim=0).norm().cpu().numpy()
        v = log((self.mean() @ self.inverse_extrinsic)
                [None, ...])[0].cpu().numpy()
        string = str(v[0]) + " " + str(v[1]) + " " + \
            str(v[2]) + " " + str(std) + "\n"
        self.traj_file.write(string)

    def scan_in_particles_frame(self, scan: torch.tensor) -> torch.tensor:
        assert scan.shape[-1] == 3, "ohh no, its not homogenous"
        particles_scans = exp(self.particles[:, :3]) @ scan.T
        particles_scans = particles_scans.permute(0, 2, 1)
        return particles_scans

    def has_moved_enough(self):
        return (
            self.cumulative_translation > self.min_translation_update
            or self.cumulative_rotation > self.min_rotation_update
        )

    def apply_observation_model(self, distances: torch.tensor):
        if not self.has_moved_enough():
            return
        self.cumulative_rotation = 0.0
        self.cumulative_translation = 0.0
        assert distances.shape[0] == self.particles.shape[0]
        mask_distances = distances >= 0.0
        num_valid_points = torch.sum(mask_distances, dim=1)
        mask_invalid_observations = num_valid_points < self.min_valid_points
        log_harmonic_mean = torch.sum(distances * mask_distances, dim=1) / (
            num_valid_points + 1e-8
        )
        weights = torch.exp(-self.likelihood_gain *
                            log_harmonic_mean) + self.min_weight
        weights[mask_invalid_observations] = self.min_weight
        weights /= torch.sum(weights)
        self.particles[:, -1] = weights
        self.resample(self.particles[:, 3])
