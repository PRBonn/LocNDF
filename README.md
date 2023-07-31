# LocNDF

LocNDF: Neural Distance Field Mapping for Robot Localization

## Installation

For the installation simply clone the repo and pip install it.

```sh
git clone git@github.com:PRBonn/LocNDF.git
cd LocNDF
pip install .
```

Optionally one can use `pip install -e .` for the editable mode if you plan to change the src code.

## Usage

The following commands are expected to be executed in this root directory.

### Pose tracking in Apollo Southbay

First, get the data:
Download the [Apollo Southbay](https://developer.apollo.auto/southbay.html) dataset and place it in `data/` (or create a symlink). The `ColumbiaPark/` set is enough for the examples.

#### Training one Submap

For training a single model you can configure `config/config.yaml` and run `scripts_pose_tracking/train.py`. 

Registering a scan to the trained model can be done using  `scripts_pose_tracking/register_scan.py` while only visualizing the meshed result one can use `scripts_pose_tracking/mesh_it.py`.

#### Training multiple key-poses

For the training of multiple key-poses you can use the `config/config_mapping.yaml` file and run `scripts_pose_tracking/train.py -c config/config_mapping.yaml`.

Tracking the car pose in the trained submaps can be done using `python3 scripts/pose_tracking.py experiments/PATH-TO-THE-CHECKPOINTS/best-v*.ckpt -vis`.

Pretrained models can be downloaded [here](https://www.ipb.uni-bonn.de/html/projects/locndf/experiments.zip) and should be placed under `/experiments`. Those models can be used as explained above.

### Training on your own data

Most importantly implement your own dataloader. An example can be seen in `src/loc_ndf/datasets/datasets.py`. Second, exchange the dataloader in the training script by you dataloader. Ready to train.

### 2D - MCL

The data can be downloaded [here](https://www.ipb.uni-bonn.de/html/projects/locndf/indoor_scan_poses_2d.zip). The training data consists of `poses.txt` and `scans/*.npy`. The evaluation is done on seq1 to seq5 using the provided scans as well as the `odometry.txt`. The extrected files are expected to be in in `data/`.

#### Training

For training a model you can configure `scripts_mcl/config.yaml` and run `scripts_mcl/train.py`.

After training a model, one can run the MCL example, e.g. (`scripts_mcl/run_mcl.py -c PATH-TO-YOUR_CKPT -i data/indoor_scan_poses_2d/seqX -cal data/indoor_scan_poses_2d/base2laser.txt -o out_poses.txt`) with the trained model.

The Pretrained models can be downloaded [here](https://www.ipb.uni-bonn.de/html/projects/locndf/experiments.zip) and should be placed under `/experiments`.

## Citation

If you use this library for any academic work, please cite the original paper.

```bibtex
@article{wiesmann2023ral,
author = {L. Wiesmann and T. Guadagnino and I. Vizzo and N. Zimmerman and Y. Pan and H. Kuang and J. Behley and C. Stachniss},
title = {{LocNDF: Neural Distance Field Mapping for Robot Localization}},
journal = ral,
volume = {8},
number = {8},
pages = {4999--5006},
year = 2023,
issn = {2377-3766},
doi = {10.1109/LRA.2023.3291274},
codeurl = {https://github.com/PRBonn/LocNDF}
}
```
