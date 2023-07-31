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

Pretrained models can be downloaded !here! and should be placed under `/experiments`. Those models can be used as explained above.

### Training on your own data
Most importantly implement your own dataloader. An example can be seen in `src/loc_ndf/datasets/datasets.py`. Second, exchange the dataloader in the training script by you dataloader. Ready to train.

## Todos:
[ ] Provide data and documentation for MCL