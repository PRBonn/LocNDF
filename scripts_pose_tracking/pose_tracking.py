import click
import torch
from os.path import join
from loc_ndf.utils import registration, vis, utils
from tqdm import tqdm
from pathlib import Path


@click.command()
# Add your options here
@click.argument('checkpoints',
                nargs=-1,
                required=True)
@click.option('--num_voxels',
              '-v',
              type=int,
              default=400,
              required=True)
@click.option('--threshold',
              '-t',
              type=float,
              default=0.01,
              required=False)
@click.option('--visualize', '-vis', is_flag=True, show_default=True, default=False)
@click.option('--do_test', '-test', is_flag=True, show_default=True, default=False)
def main(checkpoints, num_voxels, threshold, visualize, do_test):

    if do_test:
        folder = join(utils.DATA_DIR,"apollo/TestData/ColumbiaPark/2018-10-11")
        start_idx = 5280
        num_scans = 700
        prefix = 'test'
    else:
        folder = join(utils.DATA_DIR,
                      "apollo/TrainData/ColumbiaPark/2018-10-03")
        start_idx = 6880
        num_scans = 800
        prefix = 'validation'

    tracker = registration.PoseTracker(
        checkpoints=checkpoints,
        test_folder=folder,
        start_idx=start_idx,
        GM_k=0.3,
        max_dist=75,
        num_points=-1,
        nv=num_voxels, 
        threshold=threshold)

    if visualize:
        visulizer = vis.Visualizer(tracker)
        visulizer.run()
    else:
        gt_poses = []
        est_poses = []
        for i in tqdm(range(num_scans)):
            est, gt, _ = tracker.register_next()
            est_poses.append(est)
            gt_poses.append(gt)
        gt_poses = torch.stack(gt_poses)
        est_poses = torch.stack(est_poses)

        dt, dr = registration.pose_error(
            gt_poses, est_poses)
        memory = tracker.get_memory()

        print('Final errors')
        print('AE translation / rotation')
        print(dt, dr)
        print(f"memory {memory:.3f}MB")

        out_dir = Path(checkpoints[0]).parent.parent.absolute().joinpath(
            f'{prefix}_odom_error.txt')
        with open(out_dir, 'w') as f:
            results = f"# dt [m], dr [deg], memory [MB]\n{dt} {dr} {memory}"
            f.write(results)


if __name__ == "__main__":
    main()
