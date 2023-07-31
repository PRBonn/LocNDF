import click
from os.path import join
import subprocess
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import yaml

import loc_ndf.datasets.datasets as datasets
import loc_ndf.models.models as models
import tqdm
from loc_ndf.utils import utils


@click.command()
# Add your options here
@click.option('--config',
              '-c',
              type=str,
              help='path to the config file (.yaml)')
@click.option('--weights',
              '-w',
              type=str,
              help='path to pretrained weights (.ckpt). Use this flag if you just want to load the weights from the checkpoint file without resuming training.',
              default=None)
def main(config, weights):
    cfg = yaml.safe_load(open(config))
    cfg['git_commit_version'] = str(subprocess.check_output(
        ['git', 'rev-parse', '--short', 'HEAD']).strip())
    folder = join(utils.DATA_DIR, cfg['data']['train']['params']['folder'])
    poses, indices = datasets.get_key_poses(num_maps=cfg['data']['num_maps'],
                                            folder=folder,
                                            start_scan_idx=cfg['data']['train']['params']['scan_idx'],
                                            dist=cfg['data']['train']['params']['bb_size'][0],
                                            overlap_pct=cfg['data']['overlap_pct'])

    version = None
    for i, pose, idx in zip(range(len(poses)), tqdm.tqdm(poses, 'Maps'), indices):
        # Load data and model
        cfg['data']['pose'] = pose.flatten().tolist()
        cfg['data']['train']['params']['scan_idx'] = idx
        data = datasets.DataModule(cfg)
        cfg['bounding_box'] = data.get_train_set().bounding_box

        points = data.get_train_set().get_points()
        if weights is None:
            model = models.LocNDF(cfg, points=points)
        else:
            model = models.LocNDF.load_from_checkpoint(
                weights, points=points, strict=False, hparams=cfg)
        model.update_map_params(data.get_train_set().points)

        # Add callbacks
        lr_monitor = LearningRateMonitor(logging_interval='step')
        checkpoint_saver = ModelCheckpoint(monitor='train/loss',
                                           filename=f'best-v{str(i)}',
                                           mode='min',
                                           save_last=True)

        tb_logger = pl_loggers.TensorBoardLogger(join(utils.EXPERIMENT_DIR, cfg['experiment']['id']),
                                                 sub_dir=str(i),
                                                 default_hp_metric=False,
                                                 version=version)

        # Setup trainer
        trainer = Trainer(accelerator='gpu',
                          devices=cfg['train']['n_gpus'],
                          logger=tb_logger,
                          max_epochs=cfg['train']['max_epoch'],
                          callbacks=[lr_monitor, checkpoint_saver])

        # Train!
        trainer.fit(model, data)

        # init parameters with last trained model
        weights = join(tb_logger.log_dir, f'../checkpoints/best-v{i}.ckpt')
        version = tb_logger.version


if __name__ == "__main__":
    main()
