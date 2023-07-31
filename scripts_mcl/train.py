import click
from os.path import join, dirname, abspath
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import yaml

from loc_ndf.mcl2d import models, datasets
from loc_ndf.utils import utils


@click.command()
# Add your options here
@click.option('--config',
              '-c',
              type=str,
              help='path to the config file (.yaml)',
              default=join(dirname(abspath(__file__)), 'config.yaml'))
@click.option('--weights',
              '-w',
              type=str,
              help='path to pretrained weights (.ckpt). Use this flag if you just want to load the weights from the checkpoint file without resuming training.',
              default=None)
@click.option('--checkpoint',
              '-ckpt',
              type=str,
              help='path to checkpoint file (.ckpt) to resume training.',
              default=None)
def main(config, weights, checkpoint):
    cfg = yaml.safe_load(open(config))
    # Load data and model
    data = datasets.DataModule(cfg)
    cfg['bounding_box'] = data.get_train_set().bounding_box
    print(cfg['bounding_box'])

    if weights is None:
        model = models.MCLNet(cfg)
    else:
        model = models.MCLNet.load_from_checkpoint(
            weights, hparams=cfg, strict=False)

    # Add callbacks
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_saver = ModelCheckpoint(monitor='train/loss',
                                       filename='best',
                                       mode='min',
                                       save_last=True)

    tb_logger = pl_loggers.TensorBoardLogger(join(utils.EXPERIMENT_DIR, cfg['experiment']['id']),
                                             default_hp_metric=False)

    trainer = Trainer(accelerator='gpu',
                      devices=cfg['train']['n_gpus'],
                      logger=tb_logger,
                      resume_from_checkpoint=checkpoint,
                      max_epochs=cfg['train']['max_epoch'],
                      callbacks=[lr_monitor, checkpoint_saver])

    # Train!
    trainer.fit(model, data)


if __name__ == "__main__":
    main()
