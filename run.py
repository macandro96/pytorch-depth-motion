import os
from copy import deepcopy

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torchinfo import summary

from datamodule import KITTIRAWDatamodule
from engine.trainer import DepthMotionTrainer
from models import build_model


def setup_logging(cfg: DictConfig):
    """Sets up the log directory for the experiment if not specified in the config."""
    if cfg.exp_manager.exp_dir is None:
        with open_dict(cfg.exp_manager):
            cfg.exp_manager.exp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_logs")

    if cfg.exp_manager.create_wandb_logger:
        with open_dict(cfg.exp_manager.wandb_logger_kwargs):
            cfg.exp_manager.wandb_logger_kwargs["save_dir"] = os.path.join(
                cfg.exp_manager.exp_dir, cfg.exp_manager.exp_name
            )
        logger = WandbLogger(**cfg.exp_manager.wandb_logger_kwargs)

    return logger


def setup_dataset(cfg: DictConfig):
    """Build the Data Module."""
    dataset = None
    if cfg.mode in ["training", "validation", "testing"]:
        if cfg.dataset_name == "KITTI":
            dataset = KITTIRAWDatamodule(**cfg.dataset.kitti, **cfg.dataset)
    else:
        raise ValueError(f"Invalid mode: {cfg.mode}")

    return dataset


def setup_model(cfg: DictConfig):
    """Build the Training Module."""
    if cfg.mode in ["training", "validation", "testing"]:
        if cfg.pretrained_name is not None:
            print("Restoring model from checkpoint:", cfg.pretrained_name)
            training_module = DepthMotionTrainer.load_from_checkpoint(
                cfg.pretrained_name,
                model=build_model(cfg.pretrained_name),
                **cfg.exp_manager.logging,
            )
        else:
            training_module = DepthMotionTrainer(
                model=build_model(cfg.model),
                optimizer_cfg=cfg.model.optimizer,
                monitor_val=cfg.exp_manager.checkpoint_callback_params.monitor,
                **cfg.exp_manager.logging,
            )
    else:
        raise ValueError(f"Invalid mode: {cfg.mode}")

    return training_module


def setup_callbacks(trainer_cfg, exp_manager_cfg):
    """Sets up the checkpointing callback for the trainer."""
    callbacks = [LearningRateMonitor()]

    if trainer_cfg.enable_checkpointing:
        callbacks.append(
            ModelCheckpoint(
                dirpath=os.path.join(exp_manager_cfg.exp_dir, exp_manager_cfg.exp_name, "checkpoints"),
                filename="vl-{epoch}-{val_loss:.2f}",
                monitor=exp_manager_cfg.checkpoint_callback_params.monitor,
                mode="min",
                save_last=exp_manager_cfg.checkpoint_callback_params.save_last,
                save_top_k=exp_manager_cfg.checkpoint_callback_params.save_top_k,
            )
        )

    return callbacks


def grid_search(cfg):
    """Grid search over the hyperparameters.
    The current grid search is over the following hyperparameters:
        - motion_smoothness_weight
        - motion_sparsity_weight
        - dataset_stride
        - scheduler
    """
    grid_space = []
    exp_name = cfg.exp_manager.exp_name
    for motion_smoothness_weight in [3.0, 2.0, 1.0]:
        for motion_sparsity_weight in [5e-2, 0.1, 0.2]:
            for layernorm in [True, False]:
                for scheduler_type in [None, "CosineAnnealing", "StepLR"]:
                    config_current = deepcopy(cfg)

                    # update hydraconfig according to grid search params
                    config_current.model.loss_weights.motion_smoothing = motion_smoothness_weight
                    config_current.model.loss_weights.motion_sparsity = motion_sparsity_weight
                    config_current.model.depth_model.use_layernorm = layernorm
                    config_current.model.optimizer.scheduler.name = scheduler_type
                    config_current.exp_manager.exp_name = f"{exp_name}_{len(grid_space)}"
                    config_current.exp_manager.wandb_logger_kwargs.name = config_current.exp_manager.exp_name

                    grid_space.append(config_current)
    return grid_space


@hydra.main(version_base=None, config_path=os.path.dirname(os.path.abspath(__file__)), config_name="config-local.yaml")
def main(cfg):
    # trunk-ignore(bandit/B101)
    assert cfg.mode == "training" or (
        cfg.mode != "training" and cfg.pretrained_name is not None and os.path.exists(cfg.pretrained_name)
    ), f"Must specify a valid pretrained model checkpoint for {cfg.mode}."

    seed_everything(42, workers=True)

    if cfg.grid.index > -1:
        cfg = grid_search(cfg)[cfg.grid.index]

    logger = setup_logging(cfg)

    print(f"Hydra config:\n{OmegaConf.to_yaml(cfg)}")

    print("Building dataset...")
    dataset = setup_dataset(cfg)

    print("Creating model...")
    training_module = setup_model(cfg)
    summary(training_module.model)

    # Create trainer
    print("Creating trainer...")
    trainer = Trainer(
        **cfg.trainer,
        default_root_dir=os.path.join(cfg.exp_manager.exp_dir, cfg.exp_manager.exp_name),
        logger=logger,
        callbacks=setup_callbacks(cfg.trainer, cfg.exp_manager),
    )
    if cfg.mode == "training":
        print("Training model...")
        trainer.fit(training_module, datamodule=dataset, ckpt_path=cfg.pretrained_name)

    elif cfg.mode == "validation":
        print("Validating model...")
        training_module.eval()
        trainer.validate(model=training_module, datamodule=dataset, ckpt_path=cfg.pretrained_name)

    elif cfg.mode == "testing":
        print("Testing model...")
        training_module.eval()
        trainer.test(model=training_module, datamodule=dataset, ckpt_path=cfg.pretrained_name)


if __name__ == "__main__":
    main()
