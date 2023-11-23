from collections import OrderedDict
from typing import Union

import torch
from omegaconf import DictConfig, OmegaConf

from models.models_depth_motion import DepthMotionModel


def build_model(model_cfg: Union[DictConfig, str]):
    checkpoint = None
    motion_burnin_start_step = 0
    if isinstance(model_cfg, str):
        # Load from checkpoint
        checkpoint = torch.load(model_cfg, map_location="cpu")
        model_cfg = OmegaConf.create(checkpoint["model_config"])

        motion_burnin_start_step = checkpoint["steps"]

    loss_weights = {
        "depth_supervision": model_cfg.loss_weights.depth_supervision,
        "depth_variance": model_cfg.loss_weights.depth_variance,
        "depth_smoothing": model_cfg.loss_weights.depth_smoothing,
        "rgb_consistency": model_cfg.loss_weights.rgb_consistency,
        "depth_consistency": model_cfg.loss_weights.depth_consistency,
        "ssim": model_cfg.loss_weights.ssim,
        "rotation_cycle_consistency": model_cfg.loss_weights.rotation_cycle_consistency,
        "translation_cycle_consistency": model_cfg.loss_weights.translation_cycle_consistency,
        "motion_smoothing": model_cfg.loss_weights.motion_smoothing,
        "motion_sparsity": model_cfg.loss_weights.motion_sparsity,
    }

    model = DepthMotionModel(
        loss_weights=loss_weights,
        motion_burnin_steps=model_cfg.motion_burnin_steps,
        depth_model_args=model_cfg.depth_model,
        motion_model_args=model_cfg.motion_model,
        motion_burnin_start_step=motion_burnin_start_step,
    )

    if checkpoint is not None:
        model.load_state_dict(
            OrderedDict(
                [(k.replace("model.", "", 1), v) for k, v in checkpoint["state_dict"].items() if k.startswith("model.")]
            )
        )

    return model
