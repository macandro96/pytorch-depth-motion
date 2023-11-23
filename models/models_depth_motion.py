import torch
import torch.nn as nn

from models.depth.models_depth import DepthModel
from models.motion.models_motion import MotionVectorModel
from utils.model_utils import (
    compute_motion_fields,
    compute_projected_coords,
    l1_smoothness,
    normalize_translation,
    rotation_from_euler_angles,
    sqrt_sparsity,
)

from .losses import depth_smoothing_loss, depth_variance_loss, motion_consistency_loss, rgbd_consistency_losses


class DepthMotionModel(nn.Module):
    """Depth and motion model that predicts depth and motion from a pair of images.
    Contains all the relevant losses implemented.
    """

    def __init__(
        self,
        loss_weights,
        depth_model_args,
        motion_model_args,
        motion_burnin_steps: int = 20000,
        learn_egomotion: bool = True,
        motion_burnin_start_step: int = 0,
    ):
        super(DepthMotionModel, self).__init__()

        self.depth_model = DepthModel(**depth_model_args)
        self.motion_model = MotionVectorModel(**motion_model_args)

        self.loss_weights = loss_weights
        self.learn_egomotion = learn_egomotion
        self.learn_intrinsics = motion_model_args["learn_intrinsics"]
        self.steps = motion_burnin_start_step
        self.motion_burnin_steps = torch.tensor(motion_burnin_steps).float()

        # for checkpointing
        self.config = {
            "motion_burnin_steps": motion_burnin_steps,
            "loss_weights": loss_weights,
            "learn_egomotion": learn_egomotion,
            "depth_model": self._checkpointable_config(depth_model_args),
            "motion_model": self._checkpointable_config(motion_model_args),
        }

    def get_loss_weights(self):
        return self.loss_weights

    def compute_outputs(self, x):
        key = "color"
        images = [x[(key, -1, 0)], x[(key, 0, 0)]]  # tuple of B, 3, H, W
        B = images[0].shape[0]

        # feed augmented images to depth and flow model
        input_images = [x[("color_augmented", -1, 0)], x[("color_augmented", 0, 0)]]
        input_stack = torch.cat(input_images, dim=0)

        # Predict depth
        pred_depth = self.depth_model(input_stack)  # B x 2, 1, H, W
        pred_depth = torch.split(pred_depth, B, dim=0)  # (B, 1, H, W, B, 1, H, W)

        # Predict motion
        motion_features = [
            torch.cat((input_images[0], pred_depth[0]), dim=1),  # B, 4, H, W
            torch.cat((input_images[1], pred_depth[1]), dim=1),  # B, 4, H, W
        ]

        motion_feature_pairs = torch.cat(motion_features, dim=1)  # B, 8, H, W
        flipped_motion_feature_pairs = torch.cat(motion_features[::-1], dim=1)  # B, 8, H, W

        input_pairs = torch.cat((motion_feature_pairs, flipped_motion_feature_pairs), dim=0)  # B x 2, 8, H, W

        if not self.learn_egomotion:
            assert "rotation" in x and "inv_rotation" in x
            rotation = torch.cat([x["rotation"], x["inv_rotation"]], dim=0)  # B x 2, 1, 3
            rotation = rotation.squeeze(1)  # B x 2, 3

            assert "translation" in x and "inv_translation" in x
            translation = torch.cat([x["translation"], x["inv_translation"]], dim=0)  # B x 2, 3
            translation = translation.squeeze(1).unsqueeze(-1).unsqueeze(-1)  # B x 2, 3, 1, 1

            _, _, residual_translation, intrinsics_mat = self.motion_model(input_pairs)

        else:
            # motion and ego motion for both flipped and non-flipped images
            rotation, translation, residual_translation, intrinsics_mat = self.motion_model(input_pairs)

        rotation = rotation_from_euler_angles(rotation, expand=False)  # B x 2, 3, 3

        rotation = torch.split(rotation, B, dim=0)  # rotation, inv_rotation
        translation = torch.split(translation, B, dim=0)  # translation, inv_translation

        if self.motion_burnin_steps > 0:
            steps = torch.tensor(self.steps).float()
            residual_translation_burnin_scale = torch.clamp(2 * steps / self.motion_burnin_steps - 1, 0.0, 1.0)
            residual_translation *= residual_translation_burnin_scale

        residual_translation = torch.split(residual_translation, B, dim=0)

        if not self.learn_intrinsics:
            assert "K" in x
            intrinsics_mat = x["K"]

        else:
            # The intrinsic matrix should be the same, no matter the order of
            # images (mat = inv_mat). It's probably a good idea to enforce this
            # by a loss, but for now we just take their average as a prediction for the
            # intrinsic matrix.
            intrinsics_mat = 0.5 * sum(torch.split(intrinsics_mat, B, dim=0))

        outputs = {
            "rgb": images,  # tuple of (B, 3, H, W)
            "depth": pred_depth,  # tuple of (B, 1, H, W)
            "rotation": rotation,  # tuple of (B, 3, 3)
            "background_translation": translation,  # tuple of (B, 3, 1, 1)
            "residual_translation": residual_translation,  # tuple of (B, 3, H, W)
            "intrinsics": [intrinsics_mat] * 2,  # tuple of (B, 3, 3)
        }
        return outputs

    def forward(self, x, compute_motion_field=False):
        """Forward pass of the model. Computes outputs and loss metrics.
        Additionally, computes motion fields when in validation mode.
        """
        outputs = self.compute_outputs(x)
        losses = {}
        additional_metrics = {}

        rgb_stack = torch.cat(outputs["rgb"], dim=0)  # B x 2, 3, H, W

        flipped_rgb_stack = torch.cat(outputs["rgb"][::-1], dim=0)  # B x 2, 3, H, W
        pred_depth_stack = torch.cat(outputs["depth"], dim=0)  # B x 2, 1, H, W

        flipped_pred_depth_stack = torch.cat(outputs["depth"][::-1], dim=0).detach()  # B x 2, 1, H, W

        # prevent depth collapse by penalizing inverse of depth variance
        losses["depth_variance"] = depth_variance_loss(pred_depth_stack)

        # depth smoothing regularization
        losses["depth_smoothing"], disparity = depth_smoothing_loss(pred_depth_stack, rgb_stack)
        outputs["disparity"] = disparity

        background_translation = torch.cat(outputs["background_translation"], dim=0)  # B x 2, 3, 1, 1
        residual_translation = torch.cat(outputs["residual_translation"], dim=0)  # B x 2, 3, H, W
        flipped_background_translation = torch.cat(outputs["background_translation"][::-1], dim=0)  # B x 2, 3, 1, 1
        flipped_residual_translation = torch.cat(outputs["residual_translation"][::-1], dim=0)  # B x 2, 3, H, W
        translation = background_translation + residual_translation  # B x 2, 3, H, W]
        flipped_translation = flipped_background_translation + flipped_residual_translation  # B x 2, 3, H, W
        outputs["translation"] = translation
        rotation = torch.cat(outputs["rotation"], dim=0)  # B x 2, 3, 3
        flipped_rotation = torch.cat(outputs["rotation"][::-1], dim=0)  # B x 2, 3, 3

        intrinsics = torch.cat(outputs["intrinsics"], dim=0)  # B x 2, 3, 3

        projected_coords, projected_mask = compute_projected_coords(pred_depth_stack, translation, rotation, intrinsics)

        # rgbd consistency losses
        (
            rgb_error,
            depth_error,
            ssim_error,
            frame1_closer_to_camera,
            _,
            frame2rgbd_resampled,
        ) = rgbd_consistency_losses(
            projected_coords, projected_mask, rgb_stack, flipped_pred_depth_stack, flipped_rgb_stack
        )
        if compute_motion_field:
            with torch.no_grad():  # no need gradients while computing motion fields
                motion_fields = compute_motion_fields(outputs, projected_coords, frame1_closer_to_camera)
                outputs.update(motion_fields)

        losses["rgb_consistency"] = rgb_error
        losses["depth_consistency"] = depth_error
        losses["ssim"] = ssim_error

        # motion consistency losses
        rotation_error, translation_error = motion_consistency_loss(
            projected_coords, frame1_closer_to_camera, rotation, translation, flipped_rotation, flipped_translation
        )

        losses["rotation_cycle_consistency"] = rotation_error
        losses["translation_cycle_consistency"] = translation_error

        outputs["frame2rgb_resampled"] = frame2rgbd_resampled[:, :3]
        outputs["frame2depth_resampled"] = frame2rgbd_resampled[:, 3]

        normalized_translation = normalize_translation(residual_translation, translation)

        # motion smoothing regularization
        losses["motion_smoothing"] = l1_smoothness(normalized_translation, self.loss_weights["motion_sparsity"] == 0)

        # motion sparsity regularization
        losses["motion_sparsity"] = sqrt_sparsity(normalized_translation)

        for key, _ in losses.items():
            assert key in self.loss_weights, f"Key {key} not in loss weights"
            if self.loss_weights[key] == 0:
                additional_metrics[f"raw_{key}"] = losses[key].clone()
            losses[key] *= self.loss_weights[key] * 2

        self.steps += 1
        return losses, additional_metrics, outputs

    def _checkpointable_config(self, config):
        return {k: v for k, v in config.items() if k not in ["pretrained", "trainable"]}
