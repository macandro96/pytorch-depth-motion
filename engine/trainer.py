import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torchvision.utils import flow_to_image, make_grid

from engine.evaluate_depth import evaluate_depth_error
from utils.image_utils import apply_flow_mask
from utils.model_utils import normalize_image


class DepthMotionTrainer(pl.LightningModule):
    def __init__(
        self,
        model,
        optimizer_cfg,
        log_image_interval_train=1000,
        log_image_interval_val=500,
        monitor_val=None,
        mask_flow=False,
    ):
        super().__init__()
        self.model = model
        self.log_image_interval_train = log_image_interval_train
        self.log_image_interval_val = log_image_interval_val
        self.monitor_val = monitor_val
        self.mask_flow = mask_flow

        self.trainable_parameters = list(self.model.parameters())

        self.save_hyperparameters(ignore=["model", "auxillary_model"])
        self.optimizer_cfg = optimizer_cfg if isinstance(optimizer_cfg, DictConfig) else OmegaConf.create(optimizer_cfg)
        self.loss_weights = self.model.loss_weights
        self.test_depth_errors = []
        self.test_flow_errors = []

    def _plot_global_motion(self, outputs, idx=0):
        """Helper plotter function for debugging purposes."""
        global_motion = flow_to_image(outputs["global_motion"])
        object_motion = flow_to_image(outputs["object_motion"])
        background_motion = flow_to_image(outputs["background_motion"])

        with torch.no_grad():
            frame2depth_resampled = normalize_image(1 / (outputs["frame2depth_resampled"] + 1e-7))

        plt.imsave("rgb.png", outputs["rgb"][0][idx].permute(1, 2, 0).cpu().numpy())
        plt.imsave("global_motion.png", global_motion[idx].permute(1, 2, 0).cpu().numpy())
        plt.imsave("object_motion.png", object_motion[idx].permute(1, 2, 0).cpu().numpy())
        plt.imsave("background_motion.png", background_motion[idx].permute(1, 2, 0).cpu().numpy())
        plt.imsave("frame2depth_resampled1.png", frame2depth_resampled[idx].cpu().numpy())

    def process_batch(self, batch, compute_motion_field=True):
        # compute auxillary outputs
        losses, _, outputs = self.model(batch, compute_motion_field=compute_motion_field)

        loss_val = sum(losses.values())

        return loss_val, outputs, losses

    def _log_image_helper(self, batch, outputs, random_idx, mode="train"):
        color_key = "color"
        # Collect ground truth
        image_a, image_b = batch[(color_key, -1, 0)], batch[(color_key, 0, 0)]
        images_gt = [image_a[random_idx], image_b[random_idx], torch.abs(image_a[random_idx] - image_b[random_idx])]

        # Collect disparity
        batch_size = outputs["disparity"].shape[0] // 2
        disp_a, disp_b = torch.split(outputs["disparity"], batch_size, dim=0)
        images_disp = [normalize_image(disp_a[random_idx].detach()), normalize_image(disp_b[random_idx].detach())]
        images_disp[0] = torch.cat([images_disp[0], images_disp[0], images_disp[0]], dim=0)
        images_disp[1] = torch.cat([images_disp[1], images_disp[1], images_disp[1]], dim=0)

        # Collect warped images
        warped_a, warped_b = torch.split(outputs["frame2rgb_resampled"], batch_size, dim=0)
        images_warped = [warped_a[random_idx].detach(), warped_b[random_idx].detach()]

        # log object motion
        motion_tensor = torch.cat(
            [outputs["object_motion"], outputs["background_motion"], outputs["global_motion"]], dim=0
        )
        # normalize all the flows at once, so that the scaling is constant throughout
        flow_images = flow_to_image(motion_tensor) / 255.0

        object_motion, background_motion, global_motion = torch.split(flow_images, batch_size, dim=0)

        images_motion = [
            object_motion[random_idx].detach(),
            background_motion[random_idx].detach(),
            global_motion[random_idx].detach(),
        ]
        if self.mask_flow and "mask" in batch:
            # apply flow mask to flow predictions
            flow_masked = apply_flow_mask(motion_tensor, batch["mask"])
            flow_images_masked = flow_to_image(flow_masked) / 255.0
            object_motion_masked, background_motion_masked, global_motion_masked = torch.split(
                flow_images_masked, batch_size, dim=0
            )
            images_motion += [
                object_motion_masked[random_idx].detach(),
                background_motion_masked[random_idx].detach(),
                global_motion_masked[random_idx].detach(),
            ]
        # log flow ground truth
        images_flow_gt = []
        if "flow_gt" in batch:
            flow_gt = F.interpolate(batch["flow_gt"], size=(image_a.shape[2], image_a.shape[3]), mode="bilinear")
            images_flow_gt = [flow_to_image(flow_gt[random_idx]) / 255.0]

        # gt_b - warped
        diff_img = [torch.abs(image_a - warped_a)[random_idx]]

        grid_img = make_grid(
            images_gt + images_motion + images_disp + images_warped + diff_img + images_flow_gt, nrow=2
        )

        self.logger.log_image(
            key=f"{mode}/images1",
            images=[grid_img],
            caption=["ground_truth, res_trans, flows, flow_gt, depth, warped"],
        )

        # Log translation scale
        background_translation = torch.cat(outputs["background_translation"], dim=0)[random_idx].detach()  # B x 2,3,1,1
        residual_translation = torch.cat(outputs["residual_translation"], dim=0)[random_idx].detach()  # B x 2, 3, H, W
        global_translation = outputs["translation"][random_idx].detach()  # B x 2, 3, H, W

        object_motion_avg = torch.mean(torch.abs(residual_translation[residual_translation != 0]))
        background_motion_avg = torch.mean(torch.abs(background_translation[background_translation != 0]))
        global_motion_avg = torch.mean(torch.abs(global_translation[global_translation != 0]))

        self.log(f"{mode}/residual_translation_avg", object_motion_avg, sync_dist=True)
        self.log(f"{mode}/background_translation_avg", background_motion_avg, sync_dist=True)
        self.log(f"{mode}/global_translation_avg", global_motion_avg, sync_dist=True)

        # Log flow scales
        background_flow = outputs["background_motion"][random_idx].detach()  # B x 2,3,1,1
        residual_flow = outputs["object_motion"][random_idx].detach()  # B x 2, 3, H, W
        global_flow = outputs["global_motion"][random_idx].detach()  # B x 2, 3, H, W

        self.log(f"{mode}/residual_flow_avg", torch.mean(torch.abs(residual_flow)), sync_dist=True)
        self.log(f"{mode}/background_flow_avg", torch.mean(torch.abs(background_flow)), sync_dist=True)
        self.log(f"{mode}/global_flow_avg", torch.mean(torch.abs(global_flow)), sync_dist=True)

    def log_images(self, batch, outputs, dataset="kitti", mode="train"):
        if self.logger is None:
            return  # No logger, no logging
        assert mode in ["train", "val", "test"]
        assert dataset in ["kitti", "kitti_flow_occ", "kitti_flow_noc"]  # kitti flow - noc and occ, kitti is raw one

        # pick random idx for plotting
        if mode != "test":
            random_idx = np.random.randint(0, len(outputs["rgb"][0]))
            self._log_image_helper(batch, outputs, random_idx, mode=f"{mode}_{dataset}")
        # if test mode then log all images of the KITTI_flow dataset
        else:
            for idx in range(len(outputs["rgb"][0])):
                self._log_image_helper(batch, outputs, idx, mode=f"{mode}_{dataset}")

    def log_kitti_flow_metrics(self, batch, outputs, dataset):
        epe_global = self.calculate_flow_metrics(outputs, batch, flow_type="global")
        epe_background = self.calculate_flow_metrics(outputs, batch, flow_type="background")
        epe_object = self.calculate_flow_metrics(outputs, batch, flow_type="object")

        self.log(f"kitti_flow/{dataset}/epe_mean", epe_global, sync_dist=True)

        self.log(f"kitti_flow/{dataset}/epe_background", epe_background, sync_dist=True)
        self.log(f"kitti_flow/{dataset}/epe_object", epe_object, sync_dist=True)

        flow_errors = {
            f"{dataset}/epe_global": epe_global,
            f"{dataset}/epe_background": epe_background,
            f"{dataset}/epe_object": epe_object,
        }
        return flow_errors

    def log_metrics(self, batch_idx, batch, outputs, loss_val, losses, mode="train", dataloader_idx=0):
        """Logging metrics for training, validation and testing.
        We log only the depth metrics for the test dataset.
        """
        depth_errors = None
        if "gt_depths" in batch:
            gt_depths = torch.cat(batch["gt_depths"], dim=0)
            pred_depths = torch.cat(outputs["depth"], dim=0)
            depth_errors = evaluate_depth_error(gt_depths, pred_depths)
            for key, value in depth_errors.items():
                depth_errors[key] = np.mean(value)
                if dataloader_idx == 0:
                    self.log(f"{mode}/depth/{key}", depth_errors[key], sync_dist=True, add_dataloader_idx=False)

        # log losses
        if dataloader_idx == 0:
            # to make model checkpointing easier.
            self.log(f"{mode}_loss", loss_val, prog_bar=True, sync_dist=True, add_dataloader_idx=False)

            res_trans = torch.cat(outputs["residual_translation"], dim=0)
            max_trans = torch.abs(res_trans).max()
            self.log(f"{mode}/max_res_trans", max_trans, prog_bar=True, sync_dist=True, add_dataloader_idx=False)

        for key, value in losses.items():
            if self.loss_weights[key] != 0:
                self.log(
                    f"{mode}/{key}",
                    value,
                    sync_dist=True,
                )

        # log images
        if mode == "train":
            log_interval = self.log_image_interval_train
        elif mode == "val":
            log_interval = self.log_image_interval_val
        else:
            log_interval = 1

        # log epe flow if it's the kitti flow dataset
        flow_errors = None
        if batch_idx % log_interval == 0:
            dataset = "kitti"
            with torch.no_grad():
                if dataloader_idx > 0:
                    if dataloader_idx == 1:
                        dataset = "kitti_flow_occ"
                    elif dataloader_idx == 2:
                        dataset = "kitti_flow_noc"
                    flow_errors = self.log_kitti_flow_metrics(batch, outputs, dataset=dataset)
                self.log_images(batch, outputs, dataset=dataset, mode=mode)

        return depth_errors, flow_errors

    def training_step(self, batch, batch_idx):
        loss, outputs, losses = self.process_batch(batch, compute_motion_field=True)
        with torch.no_grad():  # no gradients needed here
            self.log_metrics(batch_idx, batch, outputs, loss, losses, mode="train")

        return loss

    def calculate_flow_metrics(self, outputs, batch, flow_type="global"):
        """Calculates average end-point error"""
        assert flow_type in ["global", "object", "background"]

        ground_truth = batch["flow_gt"]
        flow = outputs[f"{flow_type}_motion"]

        # need to resize our output flow
        h, w = ground_truth.shape[2:]
        ph, pw = flow.shape[2:]
        ratio_h, ratio_w = h / ph, w / pw
        flow = F.interpolate(flow, size=(h, w))
        flow[:, 0] *= ratio_w
        flow[:, 1] *= ratio_h
        error = torch.norm(ground_truth - flow, p=2, dim=1)  # epe

        valid = batch["mask"] > 0  # Mask out invalid pixels
        epe = error[valid].mean()  # Filter out invalid pixels

        return epe

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, outputs, losses = self.process_batch(batch, compute_motion_field=True)
        self.log_metrics(batch_idx, batch, outputs, loss, losses, mode="val", dataloader_idx=dataloader_idx)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss, outputs, losses = self.process_batch(batch, compute_motion_field=True)
        depth_errors, flow_errors = self.log_metrics(
            batch_idx, batch, outputs, loss, losses, mode="test", dataloader_idx=dataloader_idx
        )
        batch_size = batch[("color", 0, 0)].shape[0]
        if dataloader_idx == 0:
            assert depth_errors is not None
            depth_errors["batch_size"] = batch_size
            self.test_depth_errors.append(depth_errors)

        elif dataloader_idx >= 1:
            assert flow_errors is not None
            flow_errors["batch_size"] = batch_size
            self.test_flow_errors.append(flow_errors)

    def _test_metrics_step_aggregator(self, error_dict):
        error_agg_dict = {}
        error_size_dict = {}

        for error in error_dict:
            for key, value in error.items():
                error_agg_dict[key] = error_agg_dict.get(key, 0) + value * error["batch_size"]
                error_size_dict[key] = error_size_dict.get(key, 0) + error["batch_size"]

        return {key: value / error_size_dict[key] for key, value in error_agg_dict.items()}

    def on_test_end(self):
        test_depth_errors = self._test_metrics_step_aggregator(self.test_depth_errors)
        print(test_depth_errors)
        test_flow_errors = self._test_metrics_step_aggregator(self.test_flow_errors)

        print(test_flow_errors)

    def configure_optimizers(self):
        optimizer_scheduler_dict = {}

        # Prepare the optimizer
        # ---------------------
        if self.optimizer_cfg.name == "adamw":
            optimizer_scheduler_dict["optimizer"] = AdamW(
                self.trainable_parameters,
                lr=self.optimizer_cfg.lr,
                weight_decay=self.optimizer_cfg.weight_decay,
                betas=self.optimizer_cfg.betas,
            )
        elif self.optimizer_cfg.name == "adam":
            optimizer_scheduler_dict["optimizer"] = Adam(
                self.trainable_parameters,
                lr=self.optimizer_cfg.lr,
                weight_decay=self.optimizer_cfg.weight_decay,
                betas=self.optimizer_cfg.betas,
            )
        else:
            raise ValueError("Optimizer must be either `adamw` or `adam`")

        # Prepare the scheduler
        # ---------------------
        if "scheduler" in self.optimizer_cfg:
            scheduler_cfg = self.optimizer_cfg.scheduler
            if scheduler_cfg.name == "StepLR":
                assert scheduler_cfg.step_size < self.trainer.max_epochs, "Step size must be less than max epochs"
                interval = "epoch"
                scheduler = StepLR(
                    optimizer_scheduler_dict["optimizer"],
                    step_size=scheduler_cfg.step_size,
                    gamma=scheduler_cfg.gamma,
                )
                optimizer_scheduler_dict["lr_scheduler"] = {
                    "scheduler": scheduler,
                    "interval": interval,
                }
            elif scheduler_cfg.name == "ReduceLROnPlateau":
                scheduler = ReduceLROnPlateau(optimizer_scheduler_dict["optimizer"], factor=0.8, patience=5)
                optimizer_scheduler_dict["lr_scheduler"] = {"scheduler": scheduler, "monitor": self.monitor_val}
            elif scheduler_cfg.name is None:
                pass
            else:
                raise ValueError("Scheduler must be `StepLR` or `ReduceLROnPlateau`")

        return optimizer_scheduler_dict

    def on_save_checkpoint(self, checkpoint):
        checkpoint["model_config"] = self.model.config
        checkpoint["steps"] = self.model.steps

    def on_before_zero_grad(self, _) -> None:
        # clamp the motion scaler weights
        self.model.motion_model.scaler.rot_scale.data = torch.clamp(
            self.model.motion_model.scaler.rot_scale.data, min=0.001
        )

        self.model.motion_model.scaler.trans_scale.data = torch.clamp(
            self.model.motion_model.scaler.trans_scale.data, min=0.001
        )
