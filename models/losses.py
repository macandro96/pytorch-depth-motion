import torch
import torch.nn.functional as F

from utils.model_utils import _gradient_x, _gradient_y, _weighted_average, multiply_no_nan, weighted_ssim


def depth_supervision_loss(pred_depth, gt_depth):
    depth_error = torch.abs(pred_depth - gt_depth)
    depth_filter = torch.where(gt_depth > 0.2, torch.ones_like(gt_depth), torch.zeros_like(gt_depth)).float()
    return torch.mean(depth_error * depth_filter) / torch.mean(depth_filter)


def depth_variance_loss(pred_depth):
    mean_depth = torch.mean(pred_depth)
    var_depth = torch.mean(torch.square(pred_depth / mean_depth - 1.0))
    return 1.0 / var_depth


def pose_supervision_loss(rotation, translation, gt_rotation, gt_translation):
    rotation_loss = (torch.mean(torch.square(rotation - gt_rotation))) ** (0.5)
    translation_loss = (torch.mean(torch.square(translation - gt_translation))) ** 0.5

    return rotation_loss, translation_loss


def depth_smoothing_loss(pred_depth, rgb):
    disparity = 1.0 / pred_depth
    mean_disparity = torch.mean(disparity, dim=(1, 2, 3), keepdim=True)
    scaled_disparity = disparity / mean_disparity

    scaled_disparity_dx = _gradient_x(scaled_disparity)
    scaled_disparity_dy = _gradient_y(scaled_disparity)
    rgb_dx = _gradient_x(rgb)
    rgb_dy = _gradient_y(rgb)
    weights_x = torch.exp(-torch.mean(torch.abs(rgb_dx), dim=1, keepdim=True))
    weights_y = torch.exp(-torch.mean(torch.abs(rgb_dy), dim=1, keepdim=True))
    smoothness_x = scaled_disparity_dx * weights_x
    smoothness_y = scaled_disparity_dy * weights_y

    return torch.mean(torch.abs(smoothness_x)) + torch.mean(torch.abs(smoothness_y)), disparity


def rgbd_consistency_losses(projected_coords, mask, rgb, flipped_pred_depth, flipped_rgb):
    frame1_projected_coords = projected_coords[:, :2]  # B x 2, 2, H, W
    frame1_projected_depth = projected_coords[:, 2:3]  # B x 2, 1, H, W

    frame2rgbd = torch.cat([flipped_rgb, flipped_pred_depth], dim=1)  # B x 2, 4, H, W

    frame2rgbd_resampled = (
        torch.nan_to_num(
            F.grid_sample(frame2rgbd, frame1_projected_coords.permute(0, 2, 3, 1), align_corners=True),
            posinf=0.0,
            neginf=0.0,
        )
        * mask.float()
    )  # B x 2, 4, H, W

    frame2rgb_resampled = frame2rgbd_resampled[:, :3]
    frame2depth_resampled = frame2rgbd_resampled[:, 3:4]

    frame1_closer_to_camera = torch.logical_and(
        mask, frame1_projected_depth < frame2depth_resampled  # B x 2, 1, H, W
    ).float()

    depth_l1_diff = torch.abs(frame2depth_resampled - frame1_projected_depth)  # B x 2, 1, H, W
    depth_error = torch.mean(multiply_no_nan(depth_l1_diff, frame1_closer_to_camera))
    rgb_l1_diff = torch.abs(frame2rgb_resampled - rgb)  # B x 2, 3, H, W

    rgb_error = torch.mean(rgb_l1_diff * torch.nan_to_num(frame1_closer_to_camera, posinf=0.0, neginf=0.0))
    warped_rgb_diff = torch.sum(rgb_l1_diff)

    depth_error_second_moment = _weighted_average(torch.square(depth_l1_diff), frame1_closer_to_camera) + 1e-4
    depth_proximity_weight = (
        torch.nan_to_num(
            depth_error_second_moment / (torch.square(depth_l1_diff) + depth_error_second_moment),
            posinf=0.0,
            neginf=0.0,
        )
        * mask.float()
    )
    depth_proximity_weight = depth_proximity_weight.detach()

    ssim_error, avg_weight = weighted_ssim(frame2rgb_resampled, rgb, depth_proximity_weight, c1=float("inf"), c2=9e-6)
    ssim_error_mean = torch.mean(multiply_no_nan(ssim_error, avg_weight))
    return rgb_error, depth_error, ssim_error_mean, frame1_closer_to_camera, warped_rgb_diff, frame2rgbd_resampled


def motion_consistency_loss(projected_coords, mask, rotation, translation, flipped_rotation, flipped_translation):
    frame1_projected_coords = projected_coords.detach()[:, :2].permute(0, 2, 3, 1)  # B x 2, H, W, 2
    translation2_resampled = F.grid_sample(
        flipped_translation, frame1_projected_coords, align_corners=True
    )  # B x 2, 3, H, W

    combined_rotations = torch.matmul(rotation, flipped_rotation)  # B x 2, 3, 3
    combined_translations = (
        torch.einsum("bij,bjhw->bihw", rotation, translation2_resampled) + translation  # B x 2, 3, H, W
    )

    eye = torch.eye(3)[None, :].to(rotation.device)  # 1, 3, 3
    rotation_error = torch.mean(torch.square(combined_rotations - eye), dim=(1, 2))
    rotation1_scale = torch.mean(torch.square(rotation - eye), dim=(1, 2))
    rotation2_scale = torch.mean(torch.square(flipped_rotation - eye), dim=(1, 2))
    rotation_error /= 1e-24 + rotation1_scale + rotation2_scale
    rotation_error = torch.mean(rotation_error)

    def norm(x):
        return torch.sum(torch.square(x), dim=1)

    translation_error = torch.mean(
        mask
        * torch.nan_to_num(
            (norm(combined_translations) / (1e-24 + norm(translation) + norm(translation2_resampled))),
            posinf=0.0,
            neginf=0.0,
        )
    )

    return rotation_error, translation_error
