import copy
import warnings
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def _gradient_x(img):
    return img[:, :, :, :-1] - img[:, :, :, 1:]


def _gradient_y(img):
    return img[:, :, :-1, :] - img[:, :, 1:, :]


def invert_intrinsics_mat(intrinsics_mat):
    """Inverts intrinsics matrix."""
    intrinsics_mat_cols = torch.unbind(intrinsics_mat, dim=-1)
    if len(intrinsics_mat_cols) != 3:
        raise ValueError("Intrinsics matrix must have 3 columns.")

    fx, _, _ = torch.unbind(intrinsics_mat_cols[0], dim=-1)
    _, fy, _ = torch.unbind(intrinsics_mat_cols[1], dim=-1)
    x0, y0, _ = torch.unbind(intrinsics_mat_cols[2], dim=-1)

    zeros = torch.zeros_like(fx)
    ones = torch.ones_like(fy)

    row1 = torch.stack([1.0 / fx, zeros, zeros], dim=-1)
    row2 = torch.stack([zeros, 1.0 / fy, zeros], dim=-1)
    row3 = torch.stack([-x0 / fx, -y0 / fy, ones], dim=-1)
    return torch.stack([row1, row2, row3], dim=-1)


def multiply_no_nan(x, y):
    """Multiplies two tensors, replacing NaNs with zeros."""
    mask = torch.isnan(x) | torch.isnan(y)
    return torch.where(mask, torch.zeros_like(x), x * y)


def rotation_from_euler_angles(vec, expand=True):
    """Convert an Euler angle representation to a rotation matrix.

    refer to the implementation in tensorflow-graphics
    """
    sa = torch.sin(vec)
    ca = torch.cos(vec)

    sx, sy, sz = torch.unbind(sa, axis=-1)
    cx, cy, cz = torch.unbind(ca, axis=-1)

    if expand:
        rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)
    else:
        rot = torch.zeros((vec.shape[0], 3, 3)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(cy * cz)
    rot[:, 0, 1] = torch.squeeze((sx * sy * cz) - (cx * sz))
    rot[:, 0, 2] = torch.squeeze((cx * sy * cz) + (sx * sz))
    rot[:, 1, 0] = torch.squeeze(cy * sz)
    rot[:, 1, 1] = torch.squeeze((sx * sy * sz) + (cx * cz))
    rot[:, 1, 2] = torch.squeeze((cx * sy * sz) - (sx * cz))
    rot[:, 2, 0] = torch.squeeze(-sy)
    rot[:, 2, 1] = torch.squeeze(sx * cy)
    rot[:, 2, 2] = torch.squeeze(cx * cy)

    if expand:
        rot[:, 3, 3] = 1

    return rot


def compute_projected_rotation(R, K, inv_K):
    """Compute the projected rotation matrix."""
    R = R.transpose(-1, -2)  # B x 2, 3, 3
    return torch.einsum("bij,bjk,bkl->bil", K, R, inv_K)  # B x 2, 3, 3


def compute_projected_translation(translation, intrinsics):
    """Compute the projected translation vector."""
    return torch.einsum("bij,bjhw->bihw", intrinsics, translation)  # B x 2, 3, H, W


def compute_projected_coords(depth, translation, rotation, intrinsics):
    """Compute the projected coordinates of a depth map.

    Source: https://github.com/bolianchen/pytorch_depth_from_videos_in_the_wild/blob/main/lib/torch_layers.py
    """
    intrinsics_inv = invert_intrinsics_mat(intrinsics)  # B x 2, 3, 3
    depth = depth.squeeze()  # B x 2, H, W

    H, W = depth.shape[1:]
    x, y = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy")
    grid = torch.stack([x, y, torch.ones_like(x)], dim=0).type(torch.float32).to(depth.device)

    projected_rotation = compute_projected_rotation(rotation, intrinsics, intrinsics_inv)  # B x 2, 3, 3
    projected_coords = torch.einsum("bij,jhw,bhw->bihw", projected_rotation, grid, depth)  # B x 2, 3, H, W
    projected_translation = compute_projected_translation(translation, intrinsics)  # B x 2, 3, H, W
    projected_coords = projected_coords + projected_translation  # B x 2, 3, H, W

    x, y, z = torch.unbind(projected_coords, axis=1)
    pixel_x, pixel_y = x / z, y / z
    coords_not_underflow = torch.logical_and(pixel_x >= 0.0, pixel_y >= 0.0)
    coords_not_overflow = torch.logical_and(pixel_x <= W - 1, pixel_y <= H - 1)
    z_positive = z > 0

    coords_not_nan = torch.logical_not(torch.logical_or(torch.isnan(x), torch.isnan(y)))

    not_nan_mask = coords_not_nan.float()
    pixel_x *= not_nan_mask
    pixel_y *= not_nan_mask
    mask = coords_not_underflow & coords_not_overflow & coords_not_nan & z_positive

    # clamp
    pixel_x = pixel_x.clamp(min=0.0, max=W - 1)
    pixel_y = pixel_y.clamp(min=0.0, max=H - 1)
    # normalize
    pixel_x /= W - 1
    pixel_y /= H - 1
    pixel_x = (pixel_x - 0.5) * 2
    pixel_y = (pixel_y - 0.5) * 2

    pix_coords = torch.cat([pixel_x.unsqueeze(-1), pixel_y.unsqueeze(-1)], dim=3)
    return torch.cat([pix_coords.permute(0, 3, 1, 2), z.unsqueeze(1)], dim=1), mask.unsqueeze(1)


def weighted_ssim(x, y, weight, c1, c2, weight_epsilon=1e-4):
    """Computes a differentiable structured image similarity measure."""
    weight_plus_epsilon = weight + weight_epsilon
    avg_pooled_weight = F.avg_pool2d(weight, kernel_size=3, stride=1)
    inv_avg_pooled_weight = 1.0 / (avg_pooled_weight + weight_epsilon)

    def weighted_avg_pool3x3(z):
        weighted_avg = F.avg_pool2d(z * weight_plus_epsilon, kernel_size=3, stride=1)
        return weighted_avg * inv_avg_pooled_weight

    mu_x = weighted_avg_pool3x3(x)
    mu_y = weighted_avg_pool3x3(y)

    sigma_x = weighted_avg_pool3x3(torch.square(x)) - torch.square(mu_x)
    sigma_y = weighted_avg_pool3x3(torch.square(y)) - torch.square(mu_y)
    sigma_xy = weighted_avg_pool3x3(x * y) - (mu_x * mu_y)

    if c1 == float("inf"):
        ssim_n = 2 * sigma_xy + c2
        ssim_d = sigma_x + sigma_y + c2
    elif c2 == float("inf"):
        ssim_n = 2 * mu_x * mu_y + c1
        ssim_d = torch.square(mu_x) + torch.square(mu_y) + c1
    else:
        ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        ssim_d = (torch.square(mu_x) + torch.square(mu_y) + c1) * (sigma_x + sigma_y + c2)
    result = ssim_n / ssim_d
    return torch.clip((1 - result) / 2, 0, 1), avg_pooled_weight


def normalize_translation(residual_translation, translation):
    """Normalize translation vector by its norm."""
    norm = 3.0 * torch.mean(torch.square(translation), axis=(1, 2, 3), keepdim=True)
    return residual_translation / torch.sqrt(norm + 1e-12)


def l1_smoothness(tensor, wrap_around=True):
    """Computes the l1 smoothness loss for a tensor."""
    # l1 smoothness loss
    tensor_dx = tensor - torch.roll(tensor, shifts=1, dims=2)
    tensor_dy = tensor - torch.roll(tensor, shifts=1, dims=3)

    if not wrap_around:
        tensor_dx = tensor_dx[:, :, 1:, 1:]
        tensor_dy = tensor_dy[:, :, 1:, 1:]
    return torch.mean(torch.sqrt(1e-24 + torch.square(tensor_dx) + torch.square(tensor_dy)))


def sqrt_sparsity(tensor):
    """Computes the sqrt sparsity loss for a tensor."""
    # sqrt sparsity loss
    tensor_abs = torch.abs(tensor)
    tensor_mean = torch.mean(tensor_abs, dim=(2, 3), keepdim=True).detach()
    return torch.mean(2 * tensor_mean * torch.sqrt(1 + tensor_abs / (tensor_mean + 1e-24)))


def _weighted_average(x, w, epsilon=1.0):
    """Computes the weighted average of a tensor along the spatial dimensions."""
    weighted_sum = torch.sum(x * w, dim=(2, 3), keepdim=True)
    sum_of_weights = torch.sum(w, dim=(2, 3), keepdim=True)
    return weighted_sum / (sum_of_weights + epsilon)


def normalize_image(x: torch.Tensor):
    """Rescale image pixels to span range [0, 1]."""
    max_val = float(x.max().cpu().item())
    min_val = float(x.min().cpu().item())
    d = max_val - min_val if max_val != min_val else 1e5
    return (x - min_val) / d


def resize(
    input_img: torch.Tensor,
    size: Optional[Tuple[int]] = None,
    scale_factor: Optional[float] = None,
    mode: str = "nearest",
    align_corners: Optional[bool] = False,
    warning: bool = True,
):
    """Resize input image."""
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input_img.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if (
                    (output_h > 1 and output_w > 1 and input_h > 1 and input_w > 1)
                    and (output_h - 1) % (input_h - 1)
                    and (output_w - 1) % (input_w - 1)
                ):
                    # trunk-ignore(ruff/B028)
                    warnings.warn(
                        f"When align_corners={align_corners}, the output would more aligned if "
                        f"input size {(input_h, input_w)} is `x+1` and out size {(output_h, output_w)} is `nx+1`"
                    )
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)

    if mode not in ["linear", "bilinear", "bicubic", "trilinear"]:
        align_corners = None

    return F.interpolate(input_img, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)


def normalize_trans(x):
    """Rescale translation

    if all values are positive, rescale the max to 1.0
    otherwise, make sure the zeros be mapped to 0.5, and
    either the max mapped to 1.0 or the min mapped to 0

    """
    # do not add the following to the computation graph
    x = x.detach()

    ma = float(x.max().cpu().data) + 1e-8
    mi = float(x.min().cpu().data) + 1e-8

    assert ma != 0 or mi != 0

    d = max(abs(ma), abs(mi))
    x[x >= 0] = 0.5 + 0.5 * x[x >= 0] / d
    x[x < 0] = 0.5 + 0.5 * x[x < 0] / d

    return x


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    vec = vec.unsqueeze(1)
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 3, 3)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)

    return rot


@torch.no_grad()
def compute_motion_fields(outputs, projected_coords, frame1_closer_to_camera):
    """Retrieve motion fields given outputs from the model. Usually used with torch.no_grad()"""
    frames = outputs["rgb"]
    rgb_stack = torch.cat(frames, dim=0)
    B = frames[0].shape[0]

    frame1_closer_to_camera = torch.split(frame1_closer_to_camera, B, dim=0)
    pred_depths = outputs["depth"]
    pred_depth_stack = torch.cat(pred_depths, dim=0)

    background_translation, rotation = outputs["background_translation"], outputs["rotation"]
    background_translation, rotation = torch.cat(background_translation, dim=0), torch.cat(rotation, dim=0)
    residual_translation = torch.cat(outputs["residual_translation"], dim=0)
    intrinsics = torch.cat(outputs["intrinsics"], dim=0)

    background_projected_coords, _ = compute_projected_coords(
        pred_depth_stack, background_translation, rotation, intrinsics
    )
    identity = torch.eye(3)[None, :].to(background_projected_coords.device)  # 1, 3, 3
    object_projected_coords, _ = compute_projected_coords(pred_depth_stack, residual_translation, identity, intrinsics)
    H, W = rgb_stack.shape[-2:]
    coords_x, coords_y = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy")
    coords = (
        torch.stack([coords_x, coords_y], dim=0).type(torch.float32).to(rgb_stack.device).unsqueeze(0)
    )  # 1, 2, H, W

    background_projected_coords[:, 0] = (background_projected_coords[:, 0] + 1) * (W - 1) / 2
    background_projected_coords[:, 1] = (background_projected_coords[:, 1] + 1) * (H - 1) / 2

    object_projected_coords[:, 0] = (object_projected_coords[:, 0] + 1) * (W - 1) / 2
    object_projected_coords[:, 1] = (object_projected_coords[:, 1] + 1) * (H - 1) / 2

    projected_coords_detached = copy.deepcopy(projected_coords.detach())
    projected_coords_detached[:, 0] = (projected_coords_detached[:, 0] + 1) * (W - 1) / 2
    projected_coords_detached[:, 1] = (projected_coords_detached[:, 1] + 1) * (H - 1) / 2

    background_motion = torch.nan_to_num(background_projected_coords[:B, :2], posinf=0.0, neginf=0.0) - coords

    object_motion = torch.nan_to_num(object_projected_coords[:B, :2], posinf=0.0, neginf=0.0) - coords

    global_motion = torch.nan_to_num(projected_coords_detached[:B, :2], posinf=0.0, neginf=0.0) - coords

    return {"background_motion": background_motion, "object_motion": object_motion, "global_motion": global_motion}
