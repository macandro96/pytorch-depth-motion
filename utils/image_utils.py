"""This file contains utility functions for image processing
"""

from typing import Tuple

import matplotlib.pyplot as plt
import torch
from PIL import Image

_DEPTH_COLORMAP = plt.get_cmap("plasma", 256)  # For plotting depth maps


def pil_loader(img_path):
    """PIL load image from path"""
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(img_path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")


def normalize_image(x: torch.Tensor):
    """Rescale image pixels to span range [0, 1]"""
    max_val = float(x.max().cpu().item())
    min_val = float(x.min().cpu().item())
    d = max_val - min_val if max_val != min_val else 1e5
    return (x - min_val) / d


def colormap(x: torch.Tensor, normalize: bool = True):
    """Convert depth map to colormap"""
    assert x.ndim in [3, 4], "Input tensor must be 3D or 4D"

    if normalize:
        x = normalize_image(x)

    x = x.detach().cpu().numpy()
    if x.ndim == 4:
        x = x.transpose([0, 2, 3, 1])
        x = _DEPTH_COLORMAP(x)
        x = x[:, :, :, 0, :3]
        x = x.transpose(0, 3, 1, 2)
    else:
        x = _DEPTH_COLORMAP(x)
        x = x[0, :, :, :3]
        x = x.transpose(2, 0, 1)

    return torch.tensor(x)


def resize_and_crop_image(img: Image.Image, resize_dims: Tuple[int], crop: Tuple[int]) -> Image.Image:
    """Bilinear resizing followed by cropping."""
    img = img.resize(resize_dims, resample=Image.BILINEAR)
    img = img.crop(crop)
    return img


def apply_flow_mask(flow, mask):
    """Apply mask to flow_img tensor. Essentialy, resizes flow_img
    applies mask, and then resizes back to original size.
    """
    mask_height, mask_width = mask.shape[-2:]
    flow_height, flow_width = flow.shape[-2:]
    flow = torch.nn.functional.interpolate(flow, size=(mask_height, mask_width), mode="bilinear", align_corners=False)
    flow = flow * mask
    flow = torch.nn.functional.interpolate(flow, size=(flow_height, flow_width), mode="bilinear", align_corners=False)
    return flow
