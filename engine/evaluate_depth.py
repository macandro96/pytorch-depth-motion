"""
Reference: https://github.com/nianticlabs/monodepth2/blob/master/evaluate_depth.py
"""
from __future__ import absolute_import, division, print_function

import os

import cv2
import numpy as np

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths"""
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25**1).mean()
    a2 = (thresh < 1.25**2).mean()
    a3 = (thresh < 1.25**3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def evaluate_depth_error(gt_depths, pred_depths):
    # TODO: vectorize error computation

    # gt_depth: B, 1, H, W
    # pred_depth: B, 1, H, W
    gt_depths = gt_depths.squeeze(1).detach().cpu().numpy()
    pred_depths = pred_depths.squeeze(1).detach().cpu().numpy()

    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    errors = {'abs_rel': [], 'sq_rel': [], 'rmse': [], 'rmse_log': [], 'a1': [], 'a2': [], 'a3': []}
    for i in range(pred_depths.shape[0]):
        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        # resize shape of predicted depth to that of ground truth depth
        pred_depth = pred_depths[i]
        # import pdb; pdb.set_trace()
        pred_depth = cv2.resize(pred_depth, (gt_width, gt_height))

        # use eigen by default
        mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

        crop = np.array(
            [0.40810811 * gt_height, 0.99189189 * gt_height, 0.03594771 * gt_width, 0.96405229 * gt_width]
        ).astype(np.int32)
        crop_mask = np.zeros(mask.shape)
        crop_mask[crop[0]: crop[1], crop[2]: crop[3]] = 1
        mask = np.logical_and(mask, crop_mask)

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        # enable median scaling
        # if not opt.disable_median_scaling:
        ratio = np.median(gt_depth) / np.median(pred_depth)
        # ratios.append(ratio)
        pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = compute_errors(gt_depth, pred_depth)
        errors['abs_rel'].append(abs_rel)
        errors['sq_rel'].append(sq_rel)
        errors['rmse'].append(rmse)
        errors['rmse_log'].append(rmse_log)
        errors['a1'].append(a1)
        errors['a2'].append(a2)
        errors['a3'].append(a3)

    return errors
