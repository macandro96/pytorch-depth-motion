"""
Source: https://github.com/nianticlabs/monodepth2/tree/master
"""
from __future__ import absolute_import, division, print_function

import os
import pickle
import random
from typing import List

import numpy as np
import PIL.Image as pil
import skimage
import torch
from PIL import Image  # using pillow-simd for increased speed
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from utils.image_utils import pil_loader
from utils.kitti_utils import generate_depth_map


# KITTI and Monodepth2 dataloaders
class MonoDataset(Dataset):
    """Superclass for monocular dataloaders. This class will be inherited by the KITTI dataset.

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """

    def __init__(
        self,
        data_path: str,
        filenames: List[str],
        height: int,
        width: int,
        frame_idxs: List[int],
        num_scales: int,
        is_train: bool = False,
        img_ext=".jpg",
        dedup_threshold=0,
    ):
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.LANCZOS

        self.frame_idxs = frame_idxs
        self.dedup_threshold = dedup_threshold

        self.is_train = is_train
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        self.resize = {}
        for i in range(self.num_scales):
            s = 2**i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s), interpolation=self.interp)

        self.load_depth = self.check_depth()

    def _setup_augmentation_parameters(self):
        brightness = 0.2
        contrast = 0.2
        saturation = 0.2
        hue = 0.1
        return transforms.ColorJitter(brightness, contrast, saturation, hue)

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_augmented", im, i)] = self.to_tensor(color_aug(f))

    def __len__(self):
        return len(self.filenames)

    def _get_metadata(self, index):
        line = self.filenames[index].split()
        folder = line[0]

        if len(line) == 3:
            frame_index = int(line[1])
            side = line[2]
        else:
            frame_index = 0
            side = None
        return folder, frame_index, side

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_augmented", <frame_id>, <scale>)      for augmented colour images,
            "K", "inv_K"                            for camera intrinsics,
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1) representing the temporal step relative to 'index',

        <scale> is an integer representing the scale of the image relative to the fullsize image:
           currently, we test with only <scale>=0.
        """
        inputs = {}

        do_color_aug = (
            self.is_train and random.random() > 0.5
        )  # A/c to the paper, 50% of the training samples are augmented
        do_flip = self.is_train and random.random() > 0.5  # A/c to the paper, 50% of the training samples are flipped

        folder, frame_index, side = self._get_metadata(index)

        inputs[("gt_depths")] = []
        # store only 2 frames with constant naming convention
        for idx, i in enumerate(self.frame_idxs):
            inputs[("color", idx - 1, -1)] = self.get_color(folder, frame_index + i, side, do_flip)
            if self.load_depth:
                depth_gt = self.get_depth(folder, frame_index, side, do_flip)
                depth_gt = np.expand_dims(depth_gt, 0)
                depth_gt = torch.from_numpy(depth_gt.astype(np.float32))
                inputs[("gt_depths")].append(depth_gt)

        if do_color_aug:
            color_aug = self._setup_augmentation_parameters()

        else:

            def identity(x):
                return x

            color_aug = identity

        self.preprocess(inputs, color_aug)

        for idx, i in enumerate(self.frame_idxs):
            del inputs[("color", idx - 1, -1)]
            del inputs[("color_augmented", idx - 1, -1)]

        inputs["img_file"] = self.filenames[index]

        inputs["K"] = torch.from_numpy(self.K)
        inputs["inv_K"] = torch.from_numpy(self.inv_K)
        return inputs

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError


class KITTIDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders"""

    def __init__(self, *args, **kwargs):
        super(KITTIDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size.
        # To normalize you need to scale the first row by 1 / image_width and the second row
        # by 1 / image_height. Monodepth2 assumes a principal point to be exactly centered.
        # If your principal point is far from the center you might need to disable the horizontal
        # flip augmentation.
        self.K = np.array([[0.58, 0, 0.5, 0], [0, 1.92, 0.5, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        self.K = self.K[:3, :3]
        self.K[0, :] *= self.width
        self.K[1, :] *= self.height
        self.inv_K = np.linalg.pinv(self.K)

        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path, scene_name, "velodyne_points/data/{:010d}.bin".format(int(frame_index))
        )

        return os.path.isfile(velo_filename)

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class KITTIRAWDataset(KITTIDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth"""

    def __init__(self, *args, **kwargs):
        super(KITTIRAWDataset, self).__init__(*args, **kwargs)

        if self.dedup_threshold > 0:
            print(f"Before deduping {len(self.filenames)}")
            self.filenames = self._dedup()
            print(f"After deduping {len(self.filenames)}")

    def _dedup(self):
        # Deduping takes a while. Hence, we cache the deduped filenames
        # If the deduping had already been done, we load it from the cached file
        cache_dir = "cache"
        filename = f"kitti_{self.dedup_threshold}.pkl"
        cached_path = os.path.join(cache_dir, filename)
        if os.path.exists(cached_path):
            with open(cached_path, "rb") as f:
                new_filenames = pickle.load(f)
                print("Using cached deduped files..")
                return new_filenames

        # Now dedup
        print("deduping...")

        def calcL1(img1, img2):
            return torch.mean(torch.abs(img1 - img2))

        new_filenames = []
        for index in tqdm(range(len(self.filenames))):
            folder, frame_index, side = self._get_metadata(index)
            frame1 = self.to_tensor(self.get_color(folder, frame_index - 1, side, do_flip=False))
            frame2 = self.to_tensor(self.get_color(folder, frame_index, side, do_flip=False))
            l1 = calcL1(frame1, frame2)
            if l1 > self.dedup_threshold:
                new_filenames.append(self.filenames[index])

        # now cache deduped filenames
        os.makedirs(cache_dir, exist_ok=True)
        pickle.dump(new_filenames, open(cached_path, mode="wb"))

        return new_filenames

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        calib_path = os.path.join(self.data_path, folder.split("/")[0])

        velo_filename = os.path.join(
            self.data_path, folder, "velodyne_points/data/{:010d}.bin".format(int(frame_index))
        )

        depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode="constant"
        )

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
