"""
Source: https://github.com/princeton-vl/RAFT/blob/master/core/datasets.py
"""
import os
import random
from glob import glob
from os.path import splitext

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

TAG_CHAR = np.array([202021.25], np.float32)


def readFlowKITTI(filename):
    flow = cv2.imread(filename, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    flow = flow[:, :, ::-1].astype(np.float32)
    flow, valid = flow[:, :, :2], flow[:, :, 2]
    flow = (flow - 2**15) / 64.0
    return flow, valid


def read_gen(file_name, pil=False):
    ext = splitext(file_name)[-1]
    if ext == ".png" or ext == ".jpeg" or ext == ".ppm" or ext == ".jpg":
        return Image.open(file_name)


class FlowDataset(data.Dataset):
    def __init__(self, height, width, sparse=False):
        self.augmentor = None
        self.sparse = sparse

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []

        self.resize = transforms.Resize((height, width), antialias=True)
        self.height = height
        self.width = width

    def __getitem__(self, index):
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None

        flow, valid = readFlowKITTI(self.flow_list[index])

        # read and scale intrinsic matrix
        with open(self.intrinsics_path[index], "r") as f:
            calib_data = f.readlines()
            P_rect_02 = np.array([float(x) for x in calib_data[25].split()[1:]]).reshape(3, 4)
        intrinsics = P_rect_02[:3, :3]
        intrinsics[:, 0] = intrinsics[:, 0] / 1242 * self.width
        intrinsics[:, 1] = intrinsics[:, 1] / 375 * self.height
        intrinsics_inv = np.linalg.pinv(intrinsics)

        img1 = read_gen(self.image_list[index][0])
        img2 = read_gen(self.image_list[index][1])

        img1 = self.resize(img1)
        img2 = self.resize(img2)

        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        # putting stuff in dictionaries
        data_dict = {}
        data_dict[("color", -1, 0)] = img1 / 255.0
        data_dict[("color", 0, 0)] = img2 / 255.0
        # augmented and color are the same in this case
        data_dict[("color_augmented", -1, 0)] = img1 / 255.0
        data_dict[("color_augmented", 0, 0)] = img2 / 255.0
        data_dict["flow_gt"] = flow * valid.float()
        data_dict["mask"] = valid.float()
        data_dict["K"] = torch.from_numpy(intrinsics).float()
        data_dict["inv_K"] = torch.from_numpy(intrinsics_inv).float()

        return data_dict

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self

    def __len__(self):
        return len(self.image_list)


class KITTIFlowData(FlowDataset):
    def __init__(self, height, width, split="training", occ_type="occ", root="datasets/KITTI"):
        assert occ_type in ["occ", "noc"]
        super(KITTIFlowData, self).__init__(height=height, width=width, sparse=True)
        if split == "testing":
            self.is_test = True

        root = os.path.join(root, split)
        images1 = sorted(glob(os.path.join(root, "image_2/*_10.png")))
        images2 = sorted(glob(os.path.join(root, "image_2/*_11.png")))
        self.intrinsics_path = sorted(glob(os.path.join(root, "calib_cam_to_cam/*.txt")))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split("/")[-1]
            self.extra_info += [[frame_id]]
            self.image_list += [[img1, img2]]

        if split == "training":
            self.flow_list = sorted(glob(os.path.join(root, f"flow_{occ_type}/*_10.png")))
            if len(self.flow_list) == 0:
                raise ValueError(f"No flow files found in {root}.")
