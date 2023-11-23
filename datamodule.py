from __future__ import absolute_import, division, print_function

import argparse
import os
from typing import List

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from datasets import KITTIFlowData, KITTIRAWDataset
from utils.kitti_utils import readlines


class KITTIRAWDatamodule(pl.LightningDataModule):
    """Datamodule for the KITTRAW dataset. This datamodule loads the KITTI Raw dataset.
    Optionally, it can also load the KITTI Flow dataset for evaluation.
    It uses the KITTI Flow dataset by default when in testing, validation mode.
    """

    def __init__(
        self,
        path: str,
        img_ext: str,
        batch_size: int,
        num_workers: int,
        num_scales: int = 1,
        height: int = 128,
        width: int = 416,
        split="eigen_full",
        side: str = None,
        dedup_threshold: float = 0,
        evaluateKITTIFlow: bool = False,
        kitti_flow_path: str = None,
        stride=1,  # stride for sampling frames, (sample: x, x+stride)
        **kwargs,
    ):
        super().__init__()
        self.data_path = path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_ext = img_ext
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.evaluateKITTIFlow = evaluateKITTIFlow
        self.kitti_flow_path = kitti_flow_path
        self.dedup_threshold = dedup_threshold
        self.stride = stride

        if side is not None and side not in ["l", "r"]:
            raise ValueError(f"side must be either 'l' or 'r' or None, but got {side}")

        fpath = os.path.join("kitti_splits", split, "{}_files.txt")
        self.train_filenames = self._filter_frames(readlines(fpath.format("train")), side)
        self.val_filenames = self._filter_frames(readlines(fpath.format("val")), side)
        self.test_filenames = self._filter_frames(readlines(fpath.format("test")), side)

    def _filter_frames(self, filenames: List[str], side: str = None) -> List[str]:
        """Filters filenames based on if they have the given side. Also, removes 0 indexed frames
        as they have no previous frame. Given that we sample only 2 frames, we can't use the 0th frame.
        """
        if side is None:
            return [file for file in filenames if int(file.split(" ")[-2]) > self.stride - 1]
        return [
            file for file in filenames if file.split(" ")[-1] == side and int(file.split(" ")[-2]) > self.stride - 1
        ]

    def setup(self, stage=None):
        self.train_dataset = KITTIRAWDataset(
            data_path=self.data_path,
            filenames=self.train_filenames,
            height=self.height,
            width=self.width,
            num_scales=self.num_scales,
            is_train=True,
            img_ext=self.img_ext,
            dedup_threshold=self.dedup_threshold,
            frame_idxs=[-self.stride, 0],
        )
        self.val_dataset = KITTIRAWDataset(
            data_path=self.data_path,
            filenames=self.val_filenames,
            height=self.height,
            width=self.width,
            num_scales=self.num_scales,
            is_train=False,
            img_ext=self.img_ext,
            dedup_threshold=0,  # no dedup for val
            frame_idxs=[-self.stride, 0],
        )
        self.test_dataset = KITTIRAWDataset(
            data_path=self.data_path,
            filenames=self.test_filenames,
            height=self.height,
            width=self.width,
            num_scales=self.num_scales,
            is_train=False,
            img_ext=self.img_ext,
            dedup_threshold=0,  # no dedup for test
            frame_idxs=[-self.stride, 0],
        )

        if self.evaluateKITTIFlow:
            self.kitti_flow_occ = KITTIFlowData(
                root=self.kitti_flow_path,
                split="training",
                occ_type="occ",
                height=self.height,
                width=self.width,
            )
            self.kitti_flow_noc = KITTIFlowData(
                root=self.kitti_flow_path,
                split="training",
                occ_type="noc",
                height=self.height,
                width=self.width,
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def val_dataloader(self):
        kitti_dl = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        if self.evaluateKITTIFlow:
            kitti_flow_occ = DataLoader(
                dataset=self.kitti_flow_occ,
                batch_size=1,  # due to varying sizes of images
                shuffle=True,
                num_workers=1,
                pin_memory=True,
                drop_last=False,
            )
            kitti_flow_noc = DataLoader(
                dataset=self.kitti_flow_noc,
                batch_size=1,  # due to varying sizes of images
                shuffle=True,
                num_workers=1,
                pin_memory=True,
                drop_last=False,
            )
            return [kitti_dl, kitti_flow_occ, kitti_flow_noc]
        return kitti_dl

    def test_dataloader(self):
        kitti_dl = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        kitti_flow_occ = DataLoader(
            dataset=self.kitti_flow_occ,
            batch_size=1,  # due to varying sizes of images
            shuffle=False,
            num_workers=1,
            pin_memory=True,
            drop_last=False,
        )
        kitti_flow_noc = DataLoader(
            dataset=self.kitti_flow_noc,
            batch_size=1,  # due to varying sizes of images
            shuffle=False,
            num_workers=1,  # since batch_size = 1
            pin_memory=True,
            drop_last=False,
        )
        return [kitti_dl, kitti_flow_occ, kitti_flow_noc]


# Testing datamodule
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="path to kitti raw dataset")
    parser.add_argument("--kitti_flow_path", type=str, required=True, help="path to kitti flow dataset")
    args = parser.parse_args()

    # test kitti raw datamodule
    kitti_raw_datamodule = KITTIRAWDatamodule(
        path=args.data_path,
        img_ext=".jpg",
        batch_size=4,
        num_workers=0,
        height=128,
        width=416,
        num_scales=1,
        split="eigen_full",
        side="l",
        dedup_threshold=0,
        evaluateKITTIFlow=True,
        kitti_flow_path=args.kitti_flow_path,
    )
    kitti_raw_datamodule.setup()

    train_dl = kitti_raw_datamodule.train_dataloader()
    val_dl, flow_occ_dl, _ = kitti_raw_datamodule.val_dataloader()

    for batch in train_dl:
        print(batch.keys())
        print(batch["color", -1, 0].shape)
        print(batch["color", 0, 0].shape)
        print(batch["color_augmented", -1, 0].shape)
        print(batch["color_augmented", 0, 0].shape)
        print(batch["img_file"])

        break

    for batch in flow_occ_dl:
        print(batch.keys())
        print(batch["color", -1, 0].shape)
        print(batch["color", 0, 0].shape)
        print(batch["color_augmented", -1, 0].shape)
        print(batch["color_augmented", 0, 0].shape)
        print(batch["flow_gt"].shape)
        break
