#!/usr/bin/env python3
import torch
import torch.utils.data as data
import os
from PIL import Image
from .utilities.data_transforms import (
    RandomFlip,
    RandomCrop,
    RandomScale,
    Normalize,
    Resize,
    Compose,
)


class COCOSegmentation(torch.utils.data.Dataset):
    def __init__(
        self,
        train=True,
        scale=(0.5, 1.0),
        crop_size=(513, 513),
        ignore_idx=255,
        coco_root_dir="./vision_datasets/coco_preprocess/",
    ):
        super(COCOSegmentation, self).__init__()
        self.train = train
        if self.train:
            data_file = os.path.join(coco_root_dir, "train_2017.txt")
        else:
            data_file = os.path.join(coco_root_dir, "val_2017.txt")

        self.images = []
        self.masks = []
        with open(data_file_coco, "r") as lines:
            for line in lines:
                rgb_img_loc = coco_root_dir + os.sep + line.split()[0]
                label_img_loc = coco_root_dir + os.sep + line.split()[1]
                assert os.path.isfile(rgb_img_loc)
                assert os.path.isfile(label_img_loc)
                self.images.append(rgb_img_loc)
                self.masks.append(label_img_loc)

        if isinstance(crop_size, tuple):
            self.crop_size = crop_size
        else:
            self.crop_size = (crop_size, crop_size)

        if isinstance(scale, tuple):
            self.scale = scale
        else:
            self.scale = (scale, scale)

        self.train_transforms, self.val_transforms = self.transforms()
        self.ignore_idx = ignore_idx

    def transforms(self):
        train_transforms = Compose(
            [
                RandomFlip(),
                RandomScale(scale=self.scale),
                RandomCrop(crop_size=self.crop_size),
                Normalize(),
            ]
        )
        val_transforms = Compose([Resize(size=self.crop_size), Normalize()])
        return train_transforms, val_transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        try:
            rgb_img = Image.open(self.images[index]).convert("RGB")
            label_img = Image.open(self.masks[index])

            if self.train:
                rgb_img, label_img = self.train_transforms(rgb_img, label_img)
            else:
                rgb_img, label_img = self.val_transforms(rgb_img, label_img)

            return rgb_img, label_img
        except:
            for index in range(len(self.images)):
                try:
                    rgb_img = Image.open(self.images[index]).convert("RGB")
                    label_img = Image.open(self.masks[index])

                    if self.train:
                        rgb_img, label_img = self.train_transforms(rgb_img, label_img)
                    else:
                        rgb_img, label_img = self.val_transforms(rgb_img, label_img)

                    return rgb_img, label_img
                except:
                    continue
