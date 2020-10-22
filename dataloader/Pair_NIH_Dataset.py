# import sys
# sys.path.insert(1, '/home/piu/Reg_pytorch_new')
import torch
from torch.utils import data

import csv
import numpy as np
import skimage.io as io
from skimage.transform import resize
import cv2
import PIL.Image as Image
class Pair_NIH_Dataset(data.Dataset):
    def __init__(self, set='train', seg=False, transforms=None):
        self.transforms = transforms
        self.seg = seg

        csv_name_s = '/home/piu/Reg_pytorch_new/dataloader/moving.csv'
        csv_name_t = '/home/piu/Reg_pytorch_new/dataloader/fixed.csv'
        if self.seg:
            csv_name_s = '/home/piu/Reg_pytorch_new/dataloader/moving_seg.csv'
            csv_name_t = '/home/piu/Reg_pytorch_new/dataloader/fixed_seg.csv'

        with open(csv_name_s) as f:
            self.list_IDs_s = list(csv.DictReader(f))
        with open(csv_name_t) as f:
            self.list_IDs_t = list(csv.DictReader(f))

    def __len__(self):
        return len(self.list_IDs_s)

    def __getitem__(self, idx):
        # Select sample
        ID_s = self.list_IDs_s[idx]['id']
        ID_t = self.list_IDs_t[idx]['id']

        # Load data
        moving_image = cv2.imread('/home/piu/Reg_pytorch_new/data_crop/moving_hm/' + ID_s, 0)
        fixed_image = cv2.imread('/home/piu/Reg_pytorch_new/data_crop/fixed/' + ID_t, 0)

        if self.seg:
            moving_seg = cv2.imread('/home/piu/Reg_pytorch_new/data_crop/moving_voc/SegmentationClassPNG/' + ID_s)
            fixed_seg = cv2.imread('/home/piu/Reg_pytorch_new/data_crop/fixed_voc/SegmentationClassPNG/' + ID_t)
            moving_seg_arg = np.argmax(moving_seg, axis=2)
            fixed_seg_arg = np.argmax(fixed_seg, axis=2)

        # Data augmentation
        if self.transforms:
            moving_image_tensor = self.transforms(moving_image)
            fixed_image_tensor = self.transforms(fixed_image)
            if self.seg:
                moving_seg_tensor = self.transforms(moving_seg_arg)
                fixed_seg_tensor = self.transforms(fixed_seg_arg)

        if self.seg:
            return moving_image_tensor, fixed_image_tensor, moving_seg_tensor, fixed_seg_tensor, ID_s, ID_t
        else:
            return moving_image_tensor, fixed_image_tensor, ID_s, ID_t
