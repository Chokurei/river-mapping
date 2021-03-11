#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
  @CreateTime:   2018-01-26T16:50:00+09:00
  @Email:  guangmingwu2010@gmail.com
  @Copyright: go-hiroaki
  @License: MIT
"""
import sys
sys.path.append('./utils')
import os
import pandas as pd
from torch.utils import data
from skimage.io import imread
import matplotlib.pyplot as plt
import argparse
from utils.transformer import *
from torchvision import transforms
from skimage import transform

Utils_DIR = os.path.dirname(os.path.abspath(__file__))

class SegDataset(data.Dataset):
    """ 'guo' dataset object
    args:
        partition: (str) partition of the data ['train', 'test']
        split: (str) split of the data ['train', 'val', 'testl']
    """

    def __init__(self, data_dir, is_multi, split='train', data_aug=True):
        self.data_dir = data_dir
        self.is_multi = is_multi
        self.split = split
        self.data_aug = data_aug
        
        self.dataset = os.path.join(
            Utils_DIR, '../dataset', self.data_dir)

        self._landpath = os.path.join(self.dataset, 'image', '%s')
        self._segpath = os.path.join(self.dataset, 'label', '%s')

        # get image ids
        infos = pd.read_csv(os.path.join(
            self.dataset, '{}.csv'.format(self.split)))
        self.ids = infos['id'].tolist()
        # get label reference
        data_name = self.data_dir.split('-')[0]
        
        if self.is_multi:  
            self.refs = pd.read_csv(os.path.join(
                self.dataset, '../classes', data_name + '-class.csv'))
            self.nb_class = self.refs.shape[0]
            self.class_names = self.refs['Land-cover'].tolist()

    def __len__(self):
        return len(self.ids)

    def show(self, idx):
        img_land = imread(self._landpath % self.ids[idx])
        img_seg = imread(self._segpath % self.ids[idx])

        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(11, 5))
        f.suptitle('Sample-{} in guo Dataset'.format(idx))
        ax1.imshow(img_land)
        ax1.set_title('Land Sample')
        ax2.imshow(img_seg[:,:,:3])
        ax2.set_title('Segmap Sample(front 3 channel)')
        plt.show()

class SegLS(SegDataset):
    """
        return 'Land-Segmap' of guoDataset
    """
    def __getitem__(self, idx):
        img_id = self.ids[idx]
        # img_land = imread(self._landpath % img_id)[:, :, :3]
        img_land = imread(self._landpath % img_id)
        if len(img_land.shape) == 2:
            img_land = (np.expand_dims(img_land, -1) / 255).astype('float32')
        else:
            img_land = img_land[:, :, :3]
        img_seg = imread(self._segpath % img_id)
        self.down_scale = 1

        if self.down_scale != 1:
            num_rows, num_cols = img_land.shape[0], img_land.shape[1]

            # down-sampling
            num_rows_new = num_rows // self.down_scale
            num_cols_new = num_cols // self.down_scale
            img_land = transform.resize(img_land, (num_rows_new, num_cols_new), preserve_range=True)
            img_seg = transform.resize(img_seg, (num_rows_new, num_cols_new), preserve_range=True)

            # up-sampling
            img_land = transform.resize(img_land, (num_rows, num_cols), preserve_range=True)
            img_seg = transform.resize(img_seg, (num_rows, num_cols), preserve_range=True)

            img_seg[img_seg <= 127] = 0
            img_seg[img_seg > 127] = 255

        # data augmentation
        # TODO: for multi-class
        if self.split == 'train' and self.data_aug and not self.is_multi:
            sample = {'src': img_land, 'tar': img_seg}

            # condition1: rescale
            condition1 = random.random()
            composed_1 = transforms.Compose([Rescale(256),
                                            RandomCrop(224)])
            # if not, set < 0
            if condition1 < 0.5:
                sample = composed_1(sample)

            # condition2: filp
            # condition in RandomVerticalFlip and RandomHorizontalFlip is 0.5
            condition2 = random.random()
            composed_2 = transforms.Compose([RandomVerticalFlip(isArray=True),
                                             RandomHorizontalFlip(isArray=True)])
            if condition2 < 0.5 :
                sample = composed_2(sample)
            img_land, img_seg = sample['src'], sample['tar']

        if len(img_seg.shape) == 2:
            img_seg = (np.expand_dims(img_seg, -1) / 255).astype('float32')
        else:
            img_seg = (img_seg / 255).astype('float32')

        img_land = (img_land / 255).astype('float32').transpose((2, 0, 1))
        img_seg = img_seg.transpose((2, 0, 1))

        return img_land, img_seg

class SegLR(SegDataset):
    """
        return LR images and related annotation
    """
    def __getitem__(self, idx):
        self.upscale_factor = 2
        img_id = self.ids[idx]
        img_land = imread(self._landpath % img_id)[:, :, :3]
        img_seg = imread(self._segpath % img_id)
        num_rows_new = img_land.shape[0] // self.upscale_factor
        num_cols_new = img_land.shape[1] // self.upscale_factor

        img_land_scaled = transform.resize(img_land, (num_rows_new, num_cols_new), preserve_range=True)
        # img_seg = transform.resize(img_seg, (num_rows_new, num_cols_new), preserve_range=True)
        # img_seg[img_seg <= 127] = 0
        # img_seg[img_seg > 127] = 255

        # data augmentation
        # TODO: for multi-class
        if self.split == 'train' and self.data_aug and not self.is_multi:
            sample = {'src': img_land, 'tar': img_seg}

            # condition1: rescale
            condition1 = random.random()
            composed_1 = transforms.Compose([Rescale(256),
                                            RandomCrop(224)])
            # if not, set < 0
            if condition1 < 0:
                sample = composed_1(sample)

            # condition2: filp
            # condition in RandomVerticalFlip and RandomHorizontalFlip is 0.5
            condition2 = random.random()
            composed_2 = transforms.Compose([RandomVerticalFlip(isArray=True),
                                             RandomHorizontalFlip(isArray=True)])
            if condition2 < 0 :
                sample = composed_2(sample)
            img_land, img_seg = sample['src'], sample['tar']

        if len(img_seg.shape) == 2:
            img_seg = (np.expand_dims(img_seg, -1) / 255).astype('float32')
        else:
            img_seg = (img_seg / 255).astype('float32')

        img_land = (img_land / 255).astype('float32').transpose((2, 0, 1))
        img_land_scaled = (img_land_scaled / 255).astype('float32').transpose((2, 0, 1))
        img_seg = img_seg.transpose((2, 0, 1))

        return img_land_scaled, img_land, img_seg

if __name__ == "__main__":
    # ====================== parameter initialization ======================= #
    parser = argparse.ArgumentParser(description='ArgumentParser')
    parser.add_argument('-partition', type=str, default='train',
                        help='partition within ["train", "test", "all", "val"]')
    parser.add_argument('-is_multi', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help='multi-class or not')
    parser.add_argument('-split', type=str, default='train',
                        help='split of the data within ["train","val","test"]')
    args = parser.parse_args()