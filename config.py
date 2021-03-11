#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
  @Email:  guo@locationmind.com
  @Copyright: guo-zhiling
  @License: MIT
"""
from utils.datasets import *
import argparse

class Seg_train(object):
    """
        Dataset setting for GUO
    """

    def __init__(self, args):     
        self.train = SegLS(args.train_data, args.is_multi, split="train", data_aug=True)
        self.val = SegLS(args.train_data, args.is_multi, split="val")
        self.test = SegLS(args.train_data, args.is_multi, split="test")
        self.in_ch = 3
        self.out_ch = 1
        self.data_dir = self.train.data_dir
        if args.is_multi:
            self.out_ch = self.train.nb_class
            self.class_names = self.train.class_names

class Seg_test(object):
    """
        Dataset setting for GUO
    """
    def __init__(self, args):     
        self.all = SegLS(args.test_data, args.is_multi, args.data_aug, split="all")
        self.in_ch = 3
        self.out_ch = 1
        self.data_dir = self.all.data_dir
        if args.is_multi:
            self.out_ch = self.all.nb_class
            self.class_names = self.all.class_names

class Seg_trans(object):
    """
        Dataset setting for GUO
    """
    def __init__(self, args):     
        self.train = SegLS(args.trans_data, args.is_multi, split="train", data_aug=True)
        self.val = SegLS(args.trans_data, args.is_multi, split="val")
        self.test = SegLS(args.trans_data, args.is_multi, split="test")
        self.in_ch = 3
        self.out_ch = 1
        self.data_dir = self.train.data_dir
        if args.is_multi:
            self.out_ch = self.train.nb_class
            self.class_names = self.train.class_names


class SRSeg_train(object):
    """
        Dataset setting for GUO
    """

    def __init__(self, args):
        self.train = SegLR(args.train_data, args.is_multi, split="train", data_aug=True)
        self.val = SegLR(args.train_data, args.is_multi, split="val")
        self.test = SegLR(args.train_data, args.is_multi, split="test")
        self.in_ch = 3
        self.out_ch = 1
        self.data_dir = self.train.data_dir
        if args.is_multi:
            self.out_ch = self.train.nb_class
            self.class_names = self.train.class_names

class SRSeg_test(object):
    """
        Dataset setting for GUO
    """
    def __init__(self, args):
        self.all = SegLR(args.test_data, args.is_multi, split="all")
        self.in_ch = 3
        self.out_ch = 1
        self.data_dir = self.all.data_dir
        if args.is_multi:
            self.out_ch = self.all.nb_class
            self.class_names = self.all.class_names

class SRSeg_trans(object):
    """
        Dataset setting for GUO
    """
    def __init__(self, args):
        self.train = SegLR(args.trans_data, args.is_multi, split="train", transform=True)
        self.val = SegLR(args.trans_data, args.is_multi, split="val")
        self.test = SegLR(args.trans_data, args.is_multi, split="test")
        self.in_ch = 3
        self.out_ch = 1
        self.data_dir = self.train.data_dir
        if args.is_multi:
            self.out_ch = self.train.nb_class
            self.class_names = self.train.class_names


class SegNII_train(object):
    """
        Dataset setting for GUO
    """

    def __init__(self, args):
        # self.train = SegLS(args.train_data, args.is_multi, split="train", transform=True)
        self.train = SegLS(args.train_data, args.is_multi, split="train", data_aug=True)
        # self.train = SegLS(args.train_data, args.is_multi, args.data_aug, split="train")

        self.val = SegLS(args.train_data, args.is_multi, split="val")
        self.test = SegLS(args.train_data, args.is_multi, split="test")
        self.in_ch = 1
        self.out_ch = 1
        self.data_dir = self.train.data_dir
        if args.is_multi:
            self.out_ch = self.train.nb_class
            self.class_names = self.train.class_names

class SegNII_test(object):
    """
        Dataset setting for GUO
    """
    def __init__(self, args):
        self.all = SegLS(args.test_data, args.is_multi, split="all")
        self.in_ch = 1
        self.out_ch = 1
        self.data_dir = self.all.data_dir
        if args.is_multi:
            self.out_ch = self.all.nb_class
            self.class_names = self.all.class_names

class SegNII_trans(object):
    """
        Dataset setting for GUO
    """
    def __init__(self, args):
        self.train = SegLS(args.trans_data, args.is_multi, split="train", transform=True)
        self.val = SegLS(args.trans_data, args.is_multi, split="val")
        self.test = SegLS(args.trans_data, args.is_multi, split="test")
        self.in_ch = 1
        self.out_ch = 1
        self.data_dir = self.train.data_dir
        if args.is_multi:
            self.out_ch = self.train.nb_class
            self.class_names = self.train.class_names

if __name__ == "__main__":
    # ====================== parameter initialization ======================= #
    parser = argparse.ArgumentParser(description='ArgumentParser')
    parser.add_argument('-is_multi', type=lambda x: (str(x).lower() == 'true'), default=False,
                        help='multi-class or not')
    parser.add_argument('-train_data', type=str, default='satellite-rand',
                        help='training data dir name')
    parser.add_argument('-test_data', type=str, default='satellite-str',
                        help='testing data dir name')
    args = parser.parse_args()
    parser.add_argument('-trans_data', type=str, default='map_trans-rand',
                        help='testing data dir name')
    dataset_train=Seg_train(args)
    dataset_test=Seg_test(args)
    dataset_trans=Seg_trans(args)
    

