#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Email:  guo@locationmind.com
  @Copyright: guo-zhiling
  @License: MIT
"""

import numpy as np
import os
from skimage.io import imread, imsave
import argparse
import sys
sys.path.append('./utils')
import warnings
warnings.simplefilter('ignore', UserWarning)

Utils_DIR = os.path.dirname(os.path.abspath(__file__))

def main(args):
    anno_dir = os.path.join(Utils_DIR, '../src', 'anno') 
    anno_raw_dir = os.path.join(anno_dir, 'raw') 
    anno_mature_dir = os.path.join(anno_dir, 'mature')
    
    anno_name = args.anno_name
    
    layers = os.listdir(os.path.join(anno_raw_dir, anno_name))
    anno_mature = []
    
    for i in range(len(layers)):
        layer_path = os.path.join(anno_raw_dir, anno_name, str(i) + '.tif')
        layer = imread(layer_path)
        layer_1d = layer[:,:,1]
        layer_1d[layer_1d==255] = 0
        layer_std = np.clip(layer_1d, 0, 1)
        anno_mature.append(layer_std)
    anno_mature = np.transpose(np.asarray(anno_mature),(1, 2, 0)).astype('float32')
    imsave(os.path.join(anno_mature_dir, anno_name + '.tif'), anno_mature)

if __name__ == "__main__":
    # ====================== parameter initialization ======================= #
    parser = argparse.ArgumentParser(description='ArgumentParser')
    parser.add_argument('-anno_name', type=str, default='shinjuku',
                        help='annotation file name')
    args = parser.parse_args()
    main(args)