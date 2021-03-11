#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
  @CreateTime:   2018-01-26T16:50:00+09:00
  @Email:  guangmingwu2010@gmail.com
  @Copyright: go-hiroaki
  @License: MIT
"""
import os
import sys
sys.path.append('./utils')
import glob
import json
import shutil
import random
import itertools
import numpy as np
import pandas as pd

from skimage.io import imread, imsave
#from tifffile import imsave
from skimage.transform import resize
import argparse
import warnings

warnings.simplefilter('ignore', UserWarning)

Utils_DIR = os.path.dirname(os.path.abspath(__file__))

class Extractor(object):
    def __init__(self):
        print("Baic Extractor")

    def save_infos(self, df):
        all_file = os.path.join(self.save_dir, 'all.csv')
        df.to_csv(all_file, index=False)
        if args.data_usage == 'train' or args.data_usage == 'trans':
            nb_list = list(range(df.shape[0]))
            tv_edge = int(df.shape[0] * self.split[0])
            vt_edge = int(df.shape[0] * (1 - self.split[2]))
            # shuffle list
            random.shuffle(nb_list)
            train_df = df.iloc[nb_list[:tv_edge], :]
            train_df.to_csv(os.path.join(self.save_dir, 'train.csv'), index=False)
            val_df = df.iloc[nb_list[tv_edge:vt_edge], :]
            val_df.to_csv(os.path.join(self.save_dir, 'val.csv'), index=False)
            test_df = df.iloc[nb_list[vt_edge:], :]
            test_df.to_csv(os.path.join(self.save_dir, 'test.csv'), index=False)

    def save_slices(self, img_slices, folder):
        if not os.path.exists(os.path.join(self.save_dir, folder)):
            os.mkdir(os.path.join(self.save_dir, folder))
        for i in range(len(img_slices)):
            imsave(os.path.join(self.save_dir, folder, "img_{0}.tif".format(i)),
                   img_slices[i])
        return 0
    
    def read_ids_info(self, data_dir, info):
        ids = []
        with open(os.path.join(data_dir, info), 'r') as f:
            for line in f.readlines():
                if line == '\n':
                    continue
                else:
                    ids.append(line.strip())
        return ids
        

class singleExtractor(Extractor):
    def __init__(self, args, 
                 stride=None,
                 threshold=0.1,
                 edge_buffer=0.1,):
        
        self.data_dir = os.path.join(Utils_DIR, '../src', args.data)

        self.nb_crop = args.nb_crop
        self.img_rows = args.img_rows
        self.img_cols = args.img_cols
        self.split = args.split

        self._srcpath = os.path.join(self.data_dir, 'image', '%s')
        self._tarpath = os.path.join(self.data_dir, 'label', '%s')

        # get image ids
        if args.data_usage == 'train':
            self.ids = self.read_ids_info(self.data_dir, 'train.txt')
        elif args.data_usage == 'test':
            self.ids = self.read_ids_info(self.data_dir, 'test.txt')
        else:
            self.ids = self.read_ids_info(self.data_dir, 'trans.txt')
            
        self.stride = stride if stride else self.img_rows
  
        self.threshold = threshold
        self.edge_buffer = edge_buffer

    def _read_tfw(self, tfw):
        with open(tfw) as f:
            params = []
            for line in f.readlines():
                param = float(line.strip())
                params.append(param)
            return params

    def _img_align(self, img_id):
        self.src_img = imread(self._srcpath % img_id)
        self.tar_img = imread(self._tarpath % img_id)
        self.src_path = self._srcpath % img_id
        self.tar_path = self._tarpath % img_id
        print(self.src_path)
        # extend tar image dimension
        # read coordinate params info from tfw
        assert os.path.exists(self.src_path.replace(
            '.tif', '.tfw')), "Source tfw doesn't exist, try slicing mode instead."
        assert os.path.exists(self.tar_path.replace(
            '.tif', '.tfw')), "Target tfw doesn't exist, try slicing mode instead."
        assert os.path.exists(self.tar_path.replace(
            '.tif', '.geojson')), "Target geojson doesn't exist, try slicing mode instead."
        tar_params = self._read_tfw(self.tar_path.replace('.tif', '.tfw'))
        src_params = self._read_tfw(self.src_path.replace('.tif', '.tfw'))
        with open(self.tar_path.replace('.tif', '.geojson'), 'r') as f:
            topos = json.load(f)['features']
        self.topos = topos
        assert len(src_params) == len(
            tar_params), "Number of params should be equal."
        assert src_params[:4] == tar_params[:4], "Resolution should be the same."

        # pixel scale along with x and y axis
        self.x_axis_scale = src_params[0]
        self.y_axis_scale = src_params[3]
        # x, y coordinate for src and tar
        x_1, y_1 = src_params[4:]
        x_2, y_2 = tar_params[4:]
        # initialize bounds[minx, miny, maxx, maxy] of consistent area
        bounds = [0, 0, 0, 0]
        # coumpute shifting pixels
        x_pixel = int((x_2 - x_1) / self.x_axis_scale)
        y_pixel = int((y_2 - y_1) / self.y_axis_scale)

        # alignment by original point (0, 0)
        # align by x-axis
        if x_pixel >= 0:
            # chop src image in x-axis -> cols
            self.src_img = self.src_img[:, x_pixel:]
            bounds[0] = x_2
        else:
            # chop ouline image in x-axis -> cols
            self.tar_img = self.tar_img[:, abs(x_pixel):]
            bounds[0] = x_1
        # align by y-axis
        if y_pixel >= 0:
            # chop src image in y-axis -> rows
            self.src_img = self.src_img[y_pixel:, :]
            bounds[3] = y_2
        else:
            # chop tar image in y-axis -> rows
            self.tar_img = self.tar_img[abs(y_pixel):, :]
            bounds[3] = y_1

        # crop max consistent area
        rows_l, cols_l = self.src_img.shape[:2]
        rows_o, cols_o = self.tar_img.shape[:2]
        # crop by rows
        self.tar_img = self.tar_img[:min(rows_l, rows_o), :]
        self.src_img = self.src_img[:min(rows_l, rows_o), :]
        # crop by cols
        self.tar_img = self.tar_img[:, :min(cols_l, cols_o)]
        self.src_img = self.src_img[:, :min(cols_l, cols_o)]
        bounds[2] = self.x_axis_scale * min(cols_l, cols_o) + bounds[0]
        bounds[1] = self.y_axis_scale * min(rows_l, rows_o) + bounds[3]
        self.bounds = bounds

    def _get_bounds(self, polygon):
        # return bounds(minx, miny, maxx, maxy) of the polygon
        bounds = [0, 0, 0, 0]
        poly = np.squeeze(np.array(polygon['geometry']['coordinates']))
        try:
            bounds[:2] = poly.min(axis=0)
            bounds[2:] = poly.max(axis=0)
            return bounds
        except:
            # print("wrong polygon")
            return None

    def extract_by_vector(self):
        print("Processing via vector")
        # make save dirs
        self.save_dir = os.path.join(
            Utils_DIR, '../dataset', args.data + '-vec')
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
        os.makedirs(self.save_dir)
        X_slices, y_slices = [], []
        posi_rates, ori_sizes = [], []
        # count number of polygon out of range
        counts = 0
        
        for img_id in self.ids:
            print("\t Image{}/{}: {}".format(self.ids.index(img_id)+1, len(self.ids), img_id))
            self._img_align(img_id)
        
            for topo in self.topos:
                # bounds = (minx, miny, maxx, maxy)
                try:
                    bounds = self._get_bounds(topo)
                    cen_pixel_x = int(
                        ((bounds[2] + bounds[0]) / 2 - self.bounds[0]) / self.x_axis_scale)
                    cen_pixel_y = int(
                        ((bounds[3] + bounds[1]) / 2 - self.bounds[3]) / self.y_axis_scale)
                    max_width = (bounds[2] - bounds[0]) / self.x_axis_scale
                    max_height = (bounds[1] - bounds[3]) / self.y_axis_scale
                    pixels = int((self.edge_buffer + 1) *
                                 max(max_width, max_height) / 2)
    
                    # crop building from src and tar with buffer
                    extract_src = self.src_img[cen_pixel_y - pixels:cen_pixel_y + pixels,
                                               cen_pixel_x - pixels:cen_pixel_x + pixels]
                    extract_tar = self.tar_img[cen_pixel_y - pixels:cen_pixel_y + pixels,
                                               cen_pixel_x - pixels:cen_pixel_x + pixels]
                    extract_src = resize(
                        extract_src, (self.img_rows, self.img_cols), mode='edge')
                    extract_tar = resize(
                        extract_tar, (self.img_rows, self.img_cols), mode='edge')
    
                    extract_src = (extract_src * 255).astype("uint8")
                    extract_tar = (extract_tar * 255).astype("uint8")
                    # denoise after resing image
                    extract_tar[extract_tar < 128] = 0
                    extract_tar[extract_tar >= 128] = 255
                    posi_rate = round(np.sum(extract_tar == 255) /
                                      (self.img_rows * self.img_cols), 3)
                    X_slices.append(extract_src)
                    y_slices.append(extract_tar)
                    posi_rates.append(posi_rate)
                    ori_sizes.append(2 * pixels)
                except:
                    counts += 1
                    print("\t Skip polygon which is out of range")
    
            print("\t Totally %d of ploygons are out of scope." % counts)
    
            _statistic = [len(y_slices), self.img_rows,
                          self.img_cols, np.mean(posi_rates)]
            _file = os.path.join(self.save_dir, 'statistic.csv')
            pd.DataFrame([_statistic],
                         columns=["nb-samples", "img_rows", "img_cols", "mean-posi"]).to_csv(_file, index=False)
    
            # save infos
            infos = pd.DataFrame(columns=['id', 'ori_size', 'posi_rate'])
            infos['id'] = ['img_{}.tif'.format(i) for i in range(len(y_slices))]
            infos['ori_size'] = ori_sizes
            infos['posi_rate'] = posi_rates
            self.save_infos(infos)
    
            # save slices
            self.save_slices(X_slices, "image")
            self.save_slices(y_slices, "label")


    def extract_by_stride_slide(self):
        print("Processing via sliding window with stride")
        # make save dirs
        self.save_dir = os.path.join(
            Utils_DIR, '../dataset', args.data + '-str')
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
        os.makedirs(self.save_dir)
        
        _statistics = []
        X_slices, y_slices = [], []
        ori_file, ori_size, ori_pos = [], [], []
        
        for img_id in self.ids:
            print("\t Image{}/{}: {}".format(self.ids.index(img_id)+1, len(self.ids), img_id))
            self.src_img = imread(self._srcpath % img_id)
            self.tar_img = imread(self._tarpath % img_id)
            assert self.src_img.shape[:2] == self.tar_img.shape[:2], \
                "Image dimension must be consistent."
            
            # extract slices from source and target imagery
            rows, cols = self.src_img.shape[:2]
            row_range = range(0, rows - self.img_rows, self.stride)
            col_range = range(0, cols - self.img_cols, self.stride)
            nb_samples = len(row_range) * len(col_range)
        
            print("\t \t Original: img_rows : {}; img_cols : {}".format(rows, cols))
            print("\t \t Original: nb_rows : {}; nb_cols : {}".format(
                len(row_range), len(col_range)))
            print("\t \t nb_crop : {}".format(nb_samples))
        
            for i, j in itertools.product(row_range, col_range):
                img_src = self.src_img[i:i + self.img_rows,
                                       j:j + self.img_cols]
                img_tar = self.tar_img[i:i + self.img_rows,
                                       j:j + self.img_cols]
                posi_rate = round(np.sum(img_tar == 255) /
                                  (self.img_rows * self.img_cols), 3)
                if posi_rate >= self.threshold:
                    X_slices.append(img_src)
                    y_slices.append(img_tar)
                    ori_file.append(img_id)
                    ori_size.append("{},{}".format(
                        len(row_range), len(col_range)))
                    ori_pos.append("{},{}".format(i, j))
            
            if args.data_usage == 'train' or args.data_usage == 'trans':
                _statistic = [img_id, nb_samples, len(row_range), len(col_range),
                              self.img_rows, self.img_cols, self.split]
            else:
                _statistic = [img_id, nb_samples, len(row_range), len(col_range),
                              self.img_rows, self.img_cols, 'no split']
            _statistics.append(_statistic)
                       
            self.save_slices(X_slices, "image")
            self.save_slices(y_slices, "label")
            
        _file = os.path.join(self.save_dir, 'statistic.csv')
        pd.DataFrame(_statistics,
                     columns=["img_name","nb-samples", "nb_rows", "nb_cols", \
                              "img_rows", "img_cols", "split"]).to_csv(_file, index=False)        
            
        # save infos
        infos = pd.DataFrame(columns=['img_name' ,'id', 'ori_size', 'ori_pos'])
        infos['id'] = ['img_{}.tif'.format(i) for i in range(len(y_slices))]
        infos['img_name'] = ori_file
        infos['ori_size'] = ori_size
        infos['ori_pos'] = ori_pos
        self.save_infos(infos)

    def extract_by_random_slide(self):
        print("Processing via randomly sliding window")
        self.save_dir = os.path.join(Utils_DIR, '../dataset', args.data + "-rand")
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
        os.makedirs(self.save_dir)
        
        _statistics = []
        X_slices, y_slices = [], []
        ori_file, ori_pos = [], []
        for img_id in self.ids:
            print("\t Image{}/{}: {}".format(self.ids.index(img_id)+1, len(self.ids), img_id))
            # self.src_img = imread(self._srcpath % img_id)
            # if not (self._srcpath % img_id).endswith('.bmp'):
            _img_id = img_id.split('.')[0]
            _img_id_bmp = _img_id + '.bmp'
            _img_id_png = _img_id + '.tif'
            try:
                self.src_img = imread(self._srcpath % _img_id_bmp)
            except:
                self.src_img = imread(self._srcpath % _img_id_png)
            try:
                self.tar_img = imread(self._tarpath % _img_id_bmp)
            except:
                self.tar_img = imread(self._tarpath % _img_id_png)

            # try:
            #     self.src_img = imread((self._srcpath % img_id).replace('.png', '.bmp'))
            # except:
            #     self.src_img = imread((self._srcpath % img_id))
            # try:
            #     self.tar_img = imread((self._tarpath % img_id).replace('.png', '.bmp'))
            # except:
            #     self.tar_img = imread((self._tarpath % img_id))

            assert self.src_img.shape[:2] == \
                self.tar_img.shape[:2], "Image dimension must be consistent."
            # extract slices from source and target imagery
            rows, cols = self.src_img.shape[:2]
            
            print("\t \t img_rows : {}; img_cols : {}".format(rows, cols))
            print("\t \t nb_crop : {}".format(self.nb_crop))

            for _ in range(self.nb_crop):
                i = random.randint(0, rows - self.img_rows)
                j = random.randint(0, cols - self.img_cols)
                img_src = self.src_img[i:i + self.img_rows,
                                       j:j + self.img_cols]
                img_tar = self.tar_img[i:i + self.img_rows,
                                       j:j + self.img_cols]
                
                posi_rate = round(np.sum(img_tar == 255) /
                                  (self.img_rows * self.img_cols), 3)
                if posi_rate >= self.threshold:
                    X_slices.append(img_src)
                    y_slices.append(img_tar)
                    ori_file.append(img_id)
                    ori_pos.append("{},{}".format(i, j))
                                                
            if args.data_usage == 'train' or args.data_usage == 'trans':
                _statistic = [img_id, self.nb_crop, \
                              self.img_rows, self.img_cols, self.split]
            else:
                _statistic = [img_id, self.nb_crop, \
                              self.img_rows, self.img_cols, 'no split']
            _statistics.append(_statistic)
            
        _file = os.path.join(self.save_dir, 'statistic.csv')
        pd.DataFrame(_statistics,
                     columns=["img_name","nb-samples", \
                              "img_rows", "img_cols", "split"]).to_csv(_file, index=False)        

        # save infos
        infos = pd.DataFrame(columns=['img_name', 'id',  'ori_pos'])
        infos['id'] = ['img_{}.tif'.format(i) for i in range(len(y_slices))]
        infos['img_name'] = ori_file
        infos['ori_pos'] = ori_pos
        self.save_infos(infos)
        
        # save slices
        self.save_slices(X_slices, "image")
        self.save_slices(y_slices, "label")

class multiExtractor(Extractor):
    """Image Data for preprocessing multi-label image
        multi-labal
    args:
        data: (str) root of the dataset e.g. 'Vaihingen'
        split: (float) split of train-val distribution
    """

    def __init__(self, args,
                 stride=None,):
        
        self.data_dir = os.path.join(Utils_DIR, '../src', args.data)
#        self.src_names = sorted(os.listdir(self.data_dir))
        self.nb_crop = args.nb_crop
        self.img_rows = args.img_rows
        self.img_cols = args.img_cols
        self.split = args.split

        self._srcpath = os.path.join(self.data_dir, 'image', '%s')
        self._tarpath = os.path.join(self.data_dir, 'label', '%s')

        # get image ids
        if args.data_usage == 'train':
            self.ids = self.read_ids_info(self.data_dir, 'train.txt')
        elif args.data_usage == 'test':
            self.ids = self.read_ids_info(self.data_dir, 'test.txt')
        else:
            self.ids = self.read_ids_info(self.data_dir, 'trans.txt')
            
        self.stride = stride if stride else self.img_rows
        if args.has_ref:
            self.ref_src_path = os.path.join(Utils_DIR, '../src', 
                                             'classes', args.data+'-class.csv')
            ref_dst_dir = os.path.join(Utils_DIR, '../dataset', 'classes')
            if not os.path.exists(ref_dst_dir):
                os.makedirs(ref_dst_dir)
            self.ref_dst_path = os.path.join(ref_dst_dir, args.data+'-class.csv')
            
    def extract_by_stride_slide(self):
        print("Processing via sliding window with stride")
        self.save_dir = os.path.join(Utils_DIR, '../dataset', args.data + "-str")

        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
        os.makedirs(self.save_dir)
        
        _statistics = []
        X_slices, y_slices = [], []
        ori_file, ori_size, ori_pos = [], [], []
        for img_id in self.ids:
            print("\t Image{}/{}: {}".format(self.ids.index(img_id)+1, len(self.ids), img_id))
            self.src_img = imread(self._srcpath % img_id)
            self.tar_img = imread(self._tarpath % img_id)
            assert self.src_img.shape[:2] == self.tar_img.shape[:2], \
                "Image dimension must be consistent."
                                                                
            # extract slices from source and target imagery
            rows, cols = self.src_img.shape[:2]
            row_range = range(0, rows - self.img_rows, self.stride)
            col_range = range(0, cols - self.img_cols, self.stride)
            nb_samples = len(row_range) * len(col_range)
            
            print("\t \t Original: img_rows : {}; img_cols : {}".format(rows, cols))
            print("\t \t Original: nb_rows : {}; nb_cols : {}".format(
                len(row_range), len(col_range)))
            print("\t \t nb_crop : {}".format(nb_samples))

            for i, j in itertools.product(row_range, col_range):
                img_src = self.src_img[i:i + self.img_rows,
                                       j:j + self.img_cols]
                img_tar = self.tar_img[i:i + self.img_rows,
                                       j:j + self.img_cols]
                X_slices.append(img_src)
                y_slices.append(img_tar)
                ori_file.append(img_id)
                ori_size.append("{},{}".format(
                    len(row_range), len(col_range)))
                ori_pos.append("{},{}".format(i, j))
            
            if args.data_usage == 'train' or args.data_usage == 'trans':
                _statistic = [img_id, nb_samples, len(row_range), len(col_range),
                              self.img_rows, self.img_cols, self.split]
            else:
                _statistic = [img_id, nb_samples, len(row_range), len(col_range),
                              self.img_rows, self.img_cols, 'no split']
            _statistics.append(_statistic)
                       
            self.save_slices(X_slices, "image")
            self.save_slices(y_slices, "label")
            
        _file = os.path.join(self.save_dir, 'statistic.csv')
        pd.DataFrame(_statistics,
                     columns=["img_name","nb-samples", "nb_rows", "nb_cols", \
                              "img_rows", "img_cols", "split"]).to_csv(_file, index=False)        
            
        # save infos
        infos = pd.DataFrame(columns=['img_name' ,'id', 'ori_size', 'ori_pos'])
        infos['id'] = ['img_{}.tif'.format(i) for i in range(len(y_slices))]
        infos['img_name'] = ori_file
        infos['ori_size'] = ori_size
        infos['ori_pos'] = ori_pos
        self.save_infos(infos)
        if args.has_ref:
            shutil.copyfile(self.ref_src_path, self.ref_dst_path)
        
    def extract_by_random_slide(self):
        print("Processing via randomly sliding window")
        self.save_dir = os.path.join(Utils_DIR, '../dataset', args.data + "-rand")
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
        os.makedirs(self.save_dir)
        
        _statistics = []
        X_slices, y_slices = [], []
        ori_file, ori_pos = [], []
        for img_id in self.ids:
            print("\t Image{}/{}: {}".format(self.ids.index(img_id)+1, len(self.ids), img_id))
            self.src_img = imread(self._srcpath % img_id)
            self.tar_img = imread(self._tarpath % img_id)
            assert self.src_img.shape[:2] == \
                self.tar_img.shape[:2], "Image dimension must be consistent."
            # extract slices from source and target imagery
            rows, cols = self.src_img.shape[:2]
            
            print("\t \t img_rows : {}; img_cols : {}".format(rows, cols))
            print("\t \t nb_crop : {}".format(self.nb_crop))

            for _ in range(self.nb_crop):
                i = random.randint(0, rows - self.img_rows)
                j = random.randint(0, cols - self.img_cols)
                img_src = self.src_img[i:i + self.img_rows,
                                       j:j + self.img_cols]
                img_tar = self.tar_img[i:i + self.img_rows,
                                       j:j + self.img_cols]
                
                X_slices.append(img_src)
                y_slices.append(img_tar)
                
                ori_file.append(img_id)
                ori_pos.append("{},{}".format(i, j))
            if args.data_usage == 'train' or args.data_usage == 'trans':
                _statistic = [img_id, self.nb_crop, \
                              self.img_rows, self.img_cols, self.split]
            else:
                _statistic = [img_id, self.nb_crop, \
                              self.img_rows, self.img_cols, 'no split']
            _statistics.append(_statistic)
            
        _file = os.path.join(self.save_dir, 'statistic.csv')
        pd.DataFrame(_statistics,
                     columns=["img_name","nb-samples", \
                              "img_rows", "img_cols", "split"]).to_csv(_file, index=False)        

        # save infos
        infos = pd.DataFrame(columns=['img_name', 'id',  'ori_pos'])
        infos['id'] = ['img_{}.tif'.format(i) for i in range(len(y_slices))]
        infos['img_name'] = ori_file
        infos['ori_pos'] = ori_pos
        self.save_infos(infos)
        
        # save slices
        self.save_slices(X_slices, "image")
        self.save_slices(y_slices, "label")
        if args.has_ref:
            shutil.copyfile(self.ref_src_path, self.ref_dst_path)

if __name__ == "__main__":
    # ====================== parameter initialization ======================= #
    parser = argparse.ArgumentParser(description='ArgumentParser')
    parser.add_argument('-data', type=str, default="river_trian_landsat-aster_20210228",
                        help='data dir for processing')
    parser.add_argument('-data_usage', type=str, default="train", \
                        choices=['train', 'test', 'trans'],\
                        help='data usage for training, testing, or transfer learning?')
    parser.add_argument('-split', type=list, default=[0.8, 0.1, 0.1],
                        help='train, val, and test partition')
    parser.add_argument('-is_multi', type=bool, default=False,
                        help='where to use multi-bands extractor')
    parser.add_argument('-mode', type=str, default="slide-rand",
                        choices=['slide-stride', 'vector', 'slide-rand'],
                        help='croping mode ')
    parser.add_argument('-img_rows', type=int, default=224,
                        help='img rows for croping ')
    parser.add_argument('-img_cols', type=int, default=224,
                        help='img cols for croping ')
    parser.add_argument('-stride', type=int, default=224,
                        help='img cols for croping ')
    parser.add_argument('-nb_crop', type=int, default=400,
                        help='random crop number')
    parser.add_argument('-has_ref', type=lambda x: (str(x).lower() == 'true'), \
                        default=False, help='has class reference or not')
    parser.add_argument('-threshold', type=float, default=0,
                        help='hourse cover rate to eliminate')
    parser.add_argument('-edge_buffer', type=float, default=0.1,
                        help='buffer area from edge')
    args = parser.parse_args()
    if args.is_multi:
        extractor = multiExtractor(args)
        if args.mode == 'slide-stride':
            extractor.extract_by_stride_slide()
        else:
            extractor.extract_by_random_slide()
    else:
        extractor = singleExtractor(args, 
                                    args.stride, args.threshold, args.edge_buffer)
        if args.mode == 'slide-stride':
            extractor.extract_by_stride_slide()
        elif args.mode == 'slide-rand':
            extractor.extract_by_random_slide()
        else:
            extractor.extract_by_vector()
