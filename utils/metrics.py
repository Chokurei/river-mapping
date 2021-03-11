#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
  @Email:  guo@locationmind.com
  @Copyright: guo-zhiling
  @License: MIT
"""
import sys
sys.path.append('./utils')

import torch
import numpy as np
from datasets import *
from torch.utils.data import DataLoader
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import confusion_matrix

esp = 1e-5


def _binarize(y_data, threshold=0.5):
    """
    args:
        y_data : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
        threshold : [0.0, 1.0]
    return binarized y_data
    """
    y_data[y_data < threshold] = 0.0
    y_data[y_data >= threshold] = 1.0
    return y_data


def _get_tp(y_pred, y_true):
    """
    args:
        y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
        y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
    return true_positive
    """
    return torch.sum(y_true * y_pred)


def _get_fp(y_pred, y_true):
    """
    args:
        y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
        y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
    return false_positive
    """
    return torch.sum((1.0 - y_true) * y_pred)


def _get_tn(y_pred, y_true):
    """
    args:
        y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
        y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
    return true_negative
    """
    return torch.sum((1.0 - y_true) * (1.0 - y_pred))


def _get_fn(y_pred, y_true):
    """
    args:
        y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
        y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
    return false_negative
    """
    return torch.sum(y_true * (1.0 - y_pred))


def confusion_matrix_single(y_pred, y_true, threshold=0.5):
    """
    args:
        y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
        y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
        threshold : [0.0, 1.0]
    return confusion matrix
    """
    if threshold:
        y_pred = _binarize(y_pred, threshold)
    nb_tp = _get_tp(y_pred, y_true)
    nb_fp = _get_fp(y_pred, y_true)
    nb_tn = _get_tn(y_pred, y_true)
    nb_fn = _get_fn(y_pred, y_true)
    return [nb_tp, nb_fp, nb_tn, nb_fn]


def overall_accuracy(y_pred, y_true, threshold=0.5):
    """
    args:
        y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
        y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
        threshold : [0.0, 1.0]
    return (tp+tn)/total
    """
    if threshold:
        y_pred = _binarize(y_pred, threshold)
    nb_tp_tn = torch.sum(y_true == y_pred)
    return nb_tp_tn.cpu().numpy() / (np.prod(y_true.shape))


def precision(y_pred, y_true, threshold=0.5):
    """
    args:
        y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
        y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
        threshold : [0.0, 1.0]
    return tp/(tp+fp)
    """
    if threshold:
        y_pred = _binarize(y_pred, threshold)
    nb_tp = _get_tp(y_pred, y_true)
    nb_fp = _get_fp(y_pred, y_true)
    return (nb_tp / (nb_tp + nb_fp + esp)).cpu().numpy()


def recall(y_pred, y_true, threshold=0.5):
    """
    args:
        y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
        y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
        threshold : [0.0, 1.0]
    return tp/(tp+fn)
    """
    if threshold:
        y_pred = _binarize(y_pred, threshold)
    nb_tp = _get_tp(y_pred, y_true)
    nb_fn = _get_fn(y_pred, y_true)
    return (nb_tp / (nb_tp + nb_fn + esp)).cpu().numpy()


def f1_score(y_pred, y_true, threshold=0.5):
    """
    args:
        y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
        y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
        threshold : [0.0, 1.0]
    return 2*precision*recall/(precision+recall)
    """
    if threshold:
        y_pred = _binarize(y_pred, threshold)
    nb_tp = _get_tp(y_pred, y_true)
    nb_fp = _get_fp(y_pred, y_true)
    nb_fn = _get_fn(y_pred, y_true)
    _precision = nb_tp / (nb_tp + nb_fp + esp)
    _recall = nb_tp / (nb_tp + nb_fn + esp)
    return (2 * _precision * _recall / (_precision + _recall + esp)).cpu().numpy()


def kappa(y_pred, y_true, threshold=0.5):
    """
    args:
        y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
        y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
        threshold : [0.0, 1.0]
    return (Po-Pe)/(1-Pe)
    """
    if threshold:
        y_pred = _binarize(y_pred, threshold)
    nb_tp = _get_tp(y_pred, y_true)
    nb_fp = _get_fp(y_pred, y_true)
    nb_tn = _get_tn(y_pred, y_true)
    nb_fn = _get_fn(y_pred, y_true)
    nb_total = nb_tp + nb_fp + nb_tn + nb_fn
    Po = (nb_tp + nb_tn) / nb_total
    Pe = ((nb_tp + nb_fp) * (nb_tp + nb_fn) +
          (nb_fn + nb_tn) * (nb_fp + nb_tn)) / (nb_total**2)
    return ((Po - Pe) / (1 - Pe + esp)).cpu().numpy()


def jaccard(y_pred, y_true, threshold=0.5):
    """
    args:
        y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
        y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
        threshold : [0.0, 1.0]
    return intersection / (sum-intersection)
    """
    if threshold:
        y_pred = _binarize(y_pred, threshold)
    _intersection = torch.sum(y_true * y_pred)
    _sum = torch.sum(y_true + y_pred)
    return (_intersection / (_sum - _intersection + esp)).cpu().numpy()


def img_to_pos_vec(img):
    """
    args:
        img : 2-d ndarray in [img_rows, img_cols], values in [0,1].
    return position vector(where pixel value==1)
    """
    idx = np.where(img == 1)
    pos_vec = [[x, y] for x, y in zip(idx[0], idx[1])]
    return pos_vec


def hausdorff(y_pred, y_true, threshold=0.5):
    """
    args:
        y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
        y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
        threshold : [0.0, 1.0]
    return hausdorff_distance
    """
    if threshold:
        y_pred = _binarize(y_pred, threshold)
    hs = 0
    for i in range(y_pred.shape[0]):
        img_pred = y_pred[i, 0, :, :]
        img_true = y_true[i, 0, :, :]
        pos_vec_pred = img_to_pos_vec(img_pred)
        pos_vec_true = img_to_pos_vec(img_true)
        hs += max(directed_hausdorff(pos_vec_pred, pos_vec_true)[0],
                  directed_hausdorff(pos_vec_true, pos_vec_pred)[0])
    return hs / y_pred.shape[0]


class ConfusionMatrix:
    def __init__(self, nclasses, classes, useUnlabeled=True):
        self.mat = np.zeros((nclasses, nclasses), dtype=np.float)
        self.precisions = np.zeros((nclasses), dtype=np.float)
        self.recalls = np.zeros((nclasses), dtype=np.float)
        self.f1scores = np.zeros((nclasses), dtype=np.float)      
       
        self.IoU = np.zeros((nclasses), dtype=np.float)
        self.mIoU = 0

        self.nclasses = nclasses
        self.classes = classes
        self.list_classes = list(range(nclasses))
        self.useUnlabeled = useUnlabeled
        self.matStartIdx = 1 if not self.useUnlabeled else 0

    def update_matrix(self, target, prediction):
        if not(isinstance(prediction, np.ndarray)) or not(isinstance(target, np.ndarray)):
            print("Expecting ndarray")
        elif len(target.shape) == 3:          # batched spatial target
            if len(prediction.shape) == 4:  # prediction is 1 hot encoded
                temp_prediction = np.argmax(prediction, axis=1).flatten()
            elif len(prediction.shape) == 3:
                temp_prediction = prediction.flatten()
            else:
                print("Make sure prediction and target dimension is correct")

            temp_target = target.flatten()
        elif len(target.shape) == 2:        # spatial target
            if len(prediction.shape) == 3:  # prediction is 1 hot encoded
                temp_prediction = np.argmax(prediction, axis=1).flatten()
            elif len(prediction.shape) == 2:
                temp_prediction = prediction.flatten()
            else:
                print("Make sure prediction and target dimension is correct")

            temp_target = target.flatten()
        elif len(target.shape) == 1:
            if len(prediction.shape) == 2:  # prediction is 1 hot encoded
                temp_prediction = np.argmax(prediction, axis=1).flatten()
            elif len(prediction.shape) == 1:
                temp_prediction = prediction
            else:
                print("Make sure prediction and target dimension is correct")

            temp_target = target
        else:
            print("Data with this dimension cannot be handled")

        self.mat += confusion_matrix(temp_target, temp_prediction, labels=self.list_classes)

    def scores(self):
        tp = 0
        fp = 0
#        tn = 0
        fn = 0
        total = 0   # Total true positives
        N = 0       # Total samples
        for i in range(self.matStartIdx, self.nclasses):
            N += sum(self.mat[:, i])
            tp = self.mat[i][i]
            fp = sum(self.mat[self.matStartIdx:, i]) - tp
            fn = sum(self.mat[i,self.matStartIdx:]) - tp

            if (tp+fp) == 0:
                self.precisions[i] = 0
            else:
                self.precisions[i] = tp/(tp + fp)
                
            if (tp+fn) == 0:
                self.recalls[i] = 0
            else:
                self.recalls[i] = tp/(tp + fn)
                
            if (self.recalls[i] + self.precisions[i]) == 0:
                self.f1scores[i] = 0
            else:
                self.f1scores[i] = 2 * self.recalls[i] * self.precisions[i] / \
                    (self.recalls[i] + self.precisions[i])

            if (tp+fp+fn) == 0:
                self.IoU[i] = 0
            else:
                self.IoU[i] = tp/(tp + fp + fn)

            total += tp

        self.mIoU = sum(self.IoU[self.matStartIdx:])/(self.nclasses - self.matStartIdx)
        self.accuracy = total/(sum(sum(self.mat[self.matStartIdx:, self.matStartIdx:])))
        
        self.mprecision = sum(self.precisions[self.matStartIdx:])/(self.nclasses - self.matStartIdx)
        self.mrecall = sum(self.recalls[self.matStartIdx:])/(self.nclasses - self.matStartIdx)
        self.mf1score = sum(self.f1scores[self.matStartIdx:])/(self.nclasses - self.matStartIdx)
        
        return self.precisions, self.recalls, \
            self.f1scores, self.mat, self.IoU, self.mIoU, self.accuracy, self.mprecision, self.mrecall, self.mf1score

    def plot_confusion_matrix(self, filename):
        # Plot generated confusion matrix
        print(filename)


    def reset(self):
        self.mat = np.zeros((self.nclasses, self.nclasses), dtype=float)   

        self.precisions = np.zeros((self.nclasses), dtype=np.float)
        self.recalls = np.zeros((self.nclasses), dtype=np.float)
        self.f1scores = np.zeros((self.nclasses), dtype=np.float)      
        self.IoU = np.zeros((self.nclasses), dtype=np.float)
        self.mIoU = 0

if __name__ == "__main__":
    img_rows, img_cols, batch_size = 224, 224, 32
    dataset = nzLS(split="all")
    
    data_loader = DataLoader(dataset, batch_size, num_workers=4,
                             shuffle=False)
    batch_iterator = iter(data_loader)
    x, y_true = next(batch_iterator)
    # add one row of noice in the middle
    y_pred = np.copy(y_true)
    y_pred[:, :, img_rows // 2, :] = 1
    y_true = torch.FloatTensor(y_true)
    y_pred = torch.FloatTensor(y_pred)

    matrix = confusion_matrix_single(y_pred, y_true)
    print('confusion:', matrix)

    hs = hausdorff(y_pred, y_true)
    print('hausdorff:', hs)
