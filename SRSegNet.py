#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
  @CreateTime:   2018-01-02T20:57:09+09:00
  @Email:  guangmingwu2010@gmail.com
  @Copyright: go-hiroaki
  @License: MIT
"""
import os
import argparse
import torch
import torch.optim as optim

from models import srsegnet
from utils.runner import Trainer_SRSEG, Tester_SRSEG, Transfer, load_checkpoint
from config import SRSeg_train, SRSeg_test, SRSeg_trans
import torch.nn as nn


class SRSegNet(object):
    """
        packed generator and discriminator in patchGAN
        input and output are paired for next step classification
    """

    def __init__(self, args):
        self.model = srsegnet.SRSegNet(args.in_ch, args.out_ch, args.base_kernel_sr,
                                   args.base_kernel_seg, args.upscale_factor)

def main(args):
    method = os.path.basename(__file__).split(".")[0]

    if args.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    if args.train:
        print('===> Loading datasets')
        # initialize datasets
        datasets_train = SRSeg_train(args)
        args.in_ch = datasets_train.in_ch
        args.out_ch = datasets_train.out_ch
        args.data_dir = datasets_train.data_dir

        # initialize network
        net = SRSegNet(args)
        if args.cuda:
            net.model.cuda()
        net.optimizer = optim.Adam(
            net.model.parameters(), lr=args.lr, betas=optim_betas)

        if args.is_multi:
            args.class_names = datasets_train.class_names

        # initialize runner
        run_train = Trainer_SRSEG(args, method)

        print("===> Start training {}...".format(method))
        run_train.training(net, [datasets_train.train, datasets_train.val])

        run_train.save_log()
        run_train.learning_curve()

        run_train.evaluating(net.model, datasets_train.train, 'train')
        run_train.evaluating(net.model, datasets_train.val, 'val')
        run_train.evaluating(net.model, datasets_train.test, "test")
        model_name = run_train.save_checkpoint(net.model)
        print('===> Complete training')

    if args.test:
        print("===> Start testing ...")
        if args.train:
            args.test_model_name = model_name
        args.model_name = args.test_model_name
        model = load_checkpoint(args.model_name)

        datasets_test = SRSeg_test(args)
        args.in_ch = datasets_test.in_ch
        args.out_ch = datasets_test.out_ch
        args.data_dir = datasets_test.data_dir
        if args.is_multi:
            args.class_names = datasets_test.class_names

        run_test = Tester_SRSEG(args, method)
        run_test.testing(model, datasets_test.all, 'all')
        print('===> Complete Testing')

    if args.trans:
        print("===> Start transfer learning ...")

        datasets_trans = SRSeg_trans(args)
        args.in_ch = datasets_trans.in_ch
        args.out_ch = datasets_trans.out_ch
        args.data_dir = datasets_trans.data_dir
        if args.is_multi:
            args.class_names = datasets_trans.class_names

        if args.train:
            args.trans_model_name = model_name
        args.model_name = args.trans_model_name
        model = load_checkpoint(args.model_name)

        if args.is_freeze:
            print('===> Frozen mode')
            for param in model.parameters():
                param.requires_grad = False
        else:
            print('===> Non-Frozen mode')

        num_ftrs = model.outconv1[0].in_channels
        model.outconv1[0] = nn.Conv2d(num_ftrs, args.out_ch, kernel_size=(1, 1), stride=(1, 1))

        if args.cuda:
            model.cuda()
        optimizer = optim.Adam(
            model.parameters(), lr=args.lr, betas=optim_betas)
        args.optimizer = optimizer

        run_trans = Transfer(args, method)
        run_trans.training(model, [datasets_trans.train, datasets_trans.val])

        run_trans.save_log_trans()
        run_trans.learning_curve_trans()

        run_trans.evaluating(model, datasets_trans.train, 'train')
        run_trans.evaluating(model, datasets_trans.val, 'val')
        run_trans.evaluating(model, datasets_trans.test, "test")
        print('===> Complete transfer learning')
        model_name = run_trans.save_checkpoint_trans(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ArgumentParser')
    parser.add_argument('-train', type=lambda x: (str(x).lower() == 'true'), \
                        default=True, help='train or not?')
    parser.add_argument('-train_data', type=str, default='setagaya-rand',
                        help='training data dir name')
    parser.add_argument('-is_multi', type=lambda x: (str(x).lower() == 'true'), default=False,
                        help='multi-class or not')
    parser.add_argument('-trigger', type=str, default='epoch', choices=['epoch', 'iter'],
                        help='trigger type for logging')
    parser.add_argument('-interval', type=int, default=2,
                        help='interval for logging')
    parser.add_argument('-terminal', type=int, default=200,
                        help='terminal for training ')
    parser.add_argument('-seg_para', type=float, default=1.0,
                        help='loss = 1 * sr_loss + seg_para * seg_loss')
    parser.add_argument('-save_best', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help='only save best val_loss model')
    parser.add_argument('-base_kernel_sr', type=int, default=24,
                        help='base number of kernels')
    parser.add_argument('-base_kernel_seg', type=int, default=24,
                        help='base number of kernels')
    parser.add_argument('-upscale_factor', type=int, default=2,
                        help='base number of kernels')
    parser.add_argument('-batch_size', type=int, default=32,
                        help='batch_size for training ')
    parser.add_argument('-lr', type=float, default=2e-4,
                        help='learning rate for discriminator')
    parser.add_argument('-cuda', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help='using cuda for optimization')

    # testing
    parser.add_argument('-test', type=lambda x: (str(x).lower() == 'true'), \
                        default=True, help='test or not?')
    parser.add_argument('-test_data', type=str, default='setagaya-str',
                        help='testing data dir name')
    parser.add_argument('-test_model_name', type=str, \
                        default='SRSegNet_epoch_4_Dec20_12_34.pth', help='model used for testing')
    parser.add_argument('-ground_truth', type=lambda x: (str(x).lower() == 'true'), \
                        default=True, help='have ground truth or not')

    # transfer learning
    parser.add_argument('-trans', type=lambda x: (str(x).lower() == 'true'), \
                        default=False, help='transfer learning or not?')
    parser.add_argument('-trans_data', type=str, default='map_trans-rand',
                        help='transfer learning data dir name')
    parser.add_argument('-trans_model_name', type=str, \
                        default='UNet_epoch_1000_Jun14_17_14.pth', help='model used for transfer')
    parser.add_argument('-is_freeze', type=lambda x: (str(x).lower() == 'true'), \
                        default=False, help='freeze or not')
    parser.add_argument('-trans_trigger', type=str, default='epoch', choices=['epoch', 'iter'],
                        help='trigger type for logging')
    parser.add_argument('-trans_interval', type=int, default=4,
                        help='interval for logging')
    parser.add_argument('-trans_terminal', type=int, default=200,
                        help='terminal for training ')

    args = parser.parse_args()
    optim_betas = (0.9, 0.999)
    main(args)

