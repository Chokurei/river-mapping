#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
  @Email:  guo@locationmind.com
  @Copyright: guo-zhiling
  @License: MIT
"""
import os
import sys
sys.path.append('./utils')
import time
import torch
import metrics
from metrics import ConfusionMatrix
import pandas as pd
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import copy


Utils_DIR = os.path.dirname(os.path.abspath(__file__))
Logs_DIR = os.path.join(Utils_DIR, '../logs')
Checkpoint_DIR = os.path.join(Utils_DIR, '../checkpoint')

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


def load_checkpoint(name):
    assert os.path.exists("{}/{}".format(Checkpoint_DIR, name)
                          ), "{} not exists.".format(name)
    print("Loading checkpoint: {}".format(name))
    return torch.load("{}/{}".format(Checkpoint_DIR, name))


class Base(object):
    def __init__(self, args, method):
        self.args = args
        self.method = method
        if self.method.startswith('SR'):
            self.method += '_' + str(args.seg_para)
        self.is_multi = args.is_multi
        self.date = time.strftime("%h%d_%H_%M")
        self.epoch = 0
        self.iter = 0
        if args.save_best:
            self.best_acc = 0
        self.logs = []
        self.headers = ["epoch", "iter", "train_loss", "train_acc", "train_time(sec)", \
                        "train_fps", "val_loss", "val_acc", "val_time(sec)", "val_fps"]
        
    def logging(self, verbose=True):
        if self.method.startswith('SR'):
            self._train_log = self.train_log.copy()
            self._train_log.pop(1)
            self._train_log.pop(1)
            self._val_log = self.val_log.copy()
            self._val_log.pop(1)
            self._val_log.pop(1)
            self.logs.append([self.epoch, self.iter] +
                             self._train_log + self._val_log)
        else:
            self.logs.append([self.epoch, self.iter] +
                             self.train_log + self.val_log)
        if verbose and self.method.startswith('SR'):
            print("Epoch:{:02d}, Iter:{:05d}, "
                  "train_loss:{:0.3f}, train_loss_sr:{:0.3f}, train_loss_seg:{:0.3f}, train_acc:{:0.3f}, "
                "val_loss:{:0.3f}, val_loss_sr:{:0.3f}, val_loss_seg:{:0.3f}, val_acc:{:0.3f}"
                  .format(self.epoch, self.iter,
                          self.train_log[0], self.train_log[1], self.train_log[2], self.train_log[3],
                          self.val_log[0], self.val_log[1], self.val_log[2], self.val_log[3]))
        else:
            print("Epoch:{:02d}, Iter:{:05d}, "
            "train_loss:{:0.3f}, train_acc:{:0.3f}, "
            "val_loss:{:0.3f}, val_acc:{:0.3f}"
                  .format(self.epoch, self.iter,
                          self.train_log[0], self.train_log[1],
                          self.val_log[0], self.val_log[1]))

    def save_log(self):
        if not os.path.exists(os.path.join(Logs_DIR, 'raw')):
            os.makedirs(os.path.join(Logs_DIR, 'raw'))

        self.logs = pd.DataFrame(self.logs,
                                 columns=self.headers)    
        self.logs.to_csv("{}/raw/{}_{}_{}_{}.csv".format(Logs_DIR, self.method, self.args.trigger,
                             self.args.terminal, self.date), index=False, float_format='%.3f')        

    def save_checkpoint(self, model, name=None):
        if self.args.cuda:
            model.cpu()
        if name:
            model_name = "{}_{}_{}_{}_{}.pth".format(
                self.method, name, self.args.trigger, self.args.terminal, self.date)
        else:
            model_name = "{}_{}_{}_{}.pth".format(
                self.method, self.args.trigger, self.args.terminal, self.date)
        if not os.path.exists(Checkpoint_DIR):
            os.mkdir(Checkpoint_DIR)
        torch.save(model, os.path.join(Checkpoint_DIR, model_name))
        print("Saving checkpoint: {}".format(model_name))
        return model_name

    def learning_curve(self, labels=["train_loss", "train_acc", "val_loss", "val_acc"]):
        if not os.path.exists(os.path.join(Logs_DIR, "curve")):
            os.mkdir(os.path.join(Logs_DIR, "curve"))
        # set style
        sns.set_context("paper", font_scale=1.5,)
        sns.set_style("ticks", {
            "font.family": "Times New Roman",
            "font.serif": ["Times", "Palatino", "serif"]})

        for _label in labels:
            plt.plot(self.logs[self.args.trigger],
                     self.logs[_label], label=_label)
        plt.ylabel("Loss / Overall Accuracy")
        if self.args.trigger == 'epoch':
            plt.xlabel("Epochs")
        else:
            plt.xlabel("Iterations")
        plt.suptitle("Training log of {}".format(self.method))

        plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

        plt.savefig(os.path.join(Logs_DIR, 'curve', '{}_{}_{}_{}.png'.format(self.method, self.args.trigger, self.args.terminal, self.date)),
                    format='png', bbox_inches='tight', dpi=1200)
        # delete plt cache
        plt.clf()
        
        return 0

    def save_log_trans(self):
        if not os.path.exists(os.path.join(Logs_DIR, 'raw')):
            os.makedirs(os.path.join(Logs_DIR, 'raw'))

        self.logs = pd.DataFrame(self.logs,
                                 columns=self.headers)
        
        self.logs.to_csv("{}/raw/{}_trans_{}_{}_{}.csv".format(Logs_DIR, self.model_basename, self.args.trans_trigger,
                             self.args.trans_terminal, self.date), index=False, float_format='%.3f')

    def save_checkpoint_trans(self, model, name=None):
        if self.args.cuda:
            model.cpu()
        model_name = "{}_trans_{}_{}_{}.pth".format(
                self.model_basename, self.args.trans_trigger, self.args.trans_terminal, self.date)
        if not os.path.exists(Checkpoint_DIR):
            os.mkdir(Checkpoint_DIR)
        torch.save(model, os.path.join(Checkpoint_DIR, model_name))
        print("Saving checkpoint: {}".format(model_name))
        return model_name

    
    def learning_curve_trans(self, labels=["train_loss", "train_acc", "val_loss", "val_acc"]):
        if not os.path.exists(os.path.join(Logs_DIR, "curve")):
            os.mkdir(os.path.join(Logs_DIR, "curve"))
        # set style
        sns.set_context("paper", font_scale=1.5,)
        sns.set_style("ticks", {
            "font.family": "Times New Roman",
            "font.serif": ["Times", "Palatino", "serif"]})

        for _label in labels:
            plt.plot(self.logs[self.args.trans_trigger],
                     self.logs[_label], label=_label)
        plt.ylabel("BCE-Loss / Overall Accuracy")
        if self.args.trans_trigger == 'epoch':
            plt.xlabel("Epochs")
        else:
            plt.xlabel("Iterations")
        plt.suptitle("Training log of {}".format(self.method))

        plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
        
        plt.savefig(os.path.join(Logs_DIR, 'curve', '{}_trans_{}_{}_{}.png'.format(self.model_basename, self.args.trans_trigger, self.args.trans_terminal, self.date)),
                    format='png', bbox_inches='tight', dpi=1200)
        
        plt.clf()
        return 0


class Trainer(Base):
    def training(self, net, datasets):
        """
          input:
            net: (object) model & optimizer
            datasets : (list) [train, val] dataset object
        """
        args = self.args
        net.model.train()
        steps = len(datasets[0]) // args.batch_size
        if args.trigger == 'epoch':
            args.epochs = args.terminal
            args.iters = steps * args.terminal
            args.iter_interval = steps * args.interval
        else:
            args.epochs = args.terminal // steps + 1
            args.iters = args.terminal
            args.iter_interval = args.interval

        train_loss, train_acc = 0, 0
        start = time.time()            
        
        for epoch in range(1, args.epochs + 1):
            self.epoch = epoch
            # setup data loader
            data_loader = DataLoader(datasets[0], args.batch_size, num_workers=4,
                                     shuffle=True)
            batch_iterator = iter(data_loader)
            for step in range(steps):
                self.iter += 1
                if self.iter > args.iters:
                    self.iter -= 1
                    break
                # convert numpy.ndarray into pytorch tensor
                x, y = next(batch_iterator)
                x = torch.tensor(x)
                y = torch.tensor(y)
                if args.cuda:
                    x = x.cuda()
                    y = y.cuda()
                # training
                gen_y = net.model(x)
#                if self.is_multi:
#                    gen_y = gen_y[0]
                loss = F.binary_cross_entropy(gen_y, y)
                # Update generator parameters
                net.optimizer.zero_grad()
                loss.backward()
                net.optimizer.step()
                train_loss += loss.item()
                # train_loss += loss.data[0].cpu().numpy()
#                train_acc += metrics.overall_accuracy(gen_y.data, y.data)
                if not self.is_multi:
                    train_acc += metrics.jaccard(gen_y.data, y.data)
                else:
                    train_acc += metrics.overall_accuracy(gen_y.data, y.data)
                
                # logging
                if self.iter % args.iter_interval == 0:
                    _time = time.time() - start
                    nb_samples = args.iter_interval * args.batch_size
                    train_log = [train_loss / args.iter_interval, train_acc /
                                 args.iter_interval, _time, nb_samples / _time]
                    self.train_log = [round(x, 3) for x in train_log]
                    self.validating(net.model, datasets[1])
                    self.logging(verbose=True)
                    # reinitialize
                    start = time.time()
                    train_loss, train_acc = 0, 0
        
        if args.save_best:
            net.model = self.best_model

    def validating(self, model, dataset):
        """
          input:
            model: (object) pytorch model
            batch_size: (int)
            dataset : (object) dataset
          return [val_acc, val_loss]
        """
        args = self.args
        if args.cuda:
            model.cuda()
        val_loss, val_acc = 0, 0
        data_loader = DataLoader(dataset, args.batch_size, num_workers=4,
                                 shuffle=False)
        batch_iterator = iter(data_loader)
        steps = len(dataset) // args.batch_size
        model.eval()
        start = time.time()
        if self.is_multi:
            metrics_mul = ConfusionMatrix(args.out_ch, args.class_names, useUnlabeled=True)
        for step in range(steps):
            x, y = next(batch_iterator)
            x = torch.tensor(x)
            y = torch.tensor(y)
            if args.cuda:
                x = x.cuda()
                y = y.cuda()

            # calculate pixel accuracy of generator
            gen_y = model(x)
#            if self.is_multi:
#                gen_y = gen_y[0]
            val_loss += F.binary_cross_entropy(gen_y, y).item()
#             val_loss += F.binary_cross_entropy(gen_y, y).data[0].cpu().numpy()
#            val_acc += metrics.overall_accuracy(gen_y.data, y.data)
            if not self.is_multi:
                val_acc += metrics.jaccard(gen_y.data, y.data)
            else:
                # Note: need optimize into mIoU
                pred = y.cpu().detach().numpy()
                gt = gen_y.cpu().detach().numpy()
                gt = np.argmax(gt, axis=1)
                metrics_mul.update_matrix(gt, pred)
        if self.is_multi:
            precisions, recalls, f1scores, mat, IoU, mIoU, \
                            accuracy, mprecision, mrecall, mf1score = metrics_mul.scores()
            metrics_mul.reset()
            val_acc = mIoU

        _time = time.time() - start
        nb_samples = steps * args.batch_size
        if not self.is_multi:
            val_log = [val_loss / steps, val_acc /
                       steps, _time, nb_samples / _time]
        else:
            val_log = [val_loss / steps, val_acc, 
                       _time, nb_samples / _time]
            
        self.val_log = [round(x, 3) for x in val_log]
        
        if args.save_best:
            self.best_checkpoint(model, val_log)
        
    def best_checkpoint(self, model, val_log):
        """
        log: val_log
        """
        val_acc = val_log[1]
        if val_acc >= self.best_acc:
            self.best_acc = val_acc
            self.best_model = copy.deepcopy(model)
            self.best_val_log = val_log
            self.best_train_log = self.train_log
        

    def evaluating(self, model, dataset, split):
        """
          input:
            model: (object) pytorch model
            dataset: (object) dataset
            split: (str) split of dataset in ['train', 'val', 'test']
          return [overall_accuracy, precision, recall, f1-score, jaccard, kappa]
        """
        args = self.args
        if not self.is_multi:
            oa, precision, recall, f1, jac, kappa = 0, 0, 0, 0, 0, 0
            if args.cuda:
                model.cuda()
            model.eval()
            data_loader = DataLoader(dataset, args.batch_size, num_workers=4,
                                     shuffle=False)
            batch_iterator = iter(data_loader)
            steps = len(dataset) // args.batch_size
    
            start = time.time()
            for step in range(steps):
                x, y = next(batch_iterator)
                x = torch.tensor(x)
                y = torch.tensor(y)
                if args.cuda:
                    x = x.cuda()
                    y = y.cuda()
                # calculate pixel accuracy of generator
                gen_y = model(x)
    #            print(gen_y.shape)
                oa += metrics.overall_accuracy(gen_y.data, y.data)
                precision += metrics.precision(gen_y.data, y.data)
                recall += metrics.recall(gen_y.data, y.data)
                f1 += metrics.f1_score(gen_y.data, y.data)
                jac += metrics.jaccard(gen_y.data, y.data)
                kappa += metrics.kappa(gen_y.data, y.data)
    
            _time = time.time() - start
    
            if not os.path.exists(os.path.join(Logs_DIR, 'statistic')):
                os.makedirs(os.path.join(Logs_DIR, 'statistic'))
    
            # recording performance of the model
            nb_samples = steps * args.batch_size
            basic_info = [self.date, self.method, args.data_dir,
                          self.epoch, self.iter, nb_samples, _time]
            basic_info_names = ['date', 'method', 'dataset_train','epochs',
                                'iters', 'nb_samples', 'time(sec)']
    
            perform = [round(idx / steps, 3)
                       for idx in [oa, precision, recall, f1, jac, kappa]]
            perform_names = ["overall_accuracy", "precision",
                             "recall", "f1-score", "jaccard", "kappa"]
            cur_log = pd.DataFrame([basic_info + perform],
                                   columns=basic_info_names + perform_names)
            # save performance
            if os.path.exists(os.path.join(Logs_DIR, 'statistic', "{}.csv".format(split))):
                logs = pd.read_csv(os.path.join(
                    Logs_DIR, 'statistic', "{}.csv".format(split)))
            else:
                logs = pd.DataFrame([])
            logs = logs.append(cur_log, ignore_index=True)
            logs.to_csv(os.path.join(Logs_DIR, 'statistic',
                                     "{}.csv".format(split)), index=False, float_format='%.3f')
        else:
            metrics_mul = ConfusionMatrix(args.out_ch, args.class_names, useUnlabeled=True)
            if args.cuda:
                model.cuda()
            model.eval()
            data_loader = DataLoader(dataset, args.batch_size, num_workers=4,
                                     shuffle=False)
            batch_iterator = iter(data_loader)
            steps = len(dataset) // args.batch_size
    
            start = time.time()
            for step in range(steps):
                x, y = next(batch_iterator)
                x = torch.tensor(x)
                y = torch.tensor(y)
                if args.cuda:
                    x = x.cuda()
                    y = y.cuda()
                # calculate pixel accuracy of generator
                gen_y = model(x)
                # calculate mIoU

                pred = y.cpu().detach().numpy()
                gt = gen_y.cpu().detach().numpy()
                gt = np.argmax(gt, axis=1)
                metrics_mul.update_matrix(gt, pred)
            precisions, recalls, f1scores, mat, IoU, mIoU, \
                            accuracy, mprecision, mrecall, mf1score = metrics_mul.scores()
            metrics_mul.reset()
            _time = time.time() - start 
            
            if not os.path.exists(os.path.join(Logs_DIR, 'statistic')):
                os.makedirs(os.path.join(Logs_DIR, 'statistic'))
    
            # recording performance of the model
            nb_samples = steps * args.batch_size
            basic_info = [self.date, self.method, args.data_dir,
                          self.epoch, self.iter, nb_samples, _time]
            basic_info_names = ['date', 'method', 'dataset_train','epochs',
                                'iters', 'nb_samples', 'time(sec)']
                            
            np.set_printoptions(threshold=200)
            perform = [np.round(idx , 3)\
                       for idx in [precisions, recalls, f1scores, mat, IoU, mIoU, \
                            accuracy, mprecision, mrecall, mf1score ]]
            perform_names = ['precisions', 'recalls', 'f1scores', 'mat', 'IoU', 'mIoU', \
                             'over_accuracy', 'mprecision', 'mrecall', 'mf1score']
            cur_log = pd.DataFrame([basic_info + perform],
                                   columns=basic_info_names + perform_names)
            # save performance
            if os.path.exists(os.path.join(Logs_DIR, 'statistic', "{}_mul.csv".format(split))):
                logs = pd.read_csv(os.path.join(
                    Logs_DIR, 'statistic', "{}_mul.csv".format(split)))
            else:
                logs = pd.DataFrame([])
            logs = logs.append(cur_log, ignore_index=True)
            pd.set_option('display.width', None)
            logs.to_csv(os.path.join(Logs_DIR, 'statistic',
                                     "{}_mul.csv".format(split)), index=False, float_format='%.3f')


class Tester(Base):
    def testing(self, model, dataset, split):
        """
          input:
            model: (object) pytorch model
            dataset: (object) dataset
            split: (str) split of dataset in ['train', 'val', 'test']
          return [overall_accuracy, precision, recall, f1-score, jaccard, kappa]
        """
        args = self.args
        self.model_basename = args.model_name.strip('.pth')
        if args.cuda:
                model.cuda()
        model.eval()
        data_loader = DataLoader(dataset, args.batch_size, num_workers=4,
                                     shuffle=False)
        batch_iterator = iter(data_loader)
        steps = len(dataset) // args.batch_size
        
        if not self.is_multi:
            start = time.time()
            oa, precision, recall, f1, jac, kappa = 0, 0, 0, 0, 0, 0
            for step in range(steps):
                x, y = next(batch_iterator)
                x = torch.tensor(x)
                y = torch.tensor(y)
                if args.cuda:
                    x = x.cuda()
                    y = y.cuda()
                # calculate pixel accuracy of generator
                gen_y = model(x)
                oa += metrics.overall_accuracy(gen_y.data, y.data)
                precision += metrics.precision(gen_y.data, y.data)
                recall += metrics.recall(gen_y.data, y.data)
                f1 += metrics.f1_score(gen_y.data, y.data)
                jac += metrics.jaccard(gen_y.data, y.data)
                kappa += metrics.kappa(gen_y.data, y.data)
            _time = time.time() - start
    
            if not os.path.exists(os.path.join(Logs_DIR, 'statistic')):
                os.makedirs(os.path.join(Logs_DIR, 'statistic'))
    
            # recording performance of the model
            nb_samples = steps * args.batch_size
            basic_info = [self.date, args.model_name, args.data_dir,
                          nb_samples, _time]
            basic_info_names = ['date', 'method', 'dataset_test',
                                'nb_samples', 'time(sec)']
    
            perform = [round(idx / steps, 3)
                       for idx in [oa, precision, recall, f1, jac, kappa]]
            perform_names = ["overall_accuracy", "precision",
                             "recall", "f1-score", "jaccard", "kappa"]
            cur_log = pd.DataFrame([basic_info + perform],
                                   columns=basic_info_names + perform_names)
            # save performance
            if os.path.exists(os.path.join(Logs_DIR, 'statistic', "{}.csv".format(split))):
                logs = pd.read_csv(os.path.join(
                    Logs_DIR, 'statistic', "{}.csv".format(split)))
            else:
                logs = pd.DataFrame([])
            logs = logs.append(cur_log, ignore_index=True)
            logs.to_csv(os.path.join(Logs_DIR, 'statistic',
                                     "{}.csv".format(split)), index=False, float_format='%.3f')

        else:
            start = time.time()
            metrics_mul = ConfusionMatrix(args.out_ch, args.class_names, useUnlabeled=True)
            for step in range(steps):
                x, y = next(batch_iterator)
                x = torch.tensor(x)
                y = torch.tensor(y)
                if args.cuda:
                    x = x.cuda()
                    y = y.cuda()
                # calculate pixel accuracy of generator
                gen_y = model(x)
                # calculate mIoU
                pred = y.cpu().detach().numpy()
                gt = gen_y.cpu().detach().numpy()
                gt = np.argmax(gt, axis=1)
                metrics_mul.update_matrix(gt, pred)
            precisions, recalls, f1scores, mat, IoU, mIoU, \
                            accuracy, mprecision, mrecall, mf1score = metrics_mul.scores()
            metrics_mul.reset()
            _time = time.time() - start 
            
            if not os.path.exists(os.path.join(Logs_DIR, 'statistic')):
                os.makedirs(os.path.join(Logs_DIR, 'statistic'))
    
            nb_samples = steps * args.batch_size
            # recording performance of the model
            basic_info = [self.date, args.model_name, args.data_dir,
                          nb_samples, _time]
            basic_info_names = ['date', 'method', 'dataset_test',
                                'nb_samples', 'time(sec)']
                            
            np.set_printoptions(threshold=200)
            perform = [np.round(idx , 3)\
                       for idx in [precisions, recalls, f1scores, mat, IoU, mIoU, \
                            accuracy, mprecision, mrecall, mf1score ]]
            perform_names = ['precisions', 'recalls', 'f1scores', 'mat', 'IoU', 'mIoU', \
                             'over_accuracy', 'mprecision', 'mrecall', 'mf1score']
            cur_log = pd.DataFrame([basic_info + perform],
                                   columns=basic_info_names + perform_names)
            # save performance
            if os.path.exists(os.path.join(Logs_DIR, 'statistic', "{}_mul.csv".format(split))):
                logs = pd.read_csv(os.path.join(
                    Logs_DIR, 'statistic', "{}_mul.csv".format(split)))
            else:
                logs = pd.DataFrame([])
            logs = logs.append(cur_log, ignore_index=True)
            logs.to_csv(os.path.join(Logs_DIR, 'statistic',
                                     "{}_mul.csv".format(split)), index=False, float_format='%.3f')

class Transfer(Base):
    def training(self, model, datasets):
        """
          input:
            net: (object) model & optimizer
            datasets : (list) [train, val] dataset object
        """
        args = self.args
        self.model_basename = args.model_name.strip('.pth')
        model.train()
        if args.cuda:
            model.cuda()
        steps = len(datasets[0]) // args.batch_size
        if args.trans_trigger == 'epoch':
            args.epochs = args.trans_terminal
            args.iters = steps * args.trans_terminal
            args.iter_interval = steps * args.trans_interval
        else:
            args.epochs = args.trans_terminal // steps + 1
            args.iters = args.trans_terminal
            args.iter_interval = args.trans_interval

        train_loss, train_acc = 0, 0
        start = time.time()
        for epoch in range(1, args.epochs + 1):
            self.epoch = epoch
            # setup data loader
            data_loader = DataLoader(datasets[0], args.batch_size, num_workers=4,
                                     shuffle=True)
            batch_iterator = iter(data_loader)
            for step in range(steps):
                self.iter += 1
                if self.iter > args.iters:
                    self.iter -= 1
                    break
                # convert numpy.ndarray into pytorch tensor
                x, y = next(batch_iterator)
                x = torch.tensor(x)
                y = torch.tensor(y)
                if args.cuda:
                    x = x.cuda()
                    y = y.cuda()
                # training
                gen_y = model(x)
                loss = F.binary_cross_entropy(gen_y, y)
                # Update generator parameters
                args.optimizer.zero_grad()
                loss.backward()
                args.optimizer.step()
                train_loss += loss.item()
                # train_loss += loss.data[0].cpu().numpy()
                train_acc += metrics.overall_accuracy(gen_y.data, y.data)
                # logging
                if self.iter % args.iter_interval == 0:
                    _time = time.time() - start
                    nb_samples = args.iter_interval * args.batch_size
                    train_log = [train_loss / args.iter_interval, train_acc /
                                 args.iter_interval, _time, nb_samples / _time]
                    self.train_log = [round(x, 3) for x in train_log]
                    self.validating(model, datasets[1])
                    self.logging(verbose=True)
                    # reinitialize
                    start = time.time()
                    train_loss, train_acc = 0, 0

    def validating(self, model, dataset):
        """
          input:
            model: (object) pytorch model
            batch_size: (int)
            dataset : (object) dataset
          return [val_acc, val_loss]
        """
        args = self.args
        if args.cuda:
            model.cuda()
        val_loss, val_acc = 0, 0
        data_loader = DataLoader(dataset, args.batch_size, num_workers=4,
                                 shuffle=False)
        batch_iterator = iter(data_loader)
        steps = len(dataset) // args.batch_size
        model.eval()
        start = time.time()
        for step in range(steps):
            x, y = next(batch_iterator)
            x = torch.tensor(x)
            y = torch.tensor(y)
            if args.cuda:
                x = x.cuda()
                y = y.cuda()

            # calculate pixel accuracy of generator
            gen_y = model(x)
#            if self.is_multi:
#                gen_y = gen_y[0]
            val_loss += F.binary_cross_entropy(gen_y, y).item()
#             val_loss += F.binary_cross_entropy(gen_y, y).data[0].cpu().numpy()
            val_acc += metrics.overall_accuracy(gen_y.data, y.data)

        _time = time.time() - start
        nb_samples = steps * args.batch_size
        val_log = [val_loss / steps, val_acc /
                   steps, _time, nb_samples / _time]
        self.val_log = [round(x, 3) for x in val_log]

    def evaluating(self, model, dataset, split):
        """
          input:
            model: (object) pytorch model
            dataset: (object) dataset
            split: (str) split of dataset in ['train', 'val', 'test']
          return [overall_accuracy, precision, recall, f1-score, jaccard, kappa]
        """
        args = self.args
        if not self.is_multi:
            oa, precision, recall, f1, jac, kappa = 0, 0, 0, 0, 0, 0
            if args.cuda:
                model.cuda()
            model.eval()
            data_loader = DataLoader(dataset, args.batch_size, num_workers=4,
                                     shuffle=False)
            batch_iterator = iter(data_loader)
            steps = len(dataset) // args.batch_size
    
            start = time.time()
            for step in range(steps):
                x, y = next(batch_iterator)
                x = torch.tensor(x)
                y = torch.tensor(y)
                if args.cuda:
                    x = x.cuda()
                    y = y.cuda()
                # calculate pixel accuracy of generator
                gen_y = model(x)
    #            print(gen_y.shape)
                oa += metrics.overall_accuracy(gen_y.data, y.data)
                precision += metrics.precision(gen_y.data, y.data)
                recall += metrics.recall(gen_y.data, y.data)
                f1 += metrics.f1_score(gen_y.data, y.data)
                jac += metrics.jaccard(gen_y.data, y.data)
                kappa += metrics.kappa(gen_y.data, y.data)
    
            _time = time.time() - start
    
            if not os.path.exists(os.path.join(Logs_DIR, 'statistic')):
                os.makedirs(os.path.join(Logs_DIR, 'statistic'))
    
            # recording performance of the model
            nb_samples = steps * args.batch_size
            basic_info = [self.date, self.args.is_freeze, self.model_basename, args.data_dir,
                          self.epoch, self.iter, nb_samples, _time]
            basic_info_names = ['date', 'is_freeze', 'base_model', 'dataset_trans','epochs',
                                'iters', 'nb_samples', 'time(sec)']
    
            perform = [round(idx / steps, 3)
                       for idx in [oa, precision, recall, f1, jac, kappa]]
            perform_names = ["overall_accuracy", "precision",
                             "recall", "f1-score", "jaccard", "kappa"]
            cur_log = pd.DataFrame([basic_info + perform],
                                   columns=basic_info_names + perform_names)
            # save performance
            if os.path.exists(os.path.join(Logs_DIR, 'statistic', "{}_trans.csv".format(split))):
                logs = pd.read_csv(os.path.join(
                    Logs_DIR, 'statistic', "{}_trans.csv".format(split)))
            else:
                logs = pd.DataFrame([])
            logs = logs.append(cur_log, ignore_index=True)
            logs.to_csv(os.path.join(Logs_DIR, 'statistic',
                                     "{}_trans.csv".format(split)), index=False, float_format='%.3f')
        
        else:
            metrics_mul = ConfusionMatrix(args.out_ch, args.class_names, useUnlabeled=True)
            if args.cuda:
                model.cuda()
            model.eval()
            data_loader = DataLoader(dataset, args.batch_size, num_workers=4,
                                     shuffle=False)
            batch_iterator = iter(data_loader)
            steps = len(dataset) // args.batch_size
    
            start = time.time()
            for step in range(steps):
                x, y = next(batch_iterator)
                x = torch.tensor(x)
                y = torch.tensor(y)
                if args.cuda:
                    x = x.cuda()
                    y = y.cuda()
                # calculate pixel accuracy of generator
                gen_y = model(x)
                # calculate mIoU

                pred = y.cpu().detach().numpy()
                gt = gen_y.cpu().detach().numpy()
                gt = np.argmax(gt, axis=1)
                metrics_mul.update_matrix(gt, pred)
            precisions, recalls, f1scores, mat, IoU, mIoU, \
                            accuracy, mprecision, mrecall, mf1score = metrics_mul.scores()
            metrics_mul.reset()
            _time = time.time() - start 
            
            if not os.path.exists(os.path.join(Logs_DIR, 'statistic')):
                os.makedirs(os.path.join(Logs_DIR, 'statistic'))
    
            # recording performance of the model
            nb_samples = steps * args.batch_size
            basic_info = [self.date, self.args.is_freeze, self.model_basename, args.data_dir,
                          self.epoch, self.iter, nb_samples, _time]
            basic_info_names = ['date', 'is_freeze', 'base_model', 'dataset_trans','epochs',
                                'iters', 'nb_samples', 'time(sec)']
                                                    
            np.set_printoptions(threshold=200)
            perform = [np.round(idx , 3)\
                       for idx in [precisions, recalls, f1scores, mat, IoU, mIoU, \
                            accuracy, mprecision, mrecall, mf1score ]]
            perform_names = ['precisions', 'recalls', 'f1scores', 'mat', 'IoU', 'mIoU', \
                             'over_accuracy', 'mprecision', 'mrecall', 'mf1score']
            cur_log = pd.DataFrame([basic_info + perform],
                                   columns=basic_info_names + perform_names)
            # save performance
            if os.path.exists(os.path.join(Logs_DIR, 'statistic', "{}_mul_trans.csv".format(split))):
                logs = pd.read_csv(os.path.join(
                    Logs_DIR, 'statistic', "{}_mul_trans.csv".format(split)))
            else:
                logs = pd.DataFrame([])
            logs = logs.append(cur_log, ignore_index=True)
            logs.to_csv(os.path.join(Logs_DIR, 'statistic',
                                     "{}_mul_trans.csv".format(split)), index=False, float_format='%.3f')


class Trainer_SRSEG(Base):
    def training(self, net, datasets):
        """
          input:
            net: (object) model & optimizer
            datasets : (list) [train, val] dataset object
        """
        args = self.args
        net.model.train()
        steps = len(datasets[0]) // args.batch_size
        if args.trigger == 'epoch':
            args.epochs = args.terminal
            args.iters = steps * args.terminal
            args.iter_interval = steps * args.interval
        else:
            args.epochs = args.terminal // steps + 1
            args.iters = args.terminal
            args.iter_interval = args.interval

        train_loss_sr, train_loss_seg = 0, 0
        train_loss, train_acc = 0, 0
        start = time.time()

        for epoch in range(1, args.epochs + 1):
            self.epoch = epoch
            # setup data loader
            data_loader = DataLoader(datasets[0], args.batch_size, num_workers=4,
                                     shuffle=True)
            batch_iterator = iter(data_loader)
            for step in range(steps):
                self.iter += 1
                if self.iter > args.iters:
                    self.iter -= 1
                    break
                # convert numpy.ndarray into pytorch tensor
                x, x_ori, y = next(batch_iterator)
                x = torch.tensor(x)
                x_ori = torch.tensor(x_ori)
                y = torch.tensor(y)
                if args.cuda:
                    x = x.cuda()
                    x_ori = x_ori.cuda()
                    y = y.cuda()
                # training
                _x, gen_y = net.model(x)

                loss_sr = torch.nn.functional.mse_loss(_x, x_ori)
                loss_seg = F.binary_cross_entropy(gen_y, y)
                loss = loss_sr + loss_seg * args.seg_para
                # Update generator parameters
                net.optimizer.zero_grad()
                loss.backward()
                net.optimizer.step()
                train_loss_sr += loss_sr.item()
                train_loss_seg += loss_seg.item()
                train_loss += loss.item()
                # train_loss += loss.data[0].cpu().numpy()
                #                train_acc += metrics.overall_accuracy(gen_y.data, y.data)
                if not self.is_multi:
                    train_acc += metrics.jaccard(gen_y.data, y.data)
                else:
                    train_acc += metrics.overall_accuracy(gen_y.data, y.data)

                # logging
                if self.iter % args.iter_interval == 0:
                    _time = time.time() - start
                    nb_samples = args.iter_interval * args.batch_size
                    train_log = [train_loss / args.iter_interval,
                                 train_loss_sr / args.iter_interval, train_loss_seg / args.iter_interval,
                                 train_acc /args.iter_interval, _time, nb_samples / _time]
                    self.train_log = [round(x, 3) for x in train_log]
                    self.validating(net.model, datasets[1])
                    self.logging(verbose=True)
                    # reinitialize
                    start = time.time()
                    train_loss_sr, train_loss_seg = 0, 0
                    train_loss, train_acc = 0, 0

        if args.save_best:
            net.model = self.best_model

    def validating(self, model, dataset):
        """
          input:
            model: (object) pytorch model
            batch_size: (int)
            dataset : (object) dataset
          return [val_acc, val_loss]
        """
        args = self.args
        if args.cuda:
            model.cuda()
        _val_loss_sr, _val_loss_seg = 0, 0
        val_loss, val_acc = 0, 0
        data_loader = DataLoader(dataset, args.batch_size, num_workers=4,
                                 shuffle=False)
        batch_iterator = iter(data_loader)
        steps = len(dataset) // args.batch_size
        model.eval()
        start = time.time()
        if self.is_multi:
            metrics_mul = ConfusionMatrix(args.out_ch, args.class_names, useUnlabeled=True)
        for step in range(steps):
            x, x_ori, y = next(batch_iterator)
            x = torch.tensor(x)
            x_ori = torch.tensor(x_ori)
            y = torch.tensor(y)
            if args.cuda:
                x = x.cuda()
                x_ori = x_ori.cuda()
                y = y.cuda()

            # calculate pixel accuracy of generator
            _x, gen_y = model(x)
            #            if self.is_multi:
            #                gen_y = gen_y[0]
            val_loss_sr = torch.nn.functional.mse_loss(_x, x_ori).item()
            val_loss_seg = F.binary_cross_entropy(gen_y, y).item()

            _val_loss_sr += val_loss_sr
            _val_loss_seg += val_loss_seg

            val_loss += val_loss_sr + val_loss_seg * args.seg_para
            if not self.is_multi:
                val_acc += metrics.jaccard(gen_y.data, y.data)
            else:
                # Note: need optimize into mIoU
                pred = y.cpu().detach().numpy()
                gt = gen_y.cpu().detach().numpy()
                gt = np.argmax(gt, axis=1)
                metrics_mul.update_matrix(gt, pred)
        if self.is_multi:
            precisions, recalls, f1scores, mat, IoU, mIoU, \
            accuracy, mprecision, mrecall, mf1score = metrics_mul.scores()
            metrics_mul.reset()
            val_acc = mIoU

        _time = time.time() - start
        nb_samples = steps * args.batch_size
        if not self.is_multi:
            val_log = [val_loss / steps, _val_loss_sr / steps, _val_loss_seg / steps, val_acc /
                       steps, _time, nb_samples / _time]
        else:
            val_log = [val_loss / steps, val_acc,
                       _time, nb_samples / _time]

        self.val_log = [round(x, 3) for x in val_log]

        if args.save_best:
            self.best_checkpoint(model, val_log)

    def best_checkpoint(self, model, val_log):
        """
        log: val_log
        """
        val_acc = val_log[3]
        if val_acc >= self.best_acc:
            self.best_acc = val_acc
            self.best_model = copy.deepcopy(model)
            self.best_val_log = val_log
            self.best_train_log = self.train_log

    def evaluating(self, model, dataset, split):
        """
          input:
            model: (object) pytorch model
            dataset: (object) dataset
            split: (str) split of dataset in ['train', 'val', 'test']
          return [overall_accuracy, precision, recall, f1-score, jaccard, kappa]
        """
        args = self.args
        if not self.is_multi:
            oa, precision, recall, f1, jac, kappa = 0, 0, 0, 0, 0, 0
            if args.cuda:
                model.cuda()
            model.eval()
            data_loader = DataLoader(dataset, args.batch_size, num_workers=4,
                                     shuffle=False)
            batch_iterator = iter(data_loader)
            steps = len(dataset) // args.batch_size

            start = time.time()
            for step in range(steps):
                x, x_ori, y = next(batch_iterator)
                x = torch.tensor(x)
                x_ori = torch.tensor(x_ori)
                y = torch.tensor(y)
                if args.cuda:
                    x = x.cuda()
                    x_ori = x_ori.cuda()
                    y = y.cuda()
                # calculate pixel accuracy of generator
                _x, gen_y = model(x)
                #            print(gen_y.shape)
                oa += metrics.overall_accuracy(gen_y.data, y.data)
                precision += metrics.precision(gen_y.data, y.data)
                recall += metrics.recall(gen_y.data, y.data)
                f1 += metrics.f1_score(gen_y.data, y.data)
                jac += metrics.jaccard(gen_y.data, y.data)
                kappa += metrics.kappa(gen_y.data, y.data)

            _time = time.time() - start

            if not os.path.exists(os.path.join(Logs_DIR, 'statistic')):
                os.makedirs(os.path.join(Logs_DIR, 'statistic'))

            # recording performance of the model
            nb_samples = steps * args.batch_size
            basic_info = [self.date, self.method, args.data_dir,
                          self.epoch, self.iter, nb_samples, _time]
            basic_info_names = ['date', 'method', 'dataset_train', 'epochs',
                                'iters', 'nb_samples', 'time(sec)']

            perform = [round(idx / steps, 3)
                       for idx in [oa, precision, recall, f1, jac, kappa]]
            perform_names = ["overall_accuracy", "precision",
                             "recall", "f1-score", "jaccard", "kappa"]
            cur_log = pd.DataFrame([basic_info + perform],
                                   columns=basic_info_names + perform_names)
            # save performance
            if os.path.exists(os.path.join(Logs_DIR, 'statistic', "{}.csv".format(split))):
                logs = pd.read_csv(os.path.join(
                    Logs_DIR, 'statistic', "{}.csv".format(split)))
            else:
                logs = pd.DataFrame([])
            logs = logs.append(cur_log, ignore_index=True)
            logs.to_csv(os.path.join(Logs_DIR, 'statistic',
                                     "{}.csv".format(split)), index=False, float_format='%.3f')
        else:
            metrics_mul = ConfusionMatrix(args.out_ch, args.class_names, useUnlabeled=True)
            if args.cuda:
                model.cuda()
            model.eval()
            data_loader = DataLoader(dataset, args.batch_size, num_workers=4,
                                     shuffle=False)
            batch_iterator = iter(data_loader)
            steps = len(dataset) // args.batch_size

            start = time.time()
            for step in range(steps):
                x, y = next(batch_iterator)
                x = torch.tensor(x)
                y = torch.tensor(y)
                if args.cuda:
                    x = x.cuda()
                    y = y.cuda()
                # calculate pixel accuracy of generator
                gen_y = model(x)
                # calculate mIoU

                pred = y.cpu().detach().numpy()
                gt = gen_y.cpu().detach().numpy()
                gt = np.argmax(gt, axis=1)
                metrics_mul.update_matrix(gt, pred)
            precisions, recalls, f1scores, mat, IoU, mIoU, \
            accuracy, mprecision, mrecall, mf1score = metrics_mul.scores()
            metrics_mul.reset()
            _time = time.time() - start

            if not os.path.exists(os.path.join(Logs_DIR, 'statistic')):
                os.makedirs(os.path.join(Logs_DIR, 'statistic'))

            # recording performance of the model
            nb_samples = steps * args.batch_size
            basic_info = [self.date, self.method, args.data_dir,
                          self.epoch, self.iter, nb_samples, _time]
            basic_info_names = ['date', 'method', 'dataset_train', 'epochs',
                                'iters', 'nb_samples', 'time(sec)']

            np.set_printoptions(threshold=200)
            perform = [np.round(idx, 3) \
                       for idx in [precisions, recalls, f1scores, mat, IoU, mIoU, \
                                   accuracy, mprecision, mrecall, mf1score]]
            perform_names = ['precisions', 'recalls', 'f1scores', 'mat', 'IoU', 'mIoU', \
                             'over_accuracy', 'mprecision', 'mrecall', 'mf1score']
            cur_log = pd.DataFrame([basic_info + perform],
                                   columns=basic_info_names + perform_names)
            # save performance
            if os.path.exists(os.path.join(Logs_DIR, 'statistic', "{}_mul.csv".format(split))):
                logs = pd.read_csv(os.path.join(
                    Logs_DIR, 'statistic', "{}_mul.csv".format(split)))
            else:
                logs = pd.DataFrame([])
            logs = logs.append(cur_log, ignore_index=True)
            pd.set_option('display.width', None)
            logs.to_csv(os.path.join(Logs_DIR, 'statistic',
                                     "{}_mul.csv".format(split)), index=False, float_format='%.3f')


class Tester_SRSEG(Base):
    def testing(self, model, dataset, split):
        """
          input:
            model: (object) pytorch model
            dataset: (object) dataset
            split: (str) split of dataset in ['train', 'val', 'test']
          return [overall_accuracy, precision, recall, f1-score, jaccard, kappa]
        """
        args = self.args
        self.model_basename = args.model_name.strip('.pth')
        if args.cuda:
            model.cuda()
        model.eval()
        data_loader = DataLoader(dataset, args.batch_size, num_workers=4,
                                 shuffle=False)
        batch_iterator = iter(data_loader)
        steps = len(dataset) // args.batch_size

        if not self.is_multi:
            start = time.time()
            oa, precision, recall, f1, jac, kappa = 0, 0, 0, 0, 0, 0
            for step in range(steps):
                x, x_ori, y = next(batch_iterator)
                x = torch.tensor(x)
                y = torch.tensor(y)
                if args.cuda:
                    x = x.cuda()
                    y = y.cuda()
                # calculate pixel accuracy of generator
                _x, gen_y = model(x)
                oa += metrics.overall_accuracy(gen_y.data, y.data)
                precision += metrics.precision(gen_y.data, y.data)
                recall += metrics.recall(gen_y.data, y.data)
                f1 += metrics.f1_score(gen_y.data, y.data)
                jac += metrics.jaccard(gen_y.data, y.data)
                kappa += metrics.kappa(gen_y.data, y.data)
            _time = time.time() - start

            if not os.path.exists(os.path.join(Logs_DIR, 'statistic')):
                os.makedirs(os.path.join(Logs_DIR, 'statistic'))

            # recording performance of the model
            nb_samples = steps * args.batch_size
            basic_info = [self.date, args.model_name, args.data_dir,
                          nb_samples, _time]
            basic_info_names = ['date', 'method', 'dataset_test',
                                'nb_samples', 'time(sec)']

            perform = [round(idx / steps, 3)
                       for idx in [oa, precision, recall, f1, jac, kappa]]
            perform_names = ["overall_accuracy", "precision",
                             "recall", "f1-score", "jaccard", "kappa"]
            cur_log = pd.DataFrame([basic_info + perform],
                                   columns=basic_info_names + perform_names)
            # save performance
            if os.path.exists(os.path.join(Logs_DIR, 'statistic', "{}.csv".format(split))):
                logs = pd.read_csv(os.path.join(
                    Logs_DIR, 'statistic', "{}.csv".format(split)))
            else:
                logs = pd.DataFrame([])
            logs = logs.append(cur_log, ignore_index=True)
            logs.to_csv(os.path.join(Logs_DIR, 'statistic',
                                     "{}.csv".format(split)), index=False, float_format='%.3f')

        else:
            start = time.time()
            metrics_mul = ConfusionMatrix(args.out_ch, args.class_names, useUnlabeled=True)
            for step in range(steps):
                x, x_ori, y = next(batch_iterator)
                x = torch.tensor(x)
                y = torch.tensor(y)
                if args.cuda:
                    x = x.cuda()
                    y = y.cuda()
                # calculate pixel accuracy of generator
                _x, gen_y = model(x)
                # calculate mIoU
                pred = y.cpu().detach().numpy()
                gt = gen_y.cpu().detach().numpy()
                gt = np.argmax(gt, axis=1)
                metrics_mul.update_matrix(gt, pred)
            precisions, recalls, f1scores, mat, IoU, mIoU, \
            accuracy, mprecision, mrecall, mf1score = metrics_mul.scores()
            metrics_mul.reset()
            _time = time.time() - start

            if not os.path.exists(os.path.join(Logs_DIR, 'statistic')):
                os.makedirs(os.path.join(Logs_DIR, 'statistic'))

            nb_samples = steps * args.batch_size
            # recording performance of the model
            basic_info = [self.date, args.model_name, args.data_dir,
                          nb_samples, _time]
            basic_info_names = ['date', 'method', 'dataset_test',
                                'nb_samples', 'time(sec)']

            np.set_printoptions(threshold=200)
            perform = [np.round(idx, 3) \
                       for idx in [precisions, recalls, f1scores, mat, IoU, mIoU, \
                                   accuracy, mprecision, mrecall, mf1score]]
            perform_names = ['precisions', 'recalls', 'f1scores', 'mat', 'IoU', 'mIoU', \
                             'over_accuracy', 'mprecision', 'mrecall', 'mf1score']
            cur_log = pd.DataFrame([basic_info + perform],
                                   columns=basic_info_names + perform_names)
            # save performance
            if os.path.exists(os.path.join(Logs_DIR, 'statistic', "{}_mul.csv".format(split))):
                logs = pd.read_csv(os.path.join(
                    Logs_DIR, 'statistic', "{}_mul.csv".format(split)))
            else:
                logs = pd.DataFrame([])
            logs = logs.append(cur_log, ignore_index=True)
            logs.to_csv(os.path.join(Logs_DIR, 'statistic',
                                     "{}_mul.csv".format(split)), index=False, float_format='%.3f')


if __name__ == "__main__":
    # ====================== parameter initialization ======================= #
    print("Hello")
