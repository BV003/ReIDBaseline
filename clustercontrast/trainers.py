from __future__ import print_function, absolute_import
import time
from .utils.meters import AverageMeter
import torch.nn as nn
import torch
from torch.nn import functional as F
from torch.nn import Module
import collections
from torch import einsum
from torch.autograd import Variable
from clustercontrast.models.cm import ClusterMemory
from clustercontrast.utils.faiss_rerank import compute_jaccard_distance,compute_ranked_list,compute_ranked_list_cm
import numpy as np                
class Trainer(object):
    def __init__(self, encoder, memory=None):
        super(Trainer, self).__init__()
        self.encoder = encoder
        self.memory_sk = memory
        self.memory_rgb = memory

    def train(self, epoch, data_loader_sk,data_loader_rgb, optimizer, print_freq=10, train_iters=400):
        self.encoder.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        end = time.time()
        for i in range(train_iters):
            print("load data")
            # load data
            inputs_sk = data_loader_sk.next()
            inputs_rgb = data_loader_rgb.next()
            data_time.update(time.time() - end)
            print("process inputs")
            # process inputs
            inputs_sk, labels_sk, indexes_sk = self._parse_data_sk(inputs_sk)
            inputs_rgb, labels_rgb, indexes_rgb = self._parse_data_rgb(inputs_rgb)
            print("forward")
            # forward
            _,f_out_rgb,f_out_sk,_,_,labels_rgb,labels_sk,pool_rgb,pool_sk,_,_ = self._forward(inputs_rgb,inputs_sk,label_1=labels_rgb,label_2=labels_sk,modal=0)
            loss_sk = self.memory_sk(f_out_sk, labels_sk)
            loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)
            loss = loss_sk+loss_rgb
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.update(loss.item())
            # print log
            batch_time.update(time.time() - end)
            end = time.time()
            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Loss sk {:.3f}\t'
                      'Loss rgb {:.3f}\t'
                      .format(epoch, i + 1, len(data_loader_rgb),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,loss_sk,loss_rgb))

    def _parse_data_rgb(self, inputs):
        imgs,_,_, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _parse_data_sk(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, x1, x2, label_1=None,label_2=None,modal=0):
        return self.encoder(x1, x2, modal=modal,label_1=label_1,label_2=label_2)



