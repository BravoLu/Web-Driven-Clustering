# -*- coding: utf-8 -*-
# @Author: Lu Shaohao(Bravo)
# @Date:   2019-08-20 16:52:36
# @Last Modified by:   Lu Shaohao(Bravo)
# @Last Modified time: 2019-11-14 08:01:18
import time
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import pdb
import pickle
import shutil

from loss import *
from utils import *
from utils import Bar
from optim import *
from evaluators import Evaluator

class BaseTrainer(object):
    def __init__(self, cfg, model, dataset):
        super(BaseTrainer, self).__init__()
        self.cfg = cfg
        self.logger = Logger(cfg)
        #self.debugger = SummaryWriter(os.path.join('debug', cfg['NAME'], 'loss' ))
        #self.mAP_marker = SummaryWriter(os.path.join('debug', cfg['NAME'], 'mAP'))
        self.source_loader, self.target_loader, self.test_loader, self.query, self.gallery, self.train_transformer,self.source_train, self.target_train, self.target_cluster_loader = dataset
        self.best_mAP = 0
        self.num_gpus = torch.cuda.device_count()
        if os.path.exists(self.cfg['PRETRAIN']):
            model.load_param(self.cfg['PRETRAIN'])
            print("load checkpoint from {}".format(self.cfg['PRETRAIN']))
        self.model = nn.DataParallel(model).cuda()
        self.evaluator = Evaluator(self.cfg, self.model, dataset)

        self.num_gpus = torch.cuda.device_count()

        self.optimizer = make_optimizer(self.cfg, self.model, num_gpus=self.num_gpus)
        self.scheduler = WarmupMultiStepLR(self.optimizer, self.cfg['MILESTONES'], self.cfg['GAMMA'], self.cfg['WARMUP_FACTOR'])
        #self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.cfg['LR'], momentum=0.9, weight_decay=0, nesterov=True)
        #self.scheduler = MultiStepLR(self.optimizer, milestones=self.cfg['MILESTONES'])

        self.num_gpus = torch.cuda.device_count()
        self.logger.write('num gpus:{} \n'.format(self.num_gpus))

    def train(self):

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.cfg['LR'], momentum=0.9, weight_decay=0, nesterov=True)
        self.scheduler = MultiStepLR(self.optimizer, milestones=self.cfg['MILESTONES'])
        CE = nn.CrossEntropyLoss().cuda()

        for epoch in range(self.cfg['EPOCHS']):
            self.model.train()
            stats = ('ce_loss', 'total_loss')
            meters_trn = {stat: AverageMeter() for stat in stats}

            for i,inputs in enumerate(self.source_loader):
                imgs = Variable(inputs[0])
                labels = Variable(inputs[1]).cuda()
                scores = self.model(imgs, state='web stream')

                ce_loss = CE(scores, labels)
                total_loss = ce_loss
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                for k in stats:
                    v = locals()[k]
                    meters_trn[k].update(v.item(), self.cfg['BATCHSIZE'])

            self.logger.write("epoch: %d | lr: %.5f | loss: %.5f | \n"%(
                epoch+1,
                self.scheduler.get_lr()[0],
                meters_trn['ce_loss'].avg,
            ))
            self.scheduler.step()

            self.evaluate(epoch, stats)

    def evaluate(self, epoch, stats=None):

        if self.cfg['TARGET'] == 'VehicleID':
            mAP, cmc1, cmc5, cmc10 = self.evaluator.evaluate_VeID()
        else:
            mAP, cmc = self.evaluator.evaluate(eval_cls=True)

        cmc1, cmc5, cmc10 = cmc[0], cmc[4], cmc[9]

        '''
        if stats is not None:
            for stat in stats:
                self.mAP_marker.add_scalar(stat, mAP, epoch+1)
        '''
        is_best = mAP > self.best_mAP
        self.best_mAP = max(mAP, self.best_mAP)
        self.logger.write("mAP: {:.1f}% | cmc-1: {:.1f}% | cmc-5: {:.1f}% | cmc-10: {:.1f}% | Best mAP: {:.1f}% |\n".format(mAP * 100, cmc1 * 100, cmc5 * 100, cmc10 * 100, self.best_mAP * 100))
        self.logger.write("==========================================\n")
        save_checkpoint({
            'state_dict':self.model.module.state_dict(),
            'epoch':epoch+1,
            'best_mAP': self.best_mAP,
        }, is_best=is_best, fpath=os.path.join("ckpt", self.cfg['NAME'], 'checkpoint.pth'))

    def cls_visualization(self):

        for i,inputs in enumerate(self.target_loader):
            imgs, _, fnames = inputs[0], inputs[1], inputs[-1]
            self.model.eval()
            cls_score, _ = self.model(imgs, 'auxiliary')
            predict = torch.max(cls_score, dim=1)[1].data.squeeze()
            for p, fname in zip(predict, fnames):
                dir_ = os.path.join('vis', self.cfg['CLS_PATH'])
                mkdir_if_missing(os.path.join(dir_, '%d'%(p.item())))
                dst = os.path.join(dir_, '%d'%(p.item()), fname+'.jpg')
                src = os.path.join('/home/share/zhihui/VeRi/image_train/', fname+'.jpg')
                shutil.copyfile(src, dst)

    def tSNE(self, img_path='tSNE.jpg'):

        source_feats, aux_feats = [], []
        source_labels, aux_labels = [], []

        for i,inputs in enumerate(self.source_loader):
            imgs, vids = Variable(inputs[0]).cuda(), inputs[1]
            outputs = self.model(imgs)
            for output, vid in zip(outputs, vids):
                source_feats.append(output.data.cpu().numpy().tolist())
                source_labels.append('VehicleID')
            '''
            source_feats = np.array(source_feats)
            source_labels = np.array(source_labels)
            '''

        for i,inputs in enumerate(self.auxiliary_loader):
            imgs, tids = Variable(inputs[0]).cuda(), inputs[1]
            outputs = self.model(imgs)
            for output, tid in zip(outputs, tids):
                aux_feats.append(output.data.cpu().numpy().tolist())
                aux_labels.append('CompCars')

        tsne = TSNE(n_components=2, init='pca', random_state=501)

        source_feats = np.array(source_feats[:1000])
        aux_feats = np.array(aux_feats[:1000])
        feats = np.concatenate((source_feats,aux_feats), axis=0)
        labels = source_labels[:1000] + aux_labels[:1000]

        pickle.dump(feats, open('feat.pkl', 'wb'))
        pickle.dump(labels, open('labels.pkl', 'wb'))

        '''
        newData = tsne.fit_transform(np.array(feats))
        newLab = pd.Series(labels, name="identity")
        xData, yData = pd.Series(newData[:,0], name='x'),pd.Series(newData[:,1], name='y')

        pdb.set_trace()
        ax = sns.scatterplot(x=xData,y=yData,hue=newLab,legend="full")

        ax.grid(True)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width, box.height*0.85])
        ax.legend(loc='center left', bbox_to_anchor=(0.2,1.2), ncol=3)
        fig = ax.get_figure()
        fig.savefig(img_path)
        '''


