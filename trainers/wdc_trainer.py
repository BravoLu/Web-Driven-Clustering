# -*- coding: utf-8 -*-
# @Author: Lu Shaohao(Bravo)
# @Date:   2019-10-23 08:22:06
# @Last Modified by:   Lu Shaohao(Bravo)
# @Last Modified time: 2019-11-10 20:50:19
from collections import OrderedDict, defaultdict
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm
import shutil
import pdb
import utils
import pickle

from .base_trainer import *
from dataloader import *
from models.resnet50 import *
class WDCTrainer(BaseTrainer):

    def train(self):

        for epoch in range(self.cfg['EPOCHS']):
            # generate new target pseudo label
            self.model.eval()
            #extract features
            features = OrderedDict()
            tgt_type_labels = OrderedDict()
            type_sets = set()
            for i,data in tqdm(enumerate(self.target_cluster_loader), desc='Extracting features'):
                #pseudo was cls type given by classifier pretrained
                imgs, fnames= data[0], data[-1]
                outputs = self.model(imgs)
                outputs = nn.functional.normalize(outputs, dim=1, p=2)
                tgt_type = self.model(imgs, 'target classification')
                tgt_type = torch.max(tgt_type, dim=1)[1].data.squeeze()
                tgt_type = tgt_type.data.cpu()
                outputs = outputs.data.cpu()
                for fname, output, type_ in zip(fnames, outputs, tgt_type):
                    features[fname] = output
                    tgt_type_labels[fname] = type_.item()
                    type_sets.add(type_.item())

            cls_cluster_pair = defaultdict(int)
            cls_cluster_features = defaultdict(list)
            tgt_new_train = []
            # DBSCAN in each pseudo-based set
            start = time.time()
            for pseudo in tqdm(type_sets, desc='Clustering'):
                cls_features, fnames, fpaths = [], [], []
                for fpath, __, _, _, _, fname in self.target_train:
                    if tgt_type_labels[fname] == pseudo:
                        fnames.append(fname)
                        fpaths.append(fpath)
                        cls_features.append(features[fname].unsqueeze(0))
                cls_features = torch.cat(cls_features)
                cls_features = cls_features.numpy()

                dist = cdist(cls_features, cls_features)

                tri_mat = np.triu(dist, 1)
                tri_mat = tri_mat[np.nonzero(tri_mat)]
                tri_mat = np.sort(tri_mat, axis=None)

                # Adaptive alpha
                cls_cluster_rho = self.linear_adaptive_distance_threshold(epoch)
                #cls_cluster_rho = self.cfg['FIRST_STEP_CLUSTERING_RATIO']
                top_num = np.round(cls_cluster_rho*tri_mat.size).astype(int)
                if top_num == 0:
                    continue
                eps = tri_mat[:top_num].mean()
                '''
                eps += 0.05 * epoch
                eps = min(eps, 1.2)
                '''
                min_samples = 3

                cluster = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed',  n_jobs=8)
                try:
                    labels = cluster.fit_predict(dist)
                except ValueError:
                    self.logger.write('skip this class\n')
                    continue
                cluster_num = len(set(labels)) - 1
                after_num = 0
                for fpath, fname, label in zip(fpaths, fnames, labels):
                    if label == -1:
                        continue
                    after_num += 1
                    if (pseudo, label) in cls_cluster_pair:
                        id_ = cls_cluster_pair[(pseudo, label)]
                        cls_cluster_features[id_].append(features[fname].unsqueeze(0))
                        tgt_new_train.append([fpath, id_, fname])
                    else:
                        cls_cluster_pair[(pseudo, label)] = len(cls_cluster_pair)
                        id_ = cls_cluster_pair[(pseudo, label)]
                        cls_cluster_features[id_].append(features[fname].unsqueeze(0))
                        tgt_new_train.append([fpath, id_, fname])

                #self.logger.write("pseudo:{:3} | imgs num:{:5} | atfer num:{:5} | cluster num:{:4} | eps:{:.3f} | \n".format(pseudo,cls_features.shape[0], after_num, cluster_num, eps))
            end = time.time()
            print('first phase time consume %f'%(end - start))
            cluster_mean_feat = []
            #cluster again
            cluster_size_list = []

            for key,value in cls_cluster_features.items():
                mean_feat = torch.cat(value).numpy().mean(axis=0)
                cluster_mean_feat.append(mean_feat)

            cluster_mean_feat = np.array(cluster_mean_feat)
            dist = cdist(cluster_mean_feat, cluster_mean_feat)
            tri_mat = np.triu(dist, 1)
            tri_mat = tri_mat[np.nonzero(tri_mat)]
            #for tri_matin tri_mat:
            tri_mat = np.sort(tri_mat, axis=None)
            top_num = np.round(self.cfg['SECOND_STEP_CLUSTERING_RATIO']*tri_mat.size).astype(int)

            eps = tri_mat[:top_num].mean()

            #eps = 1e-6
            min_samples = 1

            start = time.time()
            cluster = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed', n_jobs=8)
            end = time.time()
            print('Second phase time consumed: %s'%(end - start))
            labels = cluster.fit_predict(dist)
            cluster_num = len(set(labels))
            #self.logger.write("top_num {:2} original ids num :{:4} | final ids num :{:4}\n".format(top_num ,len(cls_cluster_pair), cluster_num))
            for idx, (fpath, id_, fname) in enumerate(tgt_new_train):
                tgt_new_train[idx][1] = labels[id_]

            self.pseudo_ids = cluster_num

            self.logger.write('---- Two step clustering informations ----\n')
            self.logger.write('image number: %d\n'%len(tgt_new_train))
            self.logger.write('first stage class number: %d\n'%len(cls_cluster_pair))
            self.logger.write('final stage class number: %d\n'%self.pseudo_ids)
            self.logger.write('alpha: %f\n'%cls_cluster_rho)
            self.logger.write('------------------------------------------\n')

            tgt_new_loader = DataLoader(
                    Preprocessor(tgt_new_train, name='Pseudo', transform=self.train_transformer),
                    batch_size=self.cfg['BATCHSIZE'],
                    num_workers=4,
                    sampler=RandomIdentitySampler(tgt_new_train, self.cfg['BATCHSIZE'], self.cfg['INSTANCE'], self.cfg['TARGET']),
                    pin_memory=True,
                    drop_last=True,
            )

            for i in range(self.cfg['EPOCHS_PER_ITER']):
                self.train_epoch(epoch ,i, tgt_new_loader)

    def train_epoch(self, iter_, epoch, tgt_new_loader):

        Triplet = TripletLoss(margin=self.cfg['MARGIN']).cuda()
        #CE =  CrossEntropyLabelSmooth(num_classes=749).cuda()
        CE = nn.CrossEntropyLoss().cuda()
        #MSE = nn.BCEWithLogitsLoss().cuda()
        self.model.train()

        stats = ('web_loss', 'reid_loss', 'total_loss')
        meters_trn = {stat: AverageMeter() for stat in stats}
        src_iter = iter(self.source_loader)

        for i,inputs in enumerate(tgt_new_loader):
            tgt_imgs = Variable(inputs[0]).cuda()
            tgt_labels = Variable(inputs[1]).cuda()

            try:
                src_tuple = next(src_iter)
            except StopIteration:
                src_iter = iter(self.source_loader)
                src_tuple = next(src_iter)

            src_imgs, src_labels = Variable(src_tuple[0]).cuda(), Variable(src_tuple[1]).cuda()

            src_scores = self.model(src_imgs, 'web stream')
            tgt_feats  = self.model(tgt_imgs, 're-id stream')

            web_loss = CE(src_scores,  src_labels) #+ CE(att_scores, src_labels)
            reid_loss = Triplet(tgt_feats, tgt_labels)

            total_loss = self.cfg['ALPHA'] * web_loss + \
                        self.cfg['DELTA'] * reid_loss


            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            for k in stats:
                v = locals()[k]
                meters_trn[k].update(v.item(), self.cfg['BATCHSIZE'])

        self.logger.write("epoch: %d | lr: %.8f | web_loss: %.5f | reid_loss: %.5f | \n"%(
                    epoch+1+iter_*self.cfg['EPOCHS_PER_ITER'],
                    self.scheduler.get_lr()[0],
                    meters_trn['web_loss'].avg,
                    meters_trn['reid_loss'].avg,
            ))
        self.scheduler.step()

        if epoch == self.cfg['EPOCHS_PER_ITER'] - 1:
            self.evaluate(epoch+iter_, stats)


    def linear_adaptive_distance_threshold(self, epoch):
        init_alpha = self.cfg['FIRST_STEP_CLUSTERING_RATIO']
        max_alpha =  init_alpha * 10
        increment =  (max_alpha - init_alpha)  * (epoch /  (self.cfg['EPOCHS'] - 1))
        current_alpha = init_alpha + increment
        return current_alpha
