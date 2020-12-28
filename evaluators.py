from collections import OrderedDict, defaultdict
from utils import Bar
from utils import re_ranking
import numpy as np
from sklearn.metrics import average_precision_score
from torch.autograd import Variable
import torch.nn as nn
import torch
import os
import copy
import random
import pdb

class Evaluator(object):
    def __init__(self, cfg, model, dataset):
        self.model = model
        self.cfg = cfg
        _, _, self.test, self.query, self.gallery, _, _, _, _ = dataset


    def evaluate_VeID(self):
        self.model.eval()
        features, labels = self.extract_features()
        # caculate 10 times
        #self.gallery, self.query = [], []
        mAP_list , CMC_1, CMC_5, CMC_10 = [], [], [], []
        for i in range(10):
            self.gallery, self.query = [], []
            vid_imgs_list = copy.deepcopy(self.vid_imgs_list)
            for vid in vid_imgs_list:
                sampled_idx = random.randint(0, len(vid_imgs_list[vid])-1)
                sampled_imgName = vid_imgs_list[vid][sampled_idx]
                vid_imgs_list[vid].remove(sampled_imgName)
                imgPath = os.path.join('/home/share/zhihui/VehicleID_V1.0/image', '%s.jpg'%sampled_imgName)
                self.gallery.append([imgPath ,vid, sampled_imgName])
                for imgName in vid_imgs_list[vid]:
                    imgPath = os.path.join('/home/share/zhihui/VehicleID_V1.0/image', '%s.jpg'%imgName)
                    self.query.append([imgPath, vid, imgName])
            #pdb.set_trace()
            distmat = self.pairwise_distance(features)
            mAP, CMC = self.calculate_mAP_CMC(distmat)
            mAP_list.append(mAP)
            CMC_1.append(CMC[0])
            CMC_5.append(CMC[4])
            CMC_10.append(CMC[9])

        mAP = np.array(mAP_list).mean()
        cmc1 = np.array(CMC_1).mean()
        cmc5 = np.array(CMC_5).mean()
        cmc10 = np.array(CMC_10).mean()

        return mAP, cmc1, cmc5, cmc10

    def evaluate(self, eval_cls=False, vis_flag=False):
        self.model.eval()
        features, labels = self.extract_features()
        distmat = self.pairwise_distance(features)
        mAP, CMC = self.calculate_mAP_CMC(distmat)

        if vis_flag:
            f = open(os.path.join('logs', self.cfg['NAME'], '%s.txt'%self.cfg['NAME']), 'w')
            query_name = [q[-1] for q in self.query]
            gallery_name = [g[-1] for g in self.gallery]
            query_cam = [q[2] for q in self.query]
            gallery_cam = [g[2] for g in self.gallery]
            query_name = np.asarray(query_name)
            gallery_name = np.asarray(gallery_name)
            query_cam = np.asarray(query_cam)
            gallery_cam = np.asarray(gallery_cam)
            indices = np.argsort(distmat.numpy(), axis=1)
            m, n = distmat.shape
            for i in range(m):
                num = 0
                valid = (gallery_cam[indices[i]] != query_cam[i])
                for idx, j in enumerate(indices[i]):
                    if valid[idx]:
                        f.write(str(gallery_name[j])+' ')
                        num += 1
                    if num > 100:
                        break
                f.write('\n')
            f.close()

        return mAP, CMC

    def extract_features(self):

        features = OrderedDict()
        labels = OrderedDict()
        #
        self.vid_imgs_list = defaultdict(list)


        bar = Bar('Extracting features', max=len(self.test))

        for i,inputs in enumerate(self.test):
            imgs, vids, fnames = Variable(inputs[0]), inputs[1], inputs[-1]
            self.model.eval()
            feats = self.model(imgs)
            #normalize
            feats = nn.functional.normalize(feats,dim=1, p=2)
            feats = feats.data.cpu()
            for fname, feat, vid in zip(fnames, feats, vids):
                features[fname] = feat
                labels[fname] = vid.item()
                self.vid_imgs_list[vid.item()].append(fname)
                bar.suffix = '[{cur}/{amount}]'.format(cur=i+1, amount=len(self.test))
            bar.next()
        bar.finish()


        return features, labels

    def pairwise_distance(self, features):
        x = torch.cat([features[q[-1]].unsqueeze(0) for q in self.query], dim=0)
        y = torch.cat([features[g[-1]].unsqueeze(0) for g in self.gallery], dim=0)

        m,n = x.size(0), y.size(0)
        x = x.view(m, -1)
        y = y.view(n,-1)
        dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m,n) + torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n,m).t()
        dist.addmm_(1, -2, x, y.t())

        return dist

    def calculate_mAP_CMC(self, distmat):
        query_ids = [q[1] for q in self.query]
        gallery_ids = [g[1] for g in self.gallery]
        query_cams = [q[2] for q in self.query]
        gallery_cams = [g[2] for g in self.gallery]

        '''
        if self.cfg['TARGET'] in ['VeRi', 'VeRi_Wild']:
            query_cams = [q[2] for q in self.query]
            gallery_cams = [g[2] for g in self.gallery]
        else:
            query_cams = [q[-1] for q in self.query]
            gallery_cams = [g[-1] for g in self.gallery ]
        '''
        query_ids = np.asarray(query_ids)
        gallery_ids = np.asarray(gallery_ids)
        query_cams = np.asarray(query_cams)
        gallery_cams = np.asarray(gallery_cams)

        mAP = self.mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
        CMC_scores = self.CMC(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
        '''
        print('Mean AP: {:4.1%}'.format(mAP))
        print("CMC:\n")
        for k in [1,5,10]:
            print('top-{:<4}{:12.1%}'
                 .format(k, CMC_scores[k-1]))
        '''
        return mAP, CMC_scores

    def mean_ap(self, distmat, query_ids, gallery_ids, query_cams, gallery_cams):
        distmat = distmat.cpu().numpy()
        m,n = distmat.shape

        indices = np.argsort(distmat, axis=1)
        matches = (gallery_ids[indices] == query_ids[:, np.newaxis])

        aps = []
        for i in range(m):
            valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                    (gallery_cams[indices[i]] != query_cams[i]))

            y_true = matches[i, valid]
            y_score = -distmat[i][indices[i]][valid]

            if not np.any(y_true): continue
            aps.append(average_precision_score(y_true, y_score))
        if len(aps) == 0:
            raise RuntimeError("No valid query")
        return np.mean(aps)

    def CMC(self, distmat, query_ids, gallery_ids, query_cams, gallery_cams, topk=100):
        distmat = distmat.cpu().numpy()
        m,n = distmat.shape

        indices = np.argsort(distmat, axis=1)
        matches = (gallery_ids[indices] == query_ids[:, np.newaxis])

        ret = np.zeros(topk)
        num_valid_queries = 0
        for i in range(m):
            valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                    (gallery_cams[indices[i]] != query_cams[i]))

            y_true = matches[i, valid]
            if not np.any(y_true): continue
            index = np.nonzero(y_true)[0]
            if index.flatten()[0] < topk:
                ret[index.flatten()[0]] += 1
            num_valid_queries += 1
        if num_valid_queries == 0:
            raise RuntimeError("No valid query")

        return ret.cumsum() / num_valid_queries


