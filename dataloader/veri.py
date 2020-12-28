import os
import os.path as osp
import csv
import pandas as pd
import random
import json
from collections import defaultdict
class VeRi(object):
    def __init__(self, root='./datas', verbose=True):
        self.root = root
        self.train_num = 0
        self.gallery_num = 0
        self.gallery = []
        self.train = []
        self.query = []
        self.test = []
        self.test_num = 0
        self.num_class = 0
        self.verbose = verbose
        self.num_ID = 777
        self.load()

    def load(self,verbose=True):
        sets = set()
        GALLERY_PATH = os.path.join(self.root, 'VeRi', 'gallery.json')
        QUERY_PATH = os.path.join(self.root, 'VeRi', 'query.json')
        TRAIN_PATH = os.path.join(self.root, 'VeRi', 'train.json')
        with open(GALLERY_PATH,'r') as gf, open(QUERY_PATH,'r') as qf,open(TRAIN_PATH,'r') as tf:
            t_lines = json.load(tf)
            g_lines = json.load(gf)
            q_lines = json.load(qf)

            for line in t_lines:
                #imgPath,vid,cid,color,tid,fname = line.strip().split(' ')
                fname = os.path.basename(line['imgPath']).split('.')[0]
                self.train.append([os.path.join(self.root, 'VeRi', line['imgPath']), line['vid'], line['cam'], line['color'], line['type'], fname])

            for line in g_lines:
                fname = os.path.basename(line['imgPath']).split('.')[0]
                self.gallery.append([os.path.join(self.root, 'VeRi', line['imgPath']), int(line['vid']), line['cam'], fname])
                self.test.append([os.path.join(self.root, 'VeRi', line['imgPath']), int(line['vid']), line['cam'], fname])

            for line in q_lines:
                fname = os.path.basename(line['imgPath']).split('.')[0]
                self.query.append([os.path.join(self.root, 'VeRi', line['imgPath']), int(line['vid']), line['cam'], fname])
                self.test.append([os.path.join(self.root, 'VeRi', line['imgPath']), int(line['vid']), line['cam'], fname])

        #self.train = self.train[:1000]
        self.test_num = len(self.test)
        self.train_num = len(self.train)
        self.query_num = len(self.query)
        self.gallery_num = len(self.gallery)
        #print(pseu_ids)
        if self.verbose:
            print(self.__class__.__name__, 'dataset loaded')
            print("  set          | # images  ")
            print("  train        | # {}      ".format(self.train_num))
            print("  gallery      | # {}      ".format(self.gallery_num))
            print("  test         | # {}      ".format(self.test_num))
            print("  query        | # {}      ".format(self.query_num))
            print("  train format | # {}      ".format(self.train[0]))
            print("  test format  | # {}      ".format(self.test[0]))
            print("  ----------------  ")


if __name__ == "__main__":
    d = VeRi()
