import os
import os.path as osp
import csv
import pandas as pd
import random
import json
ROOT='/home/shaohao/CVPR20/datas/VeRi_Wild/'

class VeRi_Wild(object):
    def __init__(self,root=ROOT,test_size=3000,verbose=True):
        assert test_size in [3000, 5000, 10000]
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
        self.train_path = os.path.join(root, "train_list.json")
        self.gallery_path = os.path.join(root, "test_{}.json".format(str(test_size)))
        self.query_path = os.path.join(root, "test_{}_query.json".format(str(test_size)))
        self.load()

    def load(self,verbose=True):
        with open(self.gallery_path,'r') as gf, open(self.query_path,'r') as qf,open(self.train_path,'r') as tf:
            g_lines = json.load(gf)
            t_lines = json.load(tf)
            q_lines = json.load(qf)
            for line in t_lines:
                fname = str(line['vid']) + '_' + os.path.basename(line['imgPath']).split('.')[0] 
                self.train.append([line['imgPath'], int(line['vid']), line['cam'], line['model'], line['color'], line['type'], fname])
            for line in g_lines:
                fname = str(line['vid']) + '_' + os.path.basename(line['imgPath']).split('.')[0]
                self.gallery.append([line['imgPath'], int(line['vid']), line['cam'],  line['model'], line['color'], line['type'], fname])
                self.test.append([line['imgPath'], int(line['vid']), line['cam'],  line['model'],line['color'], line['type'], fname])
            for line in q_lines:
                fname = str(line['vid']) + '_' + os.path.basename(line['imgPath']).split('.')[0]
                self.query.append([line['imgPath'], int(line['vid']), line['cam'],  line['model'],line['color'], line['type'], fname])
                self.test.append([line['imgPath'], int(line['vid']),  line['cam'],  line['model'],line['color'], line['type'], fname])
        self.test_num = len(self.test)
        self.train_num = len(self.train)
        self.query_num = len(self.query)
        self.gallery_num = len(self.gallery)
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

class VeRi_Wild_ALL(object):
    '''
    def __init__(self, root=ROOT, verbose=True):
        self.root = root
    '''
    def __init__(self,root=ROOT,test_size=3000,verbose=True):
        assert test_size in [3000, 5000, 10000]
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
        self.train_path = os.path.join(root, "train_list.json")
        self.gallery_path = os.path.join(root, "test_{}.json".format(str(test_size)))
        self.query_path = os.path.join(root, "test_{}_query.json".format(str(test_size)))
        self.load()

    def load(self,verbose=True):
        with open(os.path.join(self.root, 'train_test_split', 'train_list.txt'), 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            path = os.path.join(self.root, 'images', '%s.jpg'%line)
            self.train.append([path])
            
        with open(self.gallery_path,'r') as gf, open(self.query_path,'r') as qf,open(self.train_path,'r') as tf:
            g_lines = json.load(gf)
            t_lines = json.load(tf)
            q_lines = json.load(qf)
            for line in g_lines:
                fname = str(line['vid']) + '_' + os.path.basename(line['imgPath']).split('.')[0]
                self.gallery.append([line['imgPath'], int(line['vid']), line['cam'],  line['model'], line['color'], line['type'], fname])
                self.test.append([line['imgPath'], int(line['vid']), line['cam'],  line['model'],line['color'], line['type'], fname])
            for line in q_lines:
                fname = str(line['vid']) + '_' + os.path.basename(line['imgPath']).split('.')[0]
                self.query.append([line['imgPath'], int(line['vid']), line['cam'],  line['model'],line['color'], line['type'], fname])
                self.test.append([line['imgPath'], int(line['vid']),  line['cam'],  line['model'],line['color'], line['type'], fname])
        self.test_num = len(self.test)
        self.train_num = len(self.train)
        self.query_num = len(self.query)
        self.gallery_num = len(self.gallery)
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
    a = VeRi_Wild_ALL()
