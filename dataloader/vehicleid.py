import os
import numpy as np
import os.path as osp
import random
from collections import OrderedDict
import json

ROOT = '/home/share/zhihui/VehicleID_V1.0/'
TRAIN_PATH='/home/shaohao/CVPR20/datas/VehicleID/data.json'

class VehicleID(object):
    def __init__(self, test_size=800,verbose=True):
        self.root = ROOT

        self.json_path = TRAIN_PATH

        assert test_size in [800, 1600, 2400, 3200, 6000, 13164]
        self.tpath = osp.join(self.root,'train_test_split', 'test_list_{}.txt'.format(str(test_size)))
        self.train, self.query, self.gallery, self.test = [], [], [], []
        self.train_num = 0
        self.gallery_num = 0
        self.query_num = 0
        self.load(verbose=verbose)

    def load(self, verbose=True):
        with open(os.path.join(self.root,self.json_path),'r') as train_file:
            train_list = json.load(train_file)
        for item in train_list:
            path = os.path.join(ROOT, 'image', item['filename'])
            self.train.append([path, item['vid'], item['camera'], item['model'], item['color']])

        #self.train = self.train[:1000]

        self.train_num = len(train_list)
        test_ID = []
        with open(self.tpath, 'r') as test_file:
            line = test_file.readlines()
            test_ID = [w.strip().split(' ')[1] for w in line]
        test_ID = sorted(set(test_ID))

        new_ID = {}
        for new,origin in enumerate(test_ID):
            new_ID[origin] = new

        with open(self.tpath, 'r') as test_file:
            test_data_lines = test_file.readlines()
            self.test_num = len(test_data_lines)
            test_imgName_list = [w.strip().split(' ')[0] for w in test_data_lines]
            test_imgPath_list = [osp.join(self.root,'image',w+'.jpg') for w in test_imgName_list]
            test_vehicleIDs_list = [new_ID[w.strip().split(' ')[-1]] for w in test_data_lines]
            dic_test_vehicleID_imgName = {}
            dic_test_imgName_vehicleID = {}

            for imgName, vehicleID in zip(test_imgName_list, test_vehicleIDs_list):
                dic_test_vehicleID_imgName.setdefault(vehicleID, []).append(imgName)
                dic_test_imgName_vehicleID[imgName] = vehicleID
        #query_imgNames = []
        #gallery_imgNames = []

        for vehicleID in dic_test_vehicleID_imgName:
            imgNames = dic_test_vehicleID_imgName[vehicleID]
            sampled_idx = random.randint(0,len(imgNames)-1)
            sampled_imgName = imgNames[sampled_idx]
            imgPath = osp.join(self.root,'image', sampled_imgName+'.jpg')
            self.gallery.append([imgPath, vehicleID, sampled_imgName])
            self.test.append([imgPath, vehicleID, sampled_imgName])
            #gallery_imgNames.append(sampled_imgName)
            imgNames.remove(sampled_imgName)
            #query_imgNames += imgNames
            for imgName in imgNames:
                imgPath = osp.join(self.root, 'image',  imgName+'.jpg')
                self.query.append([imgPath, vehicleID, imgName])
                self.test.append([imgPath,  vehicleID, imgName])
        self.gallery_num = len(self.gallery)
        self.query_num   = len(self.query)
        self.test_num = len(self.test)
        if verbose:
            print(self.__class__.__name__,"dataset loaded")
            print("  set          | # images  ")
            print("  train        | # {}      ".format(self.train_num))
            print("  test         | # {}      ".format(self.test_num))
            print("  gallery      | # {}      ".format(self.gallery_num))
            print("  query        | # {}      ".format(self.query_num))
            print("  train format | # {}      ".format(self.train[0]))
            print("  test  format | # {}      ".format(self.test[0]))
            print("  ----------------  ")

if __name__ == "__main__":
    d = VehicleID()
