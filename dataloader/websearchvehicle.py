import os
import json
import random
from collections import OrderedDict

class WebSearchVehicle(object):
    def __init__(self, root='datas/WebSearchVehicle'):
        self.train = []
        self.ids = 749
        with open(os.path.join(root, 'info.json'), 'r') as f:
            train = json.load(f)

        for item in train:
            imgPath = os.path.join(root, item['imgPath'])
            self.train.append([imgPath, item['typeID']])

        print(" WebSearchVehicle dataset information:")
        print("---------------------------------------------------")
        print("  subset  | # images | # ids |")
        print("  train   | {:8d} | {:7d} |".format(len(self.train), self.ids))
        print(self.train[0])
        print("---------------------------------------------------")
