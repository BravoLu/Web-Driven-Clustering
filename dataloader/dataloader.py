# -*- coding: utf-8 -*-
# @Author: Lu Shaohao(Bravo)
# @Date:   2019-08-21 10:05:53
# @Last Modified by:   Lu Shaohao(Bravo)
# @Last Modified time: 2019-11-06 10:48:40
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .sampler import RandomIdentitySampler
from .vehicleid import VehicleID
from .veri import VeRi
from .veri_wild import VeRi_Wild, VeRi_Wild_ALL
from .websearchvehicle import *
from .preprocessor import *
from .transform import RandomErasing
import torch


def get_vehicle_dataloader(cfg, quick_check=False):

    source = globals()[cfg['SOURCE']]()
    target = globals()[cfg['TARGET']]()
    if quick_check:
        source_train = source.train[:1000]
        target_train = target.train[:1000]
    else:
        source_train = source.train
        target_train = target.train

    target_test = target.test
    query = target.query
    gallery = target.gallery
    num_gpus = torch.cuda.device_count()
    normalizer = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    train_transformer = T.Compose([
        T.Resize((cfg['WIDTH'], cfg['HEIGHT'])),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(0),
        T.RandomCrop((cfg['WIDTH'],cfg['HEIGHT'])),
        T.ToTensor(),
        normalizer,
        RandomErasing(),
    ])

    test_transformer = T.Compose([
        T.Resize((cfg['WIDTH'], cfg['HEIGHT'])),
        T.ToTensor(),
        normalizer,
    ])
    source_loader = DataLoader(
        Preprocessor(source_train, name=cfg['SOURCE'], training=True, transform=train_transformer),
        sampler=RandomIdentitySampler(source_train, cfg['BATCHSIZE'], cfg['INSTANCE'], cfg['SOURCE']),
        batch_size=cfg['BATCHSIZE'],
        num_workers=4,
        pin_memory=True,
    )
    target_loader = DataLoader(
        Preprocessor(target_train, name=cfg['TARGET'], training=True, transform=train_transformer),
        batch_size=cfg['BATCHSIZE'],
        num_workers=4,
        #shuffle=True,
        sampler=RandomIdentitySampler(target_train, cfg['BATCHSIZE'], cfg['INSTANCE'], cfg['TARGET']),
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        Preprocessor(target_test, name=cfg['TARGET'], training=False, transform=test_transformer),
        batch_size=cfg['BATCHSIZE'],
        num_workers=4,
        shuffle=False,
        pin_memory=True,
    )
    target_cluster_loader = DataLoader(
        Preprocessor(target_train, name=cfg['TARGET'], training=True, transform=test_transformer),
        batch_size=cfg['BATCHSIZE'],
        num_workers=4,
        shuffle=False,
        pin_memory=True,
    )

    return source_loader, target_loader, test_loader, query, gallery, train_transformer, source_train, target_train, target_cluster_loader




