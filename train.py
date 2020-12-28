import torch
import os
import argparse
import torch.nn as nn
from torch.backends import cudnn

from utils import *
from dataloader import *
from torch.backends import cudnn
from models import *
from trainers import *
from evaluators import Evaluator

def parse_args():
    parser = argparse.ArgumentParser('Unsupervised Vehicle ReID')
    parser.add_argument('-c', '--config', type=str,
                                  help='the path to the training config', default='baseline')
    parser.add_argument('-t', '--test', action='store_true', default=False)
    parser.add_argument('-s', '--check', action='store_true', default=False, help="for fast check complie error in program")
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--ckpt', type=str, default='')
    args = parser.parse_args()
    print(args)
    return args

def train(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    cfg = load_configs(args.config)
    dataloader = get_vehicle_dataloader(cfg, quick_check=args.check)
    model = globals()[cfg['MODEL']](num_id=749)
    trainer = globals()[cfg['TRAINER']](cfg, model, dataloader)
    trainer.train()

def test(args):

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    cfg = load_configs(args.config)
    dataset = get_vehicle_dataloader(cfg, quick_check=args.check)
    model = globals()[cfg['MODEL']]()
    model.load_param(args.ckpt)
    model = nn.DataParallel(model).cuda()
    evaluator = Evaluator(cfg, model, dataset)
    mAP, cmc = evaluator.evaluate()
    cmc1, cmc5, cmc10 = cmc[0], cmc[4], cmc[9]
    print("(mAP: {:.5f} cmc-1: {:.5f} cmc-5: {:.5f} cmc-10: {:.5f})".format(mAP, cmc1, cmc5, cmc10))

def main():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    cudnn.benchmark = True
    args = parse_args()
    if (args.test):
        test(args)
    else:
        train(args)

if __name__ == "__main__":
    main()
