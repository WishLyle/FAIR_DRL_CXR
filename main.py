# This is main for my 3rd version python code

import argparse
import sys
import torch
import random
from torch.utils.data import DataLoader
import datetime
from torch.backends import cudnn
import numpy as np
import random as python_random
from learner import learner

print("yes,Here's main go!")


# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


parser = argparse.ArgumentParser(description='test')
parser.add_argument('--mode', default='debias', help='vanilla, debias, others')
parser.add_argument('--model', default='DenseNet', help='ResNet, MLP, DenseNet')
parser.add_argument('--disease', default='Pneumothorax', help='disease')
parser.add_argument('--epochs', type=int, default=5, help='disease')
parser.add_argument('--data_path', default=r'/u2/lxw/mimic-cxr-jpg/2.0.0/files/',
                    help=r'eg:..\files')
parser.add_argument('--train_mode', default='all',
                    help=r'all: all races data for training . white: white race for training. black.asian.')
parser.add_argument('--batch_size', type=int, default=64, help=r'batch_size')
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument("--lr", help='learning rate', default=0.0001, type=float)
parser.add_argument("--weight_decay", help='weight_decay', default=0, type=float)
parser.add_argument("--use_lr_decay", action='store_true', help="whether to use learning rate decay")
parser.add_argument("--lambda_dis_align", help="lambda_dis in Eq.2", type=float, default=1.0)
parser.add_argument("--lambda_swap_align", help="lambda_swap_b in Eq.3", type=float, default=1.0)
parser.add_argument("--lambda_swap", help="lambda swap (lambda_swap in Eq.4)", type=float, default=1.0)

parser.add_argument('--exp_name', default='debug-debias')
parser.add_argument('--tensorboard', type=int, default=1)
parser.add_argument('--device', type=int, default=4)
parser.add_argument('--ema_alpha', type=float, default=0.7)
parser.add_argument('--swap_epoch', type=int, default=3)
parser.add_argument('--class_epoch', type=int, default=0)
parser.add_argument('--lambda_d', type=float, default=0.7)
parser.add_argument('--lambda_r', type=float, default=0.7)
parser.add_argument('--mile_d', default=[2])
parser.add_argument('--mile_r', default=[3])
parser.add_argument('--seed', type=int, default=2023921)
parser.add_argument('--dataframe_path', default=r'./original/unbias_test/')
parser.add_argument('--pretrain',default=1, help = '0 for not pretrain ,1 for pretrain')
parser.add_argument('--lock', type=int, default=0, help="0 for not lock, 1 for lock ,2..")
args = parser.parse_args()

setup_seed(args.seed)
L = learner(args)

if args.mode == 'vanilla':
    L.train_basic()
elif args.mode == 'debias':
    L.train_debias()
# elif args.model == 3:
#     L.train_race()
# else:
#     L.train_debias_lock()
