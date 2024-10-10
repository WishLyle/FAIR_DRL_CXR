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
import os

print("yes,Here's main go!")


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
parser.add_argument('--disease', default='Cardiomegaly', help='disease')
parser.add_argument('--epochs', type=int, default=0, help='disease')
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

parser.add_argument('--exp_name', default='test-0725-vanilla-MLP')
parser.add_argument('--tensorboard', type=int, default=1)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--ema_alpha', type=float, default=0.7)
parser.add_argument('--swap_epoch', type=int, default=3)
parser.add_argument('--class_epoch', type=int, default=0)
parser.add_argument('--lambda_d', type=float, default=0.7)
parser.add_argument('--lambda_r', type=float, default=0.7)
parser.add_argument('--mile_d', default=[2])
parser.add_argument('--mile_r', default=[3])
parser.add_argument('--seed', type=int, default=2023921)
parser.add_argument('--dataframe_path', default=r'./original/unbias_test')
parser.add_argument('--pretrain', default=1, help='0 for not pretrain ,1 for pretrain')
parser.add_argument('--lock', type=int, default=0, help="0 for not lock, 1 for lock ,2..")
args = parser.parse_args()

setup_seed(args.seed)
L = learner(args)
date_string = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
if args.mode == 'vanilla':
    path_dir = '/lxw/0607/checkpoints/vanilla/' + L.args.model + '/'
    path_namess = os.listdir(path_dir)
    for path_names in path_namess:
        path_name = os.path.join(path_dir, path_names)

        L.write_result("Time, [ {} ]  ".format(date_string))
        L.write_result(
            "Backbone,{},mode,{},Exp_Name,{}\nDisease,{}".format(L.args.model, L.args.mode, str(path_names),
                                                                 L.args.disease))
        L.write_result(',Acc,Auc,F1,Recall,Precision,TP,FN,FP,TN')
        L.test_basic(path_name, 'w')
        L.test_basic(path_name, 'o')
        L.e_a = (L.o_a + L.w_a) / 2.0
        L.test_basic(path_name, 'l')
        L.write_result('-\n')


elif args.mode == 'debias':
    path_dir = '/lxw/0607/checkpoints/debias/' + L.args.model + '/'
    path_namess = os.listdir(path_dir)
    for path_names in path_namess:
        if(str(args.disease) in path_names and '-2' in path_names and "Cardiomegaly_-2" in path_names ):
            path_name = os.path.join(path_dir, path_names)

            L.write_result("Time, [ {} ]  ".format(date_string))
            L.write_result(
                "Backbone,{},mode,{},Exp_Name,{}\nDisease,{}".format(L.args.model, L.args.mode, str(path_names),
                                                                     L.args.disease))
            L.write_result(',Acc,Auc,F1,Recall,Precision,TP,FN,FP,TN')
            L.test_debias(path_name, 'w')
            L.test_debias(path_name, 'o')
            L.e_a = (L.o_a + L.w_a) / 2.0
            L.test_debias(path_name, 'l')
            L.write_result('-\n')
# elif args.model == 3:
#     L.train_race()
# else:
#     L.train_debias_lock()
