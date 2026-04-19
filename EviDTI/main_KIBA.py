import argparse
from time import time
import pickle
import sys
import timeit
import argparse

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from typing import List, Tuple
from torch_geometric.loader import DataLoader

from d_2D_cnn_model import LightAttention
from kiba_solver import Solver
from function import bond_angle_graph_data,KIBA_graph_data
from drugbank_configs import get_cfg_defaults
import os
import yaml

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description="EviDTI for dti prediction")
parser.add_argument('--cfg', required=True, help="path to config file", type=str)
parser.add_argument('--data', required=True, type=str, metavar='TASK',
                    help='dataset')
parser.add_argument('--split', default='random', type=str, metavar='S')
args = parser.parse_args()

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2

def main():
    # 从config.py读取config
    cfg = get_cfg_defaults()
    # 与yaml文件合并
    cfg.merge_from_file(args.cfg)
    print(f"Config yaml: {args.cfg}")
    print(f"Hyperparameters: {dict(cfg)}")
    print(f"Running on: {device}", end="\n\n")

    data_list = KIBA_graph_data('./dataset/KIBA/',
                                d_2D_path='dataset/KIBA/KIBA_d_2d_feature.npy',
                                t_1D_path = 'dataset/KIBA/KIBA_t_1D_feature.npy',
                                d_3D_path = 'dataset/KIBA/KIBA_d_3d_feature.npy',
                                label_file='dataset/KIBA/KIBA_total_cid_unid.csv',
                                metadata_path='dataset/KIBA/KIBA_total_cid_unid.csv')

    np.random.seed(1)

    shuffled_indices = np.random.permutation(len(data_list))
    train_idx = shuffled_indices[:int(0.8 * len(data_list))]
    val_idx = shuffled_indices[int(0.8 * len(data_list)):int(0.9 * len(data_list))]
    test_idx = shuffled_indices[int(0.9 * len(data_list)):]
    train_loader = DataLoader(data_list, batch_size = cfg.SOLVER.BATCH_SIZE, drop_last = True,
                              sampler = SubsetRandomSampler(train_idx))
    val_loader = DataLoader(data_list, batch_size = cfg.SOLVER.BATCH_SIZE, drop_last=False,
                            sampler=SubsetRandomSampler(val_idx))
    test_loader = DataLoader(data_list, batch_size = cfg.SOLVER.BATCH_SIZE, drop_last=False,
                            sampler=SubsetRandomSampler(test_idx))
    print('load data finish')
    model = LightAttention(cfg)

    # Needs "from torch.optim import *" and "from models import *" to work
    solver = Solver(model, cfg, device)
    solver.train(train_loader, val_loader,eval_data=test_loader)

if __name__ == '__main__':
    s = time()
    result = main()
    e = time()
    print(f"Total running time: {round(e - s, 2)}s")
