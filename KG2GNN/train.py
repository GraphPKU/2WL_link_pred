import numpy as np
import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from utils.utils_train import encoding_train
from utils.accuracy import compute_accuracy_for_train
from utils.tool import get_ei2, to_directed, reverse, reduce_loops
from GNN.model import WLGNN_d, WLGNN_d_hy
from test0 import tst_params

import optuna
import time


#parse the argument

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, required=True,
                    help='Name of training dataset')
parser.add_argument('--directed', action="store_true",
                    help='Encode direct 2-order graph')
parser.add_argument('--epoch', type=int, default=1000,
                    help='Number of epochs to train')
parser.add_argument('--train_times', type=int, default=50,
                    help='Number of times to select params')
parser.add_argument('--test_runs', type=int, default=10,
                    help='Number of runs to test')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Learning rate of the optimizer')
parser.add_argument('--weight_decay', type=float, default=5e-8,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Dimension of hidden vectors')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='Dropout rate')
parser.add_argument('--path', type=str, default="Opt/")
parser.add_argument('--check', action="store_true",
                    help='Check chosen params')
parser.add_argument('--test', action="store_true",
                    help='Test params')

args = parser.parse_args()

np.random.seed(2)
torch.manual_seed(2)
device = torch.device('cuda')
model_ver = 'dir' if args.directed else 'undir'

study = optuna.create_study(direction="maximize",
                            storage="sqlite:///" + args.path + args.dataset + "_" + model_ver +
                            ".db",
                            study_name=args.dataset,
                            load_if_exists=True)

if args.check:
    import pandas as pd

    pd.set_option('display.max_rows', None)
    df = study.trials_dataframe().drop(['state', 'datetime_start', 'datetime_complete', 'duration', 'number'],
                                       axis=1)
    import pdb
    pdb.set_trace()

#encoding the data
ei1, ei2, _, features, labels_train, labels_valid, masks_train, masks_valid, num_node, num_type, num_relation = encoding_train(args.dataset)
ei1 = torch.cat([ei1, ei1[[1, 0]]], -1)
e2 = get_ei2(num_node, ei1, ei1).to(device)
edge2, edge2_r = reverse(e2, ei1.shape[1]//2)
edge2 = reduce_loops(edge2)
edge2_r = reduce_loops(edge2_r)
del e2

if args.directed:
    ei2 = None
    edge2 = edge2.to(device)
    edge2_r = edge2_r.to(device)
    features = to_directed(features)
else:
    edge2 = None
    edge2_r = None
    ei2 = ei2.to(device)

#labels_train = to_directed(labels_train)
#labels_valid = to_directed(labels_valid)
#masks_train = to_directed(masks_train)
#masks_valid = to_directed(masks_valid)

features = features.to(device)
labels_train_gpu = labels_train.to(device)
labels_valid_gpu = labels_valid.to(device)
masks_train_gpu = masks_train.to(device)
masks_valid_gpu = masks_valid.to(device)
import pdb
#pdb.set_trace()
def val_params(**kwargs):
    # define the Model
    if kwargs["use_hy"]:
        model = WLGNN_d_hy(n_feat=features.shape[1],
                        latent_size=kwargs["latent_size"],
                        depth2=2,
                        dropout=kwargs["dropout"],
                        nclass=labels_train.shape[1],
                        directed=args.directed)
    else:
        model = WLGNN_d(n_feat=features.shape[1],
                        dim=kwargs["dim"],
                        depth2=1,
                        ln1=kwargs["ln"],
                        dp1=kwargs["dp1"],
                        act1=kwargs["act"],
                        nclass=labels_train.shape[1],
                        directed=args.directed)

    # set optimizer
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=kwargs["lr"], weight_decay=args.weight_decay)

    version = "lr{}_wd{}_hidden{}".format(str(args.lr), str(args.weight_decay), str(args.hidden))

    model_path = "models/{}".format(args.dataset)
    # model.load_state_dict(torch.load("{}/lr0.001_wd5e-08_hidden64_e927.pkl".format(model_path)))

    f1_highest = 0
    epoch_best = 0
    score_threshold = 0.5
    best_acc = 0

    def train(epoch):

        t = time.time()
        model.train()

        """train"""
        # import pdb
        # pdb.set_trace()

        output_train = model(features, edge2, edge2_r, ei2)
        output_train_accuracy = output_train.clone()

        # only compute loss for the positive & negative facts
        output_train = torch.mul(output_train, masks_train_gpu)

        # standard binary cross entropy loss
        loss = nn.BCELoss()
        loss_train = loss(output_train, labels_train_gpu)
        loss_train.backward()
        optimizer.step()

        acc_train, precision_train, recall_train, f1_train = compute_accuracy_for_train(
            output_train_accuracy.cpu(), labels_train, masks_train, score_threshold, num_type, num_relation)

        """validation"""

        model.eval()
        output_valid = model(features, edge2, edge2_r, ei2)
        output_valid_accuracy = output_valid.clone()

        # only compute loss for the positive & negative facts
        output_valid = torch.mul(output_valid, masks_valid_gpu)

        loss_valid = loss(output_valid, labels_valid_gpu)

        acc_valid, precision_valid, recall_valid, f1_valid = compute_accuracy_for_train(
            output_valid_accuracy.cpu(), labels_valid, masks_valid, score_threshold, num_type, num_relation)

        # report the best epoch according to f1_valid
        nonlocal f1_highest, best_acc
        nonlocal epoch_best
        if best_acc < acc_valid:
            best_acc = acc_valid
        if f1_valid > f1_highest:
            f1_highest = f1_valid
            epoch_best = epoch

        print('Epoch: {:04d}'.format(epoch),
              'loss_train: {:.4f}'.format(loss_train.item() * 1000),
              'precision_train: {:.4f}'.format(precision_train.item()),
              'recall_train: {:.4f}'.format(recall_train.item()),
              'F1_train: {:.4f}'.format(f1_train.item()),
              'loss_val: {:.4f}'.format(loss_valid.item() * 1000),
              'precision_val: {:.4f}'.format(precision_valid.item()),
              'recall_val: {:.4f}'.format(recall_valid.item()),
              'F1_val: {:.4f}'.format(f1_valid.item()),
              'time: {:.4f}s'.format(time.time() - t))

        save_path = "models/{}/{}".format(model_ver, args.dataset)
        save_model_name = '{}_e{}.pkl'.format(version, epoch)

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # save the model
        torch.save(model.state_dict(), '{}/{}'.format(save_path, save_model_name))

    print("Started training model......")

    # train model
    t_train_start = time.time()

    for epoch in range(args.epoch):
        train(epoch)

    print("Finished training.")
    print("Total time for training: {:.4f}s".format(time.time() - t_train_start))
    print("Best epoch:", epoch_best)

    model_save = '{}_e{}'.format(version, epoch_best)

    print(model_save)

    return tst_params(directed=args.directed, dataset=args.dataset, model_dir=args.dataset, model_name=model_save, num_runs=args.test_runs, printv=False, **kwargs)


def sel_params(trial: optuna.Trial):
    lr = trial.suggest_categorical("lr",[3e-3, 1e-3, 3e-3, 1e-2])
    dim = trial.suggest_categorical("dim", [16, 32, 64, 128])
    dp = trial.suggest_float("dp1", 0.0, 0.7, step=0.05)
    ln = trial.suggest_categorical("ln", [True, False])
    act = trial.suggest_categorical("act", [True, False])
    # dp2 = trial.suggest_float("dp2", 0.0, 0.5, step=0.1)
    # dp3 = trial.suggest_float("dp3", 0.0, 0.5, step=0.1)
    return val_params(dim, dp, ln, act, lr)

if args.test:
    param = {}
    if args.dataset not in ['GraIL-BM_fb237_v3', 'GraIL-BM_fb237_v4']:
        params = {'GraIL-BM_fb237_v1': {'act': True, 'dim': 64, 'dp1': 0.35, 'ln': True, 'lr': 0.01} ,
        'GraIL-BM_fb237_v2': {'act': True, 'dim': 128, 'dp1': 0.3, 'ln': True, 'lr': 0.003} ,
        'GraIL-BM_fb237_v3': {'act': False, 'dim': 32, 'dp1': 0.15, 'ln': True, 'lr': 0.001} ,
        'GraIL-BM_fb237_v4': {'act': True, 'dim': 128, 'dp1': 0.55, 'ln': True, 'lr': 0.003} ,
        'GraIL-BM_WN18RR_v1': {'act': True, 'dim': 32, 'dp1': 0.3, 'ln': True, 'lr': 0.003} ,
        'GraIL-BM_WN18RR_v2': {'act': True, 'dim': 64, 'dp1': 0.45, 'ln': True, 'lr': 0.001} ,
        'GraIL-BM_WN18RR_v3': {'act': False, 'dim': 16, 'dp1': 0.65, 'ln': True, 'lr': 0.003} ,
        'GraIL-BM_WN18RR_v4': {'act': True, 'dim': 128, 'dp1': 0.0, 'ln': True, 'lr': 0.003}}
        param = params[args.dataset]
        param["use_hy"] = False
    else:
        params = {'GraIL-BM_fb237_v1':	{'latent_size':96,	"dropout":0.3,	"lr":0.001},
            'GraIL-BM_fb237_v2':	{'latent_size':96,	"dropout":0,	"lr":0.001},
            'GraIL-BM_fb237_v3':	{'latent_size':96,	"dropout":0.3,	"lr":0.001},
            'GraIL-BM_fb237_v4':	{'latent_size':96,	"dropout":0.2,	"lr":0.0005},	
            'GraIL-BM_WN18RR_v1':	{'latent_size':64,	"dropout":0.5,	"lr":0.0005},
            'GraIL-BM_WN18RR_v2':	{'latent_size':96,	"dropout":0.1,	"lr":0.005},
            'GraIL-BM_WN18RR_v3':	{'latent_size':96,	"dropout":0.3,	"lr":0.001},
            'GraIL-BM_WN18RR_v4':	{'latent_size':96,	"dropout":0, "lr":0.001}}
        param = params[args.dataset]
        param["use_hy"] = True
    for i in range(10):
        val_params(**param)
    exit()


print(f"storage {'sqlite:///' + args.path + args.dataset + '.db'} study_name {args.dataset}")
study.optimize(sel_params, n_trials=args.train_times)
