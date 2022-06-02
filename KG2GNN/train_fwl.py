import numpy as np
import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from utils.utils_train_fwl import encoding_train
from utils.accuracy import compute_accuracy_for_train
from utils.tool import get_ei2, to_directed, reverse, reduce_loops
from GNN.model import WLGNN, WLGNN_hy
from test0_fwl import tst_params

import optuna
import time

#parse the argument

parser = argparse.ArgumentParser()

parser.add_argument('--dataset',
                    type=str,
                    required=True,
                    help='Name of training dataset')
parser.add_argument('--f',
                    action="store_true",
                    help='Fast sparse matrix multiplication')
parser.add_argument('--epoch',
                    type=int,
                    default=1000,
                    help='Number of epochs to train')
parser.add_argument('--train_times',
                    type=int,
                    default=50,
                    help='Number of times to select params')
parser.add_argument('--test_runs',
                    type=int,
                    default=10,
                    help='Number of runs to test')
parser.add_argument('--lr',
                    type=float,
                    default=0.001,
                    help='Learning rate of the optimizer')
parser.add_argument('--weight_decay',
                    type=float,
                    default=5e-8,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--h1',
                    type=int,
                    default=64,
                    help='Dimension of hidden vectors')
parser.add_argument('--h2',
                    type=int,
                    default=24,
                    help='Dimension of hidden vectors')
parser.add_argument('--dp1', type=float, default=0.0, help='Dropout rate')
parser.add_argument('--dp2', type=float, default=0.0, help='Dropout rate')
parser.add_argument('--path', type=str, default="Opt/")
parser.add_argument('--check', action="store_true", help='Check chosen params')
parser.add_argument('--test', action="store_true", help='Test params')

args = parser.parse_args()


def set_seed(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


np.random.seed(2)
torch.manual_seed(2)
device = torch.device('cuda')

study = optuna.create_study(direction="maximize",
                            storage="sqlite:///" + args.path + args.dataset +
                            "_" + ".db",
                            study_name=args.dataset,
                            load_if_exists=True)

#encoding the data
ei1, ei2, _, features, labels_train, labels_valid, masks_train, masks_valid, pos_pair_mask, num_node, num_type, num_relation = encoding_train(
    args.dataset)
ei1 = torch.cat([ei1, ei1[[1, 0]]], -1)
e2 = get_ei2(num_node, ei1, ei1).to(device)
edge2, edge2_r = reverse(e2, ei1.shape[1] // 2)
edge2 = reduce_loops(edge2)
edge2_r = reduce_loops(edge2_r)
del e2

ei2 = None
edge2 = edge2.to(device)
edge2_r = edge2_r.to(device)
features = to_directed(features)

#labels_train = to_directed(labels_train)
#labels_valid = to_directed(labels_valid)
#masks_train = to_directed(masks_train)
#masks_valid = to_directed(masks_valid)

ei1 = ei1.to(device)
features = features.to(device)
labels_train_gpu = labels_train.to(device)
labels_valid_gpu = labels_valid.to(device)
masks_train_gpu = masks_train.to(device)
masks_valid_gpu = masks_valid.to(device)
pos_pair_mask_gpu = pos_pair_mask.to(device)

pos_pair_mask_gpu = torch.cat([pos_pair_mask_gpu, pos_pair_mask_gpu], 0)
edge_index = ei1[:, pos_pair_mask_gpu].to(torch.long)


def val_params(lr, **kwargs):
    if kwargs["use_hy"]:
        model = WLGNN_hy(n_feat=features.shape[1],
                         nclass=labels_train.shape[1],
                         latent_size_1=kwargs["latent_size_1"],
                         latent_size_2=kwargs["latent_size_2"],
                         depth2=2,
                         dp1=kwargs["dropout"],
                         dp2=kwargs["dropout"],
                         fast=args.f)
    else:
        model = WLGNN(n_feat=features.shape[1],
                      nclass=labels_train.shape[1],
                      latent_size_1=kwargs["dim1"],
                      latent_size_2=kwargs["dim2"],
                      depth2=kwargs["depth"],
                      **kwargs,
                      fast=args.f)

    # set optimizer
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=lr,
                           weight_decay=args.weight_decay)

    version = "lr{}_wd{}_hidden{}{}".format(str(args.lr),
                                            str(args.weight_decay),
                                            str(args.h1), str(args.h2))

    model_path = "models/{}".format(args.dataset)
    # model.load_state_dict(torch.load("{}/lr0.001_wd5e-08_hidden64_e927.pkl".format(model_path)))

    f1_highest = 0
    epoch_best = 0
    score_threshold = 0.5

    def train(epoch):

        t = time.time()
        model.train()
        """train"""

        output_train = model(features, edge2, edge2_r, edge_index, ei1,
                             num_node, pos_pair_mask_gpu)
        output_train_accuracy = output_train.clone()

        # only compute loss for the positive & negative facts
        output_train = torch.mul(output_train, masks_train_gpu)

        # standard binary cross entropy loss
        loss = nn.BCELoss()
        loss_train = loss(output_train, labels_train_gpu)
        loss_train.backward()
        optimizer.step()

        acc_train, precision_train, recall_train, f1_train = compute_accuracy_for_train(
            output_train_accuracy.cpu(), labels_train, masks_train,
            score_threshold, num_type, num_relation)
        """validation"""

        model.eval()
        output_valid = model(features, edge2, edge2_r, edge_index, ei1,
                             num_node, pos_pair_mask_gpu)
        output_valid_accuracy = output_valid.clone()

        # only compute loss for the positive & negative facts
        output_valid = torch.mul(output_valid, masks_valid_gpu)

        loss_valid = loss(output_valid, labels_valid_gpu)

        acc_valid, precision_valid, recall_valid, f1_valid = compute_accuracy_for_train(
            output_valid_accuracy.cpu(), labels_valid, masks_valid,
            score_threshold, num_type, num_relation)

        # report the best epoch according to f1_valid
        nonlocal f1_highest
        nonlocal epoch_best

        if f1_valid > f1_highest:
            f1_highest = f1_valid
            epoch_best = epoch

        if (epoch % 10 == 0):
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

        save_path = "models/{}".format(args.dataset)
        save_model_name = '{}_e{}.pkl'.format(version, epoch)

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # save the model
        torch.save(model.state_dict(), '{}/{}'.format(save_path,
                                                      save_model_name))

    print("Started training model......")

    # train model
    t_train_start = time.time()

    for epoch in range(args.epoch):
        train(epoch)

    print("Finished training.")
    print("Total time for training: {:.4f}s".format(time.time() -
                                                    t_train_start))
    print("Best epoch:", epoch_best)

    model_save = '{}_e{}'.format(version, epoch_best)

    print(model_save)

    return tst_params(**kwargs,
                      lr=lr,
                      fast=args.f,
                      dataset=args.dataset,
                      model_dir=args.dataset,
                      model_name=model_save,
                      num_runs=args.test_runs,
                      printv=False,
                      **kwargs)


def sel_params(trial: optuna.Trial):
    lr = trial.suggest_categorical("lr", [3e-4, 1e-3, 3e-3, 1e-2])
    dp1 = trial.suggest_float("dp1", 0.0, 0.8, step=0.1)
    dp2 = trial.suggest_float("dp2", 0.0, 0.8, step=0.1)
    dp3 = trial.suggest_float("dp3", 0.0, 0.8, step=0.1)
    dim1 = trial.suggest_categorical("dim1", [16, 32, 64, 128, 256])
    dim2 = trial.suggest_categorical("dim2", [16, 32, 64, 128, 256])
    act1 = trial.suggest_categorical("act1", [True, False])
    act2 = trial.suggest_categorical("act2", [True, False])
    act3 = trial.suggest_categorical("act3", [True, False])
    ln1 = trial.suggest_categorical("ln1", [True, False])
    ln2 = trial.suggest_categorical("ln2", [True, False])
    ln3 = trial.suggest_categorical("ln3", [True, False])
    depth = trial.suggest_int("depth", 1, 3)
    return val_params(dim1,
                      dim2,
                      depth,
                      lr,
                      dp1=dp1,
                      dp2=dp2,
                      dp3=dp3,
                      act1=act1,
                      act2=act2,
                      act3=act3,
                      ln1=ln1,
                      ln2=ln2,
                      ln3=ln3)


if args.test:
    params = {
        'GraIL-BM_fb237_v2': {
            'act1': False,
            'act2': True,
            'act3': False,
            'depth': 1,
            'dim1': 128,
            'dim2': 16,
            'dp1': 0.2,
            'dp2': 0.0,
            'dp3': 0.7,
            'ln1': True,
            'ln2': False,
            'ln3': True,
            'lr': 0.0003
        },
        'GraIL-BM_fb237_v1': {
            'act1': False,
            'act2': False,
            'act3': False,
            'depth': 1,
            'dim1': 256,
            'dim2': 64,
            'dp1': 0.4,
            'dp2': 0.1,
            'dp3': 0.5,
            'ln1': False,
            'ln2': True,
            'ln3': True,
            'lr': 0.0003
        },
        'GraIL-BM_WN18RR_v1': {
            'act1': True,
            'act2': False,
            'act3': True,
            'depth': 1,
            'dim1': 128,
            'dim2': 16,
            'dp1': 0.1,
            'dp2': 0.1,
            'dp3': 0.2,
            'ln1': False,
            'ln2': True,
            'ln3': False,
            'lr': 0.001
        },
        'GraIL-BM_WN18RR_v2': {
            'act1': True,
            'act2': False,
            'act3': True,
            'depth': 1,
            'dim1': 32,
            'dim2': 64,
            'dp1': 0.6,
            'dp2': 0.5,
            'dp3': 0.8,
            'ln1': True,
            'ln2': False,
            'ln3': False,
            'lr': 0.001
        },
        'GraIL-BM_WN18RR_v4': {
            'lr': 0.001,
            'dp1': 0.2,
            'dp2': 0.2,
            'dp3': 0.2,
            'dim1': 32,
            "dim2": 32,
            'act1': True,
            'act2': False,
            'act3': True,
            'ln1': True,
            'ln2': True,
            'ln3': False,
            'depth': 1
        },
        'GraIL-BM_WN18RR_v3': {
            'act1': True,
            'act2': False,
            'act3': False,
            'depth': 1,
            'dim1': 128,
            'dim2': 64,
            'dp1': 0.5,
            'dp2': 0.2,
            'dp3': 0.4,
            'ln1': True,
            'ln2': True,
            'ln3': False,
            'lr': 0.0003
        }
    }
    if args.dataset in params:
        param = params[args.dataset]
        param["use_hy"] = False
    else:
        params = {'GraIL-BM_fb237_v1':	{'latent_size_1':96,"latent_size_2":24,"dropout":0.3,	"lr":0.001},
            'GraIL-BM_fb237_v2':	{'latent_size_1':96,"latent_size_2":24,"dropout":0,	"lr":0.001},
            'GraIL-BM_fb237_v3':	{'latent_size_1':96,"latent_size_2":24,"dropout":0.3,	"lr":0.001},
            'GraIL-BM_fb237_v4':	{'latent_size_1':96,"latent_size_2":24,"dropout":0.2,	"lr":0.0005},	
            'GraIL-BM_WN18RR_v1':	{'latent_size_1':64,"latent_size_2":24,"dropout":0.5,	"lr":0.0005},
            'GraIL-BM_WN18RR_v2':	{'latent_size_1':96,"latent_size_2":24,"dropout":0.1,	"lr":0.005},
            'GraIL-BM_WN18RR_v3':	{'latent_size_1':96,"latent_size_2":24,"dropout":0.3,	"lr":0.001},
            'GraIL-BM_WN18RR_v4':	{'latent_size_1':96,"latent_size_2":24,"dropout":0, "lr":0.001}}
        param = params[args.dataset]
        param["use_hy"] = True
    for i in range(10):
        set_seed(i)
        val_params(**param)
    exit()

print(
    f"storage {'sqlite:///' + args.path + args.dataset + '.db'} study_name {args.dataset}"
)
study.optimize(sel_params, n_trials=args.train_times)
