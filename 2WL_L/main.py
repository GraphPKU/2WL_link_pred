from math import e
from scipy.sparse import data
from sklearn import utils
import random
import numpy as np
from model import WLGNN, Model_HY
from datasets import load_dataset, dataset
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import roc_auc_score
import time
from ogb.linkproppred import Evaluator
from utils import sample_block, double
from tqdm import tqdm
import optuna

import warnings
warnings.filterwarnings("ignore")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def evaluate_hits(pos_pred, neg_pred, K):
    results = {}
    evaluator = Evaluator(name='ogbl-collab')
    evaluator.K = K
    hits = evaluator.eval({
        'y_pred_pos': pos_pred,
        'y_pred_neg': neg_pred,
    })[f'hits@{K}']

    results[f'Hits@{K}'] = hits

    return results

def reverse(edge_index):
    tem0 = 1 - (edge_index[0] > edge_index[0] // 2 * 2).to(torch.long) * 2
    tem1 = 1 - (edge_index[1] > edge_index[1] // 2 * 2).to(torch.long) * 2
    edge = torch.cat([(edge_index[0] + tem0).unsqueeze(0), edge_index[1].unsqueeze(0)])
    edge_r = torch.cat([edge_index[0].unsqueeze(0), (edge_index[1] + tem1).unsqueeze(0)])
    #return edge_index
    return edge, edge_r

def train(mod, opt, dataset, batch_size):
    perm1 = torch.randperm(dataset.ei.shape[1] // 2, device=dataset.x.device)
    perm2 = torch.randperm((dataset.pos1.shape[0] - dataset.ei.shape[1]) // 2,
                           device=dataset.x.device)
    out = []
    scores = []
    mod.train()
    pos_batchsize = batch_size // 2
    neg_batchsize = batch_size // 2
    # neg_batchsize = (dataset.pos1.shape[0] - dataset.ei.shape[1])//(dataset.ei.shape[1]//pos_batchsize)

    #for i in range(perm1.shape[0] // pos_batchsize):
    for i in range(1):
        idx1 = perm1[i * pos_batchsize:(i + 1) * pos_batchsize]
        idx2 = perm2[i * neg_batchsize:(i + 1) * neg_batchsize]
        y = torch.cat((torch.ones_like(idx1, dtype=torch.float),
                       torch.zeros_like(idx2, dtype=torch.float)),
                      dim=0).unsqueeze(-1)

        idx1 = double(idx1, for_index=True)
        idx2 = double(idx2, for_index=True)
        length = idx1.shape[0] + idx2.shape[0]
        # import pdb
        # pdb.set_trace()

        new_ei, new_x, pos_pos, new_ei2 = sample_block(idx1, dataset.ea, dataset.x.shape[0], dataset.ei, dataset.ei2)
        opt.zero_grad()
        pos2 = torch.cat((idx1, dataset.ei.shape[1] + idx2), dim=0)
        ei2, ei2_r = reverse(new_ei2)
        #ei2 = new_ei2
        pred = mod(new_x, dataset.na, new_ei, ei2, ei2_r, dataset.pos1, pos2)
        loss = F.binary_cross_entropy_with_logits(pred, y)
        loss.backward()
        opt.step()
        out.append(loss.item())
        with torch.no_grad():
            sig = pred.sigmoid().cpu().numpy()
            score = roc_auc_score(y.cpu().numpy(), sig)
        scores.append(score)
    print(f"trn score {sum(scores) / len(scores)}", end=" ")
    return sum(out) / len(out)


@torch.no_grad()
def test(mod, dataset, test=False):
    t = time.time()
    mod.eval()
    mask = torch.cat(
        [torch.ones([1, dataset.y.shape[0] // 2], dtype=bool), torch.zeros([1, dataset.y.shape[0] // 2], dtype=bool)]).t().reshape(-1)
    import pdb
    ei2, ei2_r = reverse(dataset.ei2)
    #ei2 = dataset.ei2
    
    pred = mod(
        dataset.x, dataset.na, dataset.ei, ei2, ei2_r, dataset.pos1, dataset.ei.shape[1] +
                                                                      torch.arange(dataset.y.shape[0],
                                                                                   device=dataset.x.device), True)
    sig = pred.sigmoid().cpu()
    mask = torch.cat(
        [torch.ones([1, sig.shape[0]], dtype=bool), torch.zeros([1, sig.shape[0]], dtype=bool)]).t().reshape(
        -1, 1)
    if True:
        result = roc_auc_score(dataset.y[mask].squeeze().cpu().numpy(), sig)
    y = dataset.y[mask].to(torch.bool)
    if test:
        print(time.time()-t)
    if False:
        result = evaluate_hits(sig[y].squeeze(), sig[~y].squeeze(), K=50)['Hits@50']
    return result


def main(device="cpu", dsname="Celegans", mod_params=(32, 2, 1, 0.0), lr=3e-4):
    device = torch.device(device)
    bg = load_dataset(dsname)
    bg.to(device)
    bg.preprocess()
    bg.setPosDegreeFeature()
    mod = WLGNN(torch.max(bg.x[2]), *mod_params).to(device)
    opt = Adam(mod.parameters(), lr=lr)
    trn_ds = dataset(*bg.split(0))
    val_ds = dataset(*bg.split(1))
    tst_ds = dataset(*bg.split(2))
    train_routine(mod, opt, trn_ds, val_ds, tst_ds, verbose=True)


def train_routine(mod, opt, trn_ds, val_ds, tst_ds, verbose=False):
    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    trn_ds.pos1 = trn_ds.pos1.to(torch.long)
    val_ds.pos1 = val_ds.pos1.to(torch.long)
    tst_ds.pos1 = tst_ds.pos1.to(torch.long)
    batch_size = val_ds.y.shape[0]
    vprint(f"batch size{batch_size}")

    length = max(2 * batch_size, tst_ds.y.shape[0], val_ds.y.shape[0])

    best_val = 0
    tst_score = 0
    early_stop = 0
    for i in range(2000):
        t1 = time.time()
        loss = train(mod, opt, trn_ds, batch_size)
        t2 = time.time()
        import pdb
        val_score = test(mod, val_ds)
        vprint(f"epoch: {i:03d}, trn: time {t2 - t1:.2f} s, loss {loss:.4f} val {val_score:.4f}",
               end=" ")
        early_stop += 1
        if val_score > best_val:
            early_stop = 0
            best_val = val_score
            if verbose:
                t0 = time.time()
                tst_score = test(mod, tst_ds, True)
                t1 = time.time()
                vprint(f"time:{t1-t0:.4f}")
            vprint(f"tst {tst_score:.4f}")
        else:
            vprint()
        if early_stop > 200:
            break
    vprint(f"end test {tst_score:.3f} time {(t2 - t1) / 8:.3f} s")
    with open(f'./records/{args.dataset}_auc_record.txt', 'a') as f:
        f.write(
            'AUC:' + str(round(tst_score, 4)) + '   ' + 'Time:' + str(
                round(t2 - t1, 4)) + '   ' + '\n')
    return val_score


def work(device="cpu", dsname="Celegans"):  # mod_params=(32, 2, 1, 0.0), lr=3e-4
    device = torch.device(device)
    bg = load_dataset(dsname)
    bg.to(device)
    bg.preprocess()
    bg.setPosDegreeFeature()
    trn_ds = dataset(*bg.split(0))
    val_ds = dataset(*bg.split(1))
    tst_ds = dataset(*bg.split(2))

    if trn_ds.na != None:
        trn_ds.na = trn_ds.na.to(device)
        val_ds.na = val_ds.na.to(device)
        tst_ds.na = tst_ds.na.to(device)
        use_node_attr = True
        na_size = trn_ds.na.shape[1]
    else:
        use_node_attr = False
        na_size = 0

    def selparam(trial):
        lr = trial.suggest_categorical("lr",
                                       [0.0005, 0.001, 0.005, 0.01, 0.05])
        layer1 = trial.suggest_int("layer1", 1, 3)
        layer2 = trial.suggest_int("layer2", 1, 2)
        hidden_dim = trial.suggest_categorical("hidden_dim", [24, 32, 64, 96])
        dp0 = trial.suggest_float("dp0", 0.0, 0.5, step=0.1)
        dp1 = trial.suggest_float("dp1", 0.0, 0.5, step=0.1)
        dp2 = trial.suggest_float("dp2", 0.0, 0.5, step=0.1)
        dp3 = trial.suggest_float("dp1", 0.0, 0.5, step=0.1)
        act0 = trial.suggest_categorical("a0", [True, False])
        act1 = trial.suggest_categorical("a1", [True, False])
        return valparam(layer1, layer2, dp0, dp1, dp2, dp3, hidden_dim, lr, act0, act1)

    def valparam(layer1, layer2, dp0, dp1, dp2, dp3, hidden_dim, lr, act0, act1):
        mod = WLGNN(torch.max(bg.x[2]), use_node_attr, na_size,
                    *(hidden_dim, layer1, layer2, dp0, dp1, dp2, dp3, act0, act1)).to(device)
        opt = Adam(mod.parameters(), lr=lr)
        return train_routine(mod, opt, trn_ds, val_ds, tst_ds)

    study = optuna.create_study(direction="maximize",
                                storage="sqlite:///" + "opt/" + dsname +
                                        ".db",
                                study_name=dsname,
                                load_if_exists=True)
    study.optimize(selparam, n_trials=100)


def testparam(device="cpu", dsname="Celegans"):  # mod_params=(32, 2, 1, 0.0), lr=3e-4
    device = torch.device(device)
    bg = load_dataset(dsname)
    bg.to(device)
    bg.preprocess()
    bg.setPosDegreeFeature()

    trn_ds = dataset(*bg.split(0))
    val_ds = dataset(*bg.split(1))
    tst_ds = dataset(*bg.split(2))
    if trn_ds.na != None:
        print("use node feature")
        trn_ds.na = trn_ds.na.to(device)
        val_ds.na = val_ds.na.to(device)
        tst_ds.na = tst_ds.na.to(device)
        use_node_attr = True
        na_size = trn_ds.na.shape[1]
    else:
        use_node_attr = False
        na_size = 0


    def valparam(layer1, layer2, dp0, dp1, dp2, dp3, hidden_dim, lr, act0=True, act1=True):

        import pdb
        mod = WLGNN(torch.max(bg.x[2]), use_node_attr, na_size,
                    *(hidden_dim, layer1, layer2, dp0, dp1, dp2, dp3, act0, act1)).to(device)
        opt = Adam(mod.parameters(), lr=lr)
        return train_routine(mod, opt, trn_ds, val_ds, tst_ds, verbose=True)

    if args.check:
        study = optuna.create_study(direction="maximize",
                                storage="sqlite:///" + args.path + dsname +
                                        ".db",
                                study_name=dsname,
                                load_if_exists=True)
        import pandas as pd
        pd.set_option('display.max_rows', None)
        df = study.trials_dataframe().drop(['state', 'datetime_start', 'datetime_complete', 'duration', 'number'], axis=1)
        import pdb
        pdb.set_trace()
        print("best param", study.best_params)
    best_params = {
        'Celegans': {
            'dp0': 0.2,
            'dp1': 0.2,
            'dp2': 0.2,
            'dp3': 0.2,
            'hidden_dim': 64,
            'layer1': 2,
            'layer2': 3,
            'lr': 0.005
        },
        'USAir': {
            'dp0': 0.1,
            'dp1': 0.1,
            'dp2': 0.1,
            'dp3': 0.1,
            'hidden_dim': 64,
            'layer1': 3,
            'layer2': 2,
            'lr': 0.001
        },
        'PB': {
            'dp0': 0.0,
            'dp1': 0.0,
            'dp2': 0.0,
            'dp3': 0.0,
            'hidden_dim': 32,
            'layer1': 3,
            'layer2': 2,
            'lr': 0.01
        },
        'NS': {
            'dp0': 0.1,
            'dp1': 0.1,
            'dp2': 0.1,
            'dp3': 0.1,
            'hidden_dim': 48,
            'layer1': 3,
            'layer2': 1,
            'lr': 0.001
        },
        'Ecoli': {
            'dp0': 0.1,
            'dp1': 0.1,
            'dp2': 0.1,
            'dp3': 0.1,
            'hidden_dim': 32,
            'layer1': 3,
            'layer2': 1,
            'lr': 0.005
        },
        'Router': {
            'dp0': 0.1,
            'dp1': 0.1,
            'dp2': 0.1,
            'dp3': 0.1,
            'hidden_dim': 48,
            'layer1': 3,
            'layer2': 2,
            'lr': 0.01
        },
        'Power': {
            'dp0': 0.1,
            'dp1': 0.1,
            'dp2': 0.1,
            'dp3': 0.1,
            'hidden_dim': 64,
            'layer1': 3,
            'layer2': 3,
            'lr': 0.05
        },
        'Yeast': {
            'dp0': 0.0,
            'dp1': 0.0,
            'dp2': 0.0,
            'dp3': 0.0,
            'hidden_dim': 64,
            'layer1': 3,
            'layer2': 3,
            'lr': 0.005
        },
        'Cora': {
            'dp0': 0.3,
            'dp1': 0.3,
            'dp2': 0.1,
            'dp3': 0.0,
            'hidden_dim': 64,
            'layer1': 2,
            'layer2': 2,
            'act0': False,
            'lr': 0.005
        },
        'Citeseer': {
            'dp0': 0.0,
            'dp1': 0.4,
            'dp2': 0.1,
            'dp3': 0.0,
            'hidden_dim': 32,
            'layer1': 2,
            'layer2': 1,
            'act0': False,
            'lr': 0.005
        },
        'Pubmed': {
            'dp0': 0.1,
            'dp1': 0.1,
            'dp2': 0.1,
            'dp3': 0.1,
            'hidden_dim': 128,
            'layer1': 1,
            'layer2': 2,
            'lr': 0.01
        },
        'ogbl-collab': {
            'dp0': 0.0,
            'dp1': 0.0,
            'dp2': 0.0,
            'dp3': 0.0,
            'hidden_dim': 16,
            'layer1': 2,
            'layer2': 1,
            'lr': 0.01
        }
    }
    params = best_params[dsname]

    valparam(**(params))
    #valparam(**(study.best_params))


def reproduce(device, ds):
    device = torch.device(device)
    bg = load_dataset(ds)
    bg.to(device)
    bg.preprocess()
    bg.setPosDegreeFeature()
    trn_ds = dataset(*bg.split(0))
    val_ds = dataset(*bg.split(1))
    tst_ds = dataset(*bg.split(2))

    if trn_ds.na != None:
        trn_ds.na = trn_ds.na.to(device)
        val_ds.na = val_ds.na.to(device)
        tst_ds.na = tst_ds.na.to(device)

    lr = 1e-2
    use_node_attr = False
    na_size = 0
    if ds == "PB":
        hidden_dim = 96
    elif ds == "Yeast":
        hidden_dim = 64
    elif ds == "Celegans":
        hidden_dim = 32
    elif ds == "Power":
        hidden_dim = 64
    elif ds == "USAir":
        hidden_dim = 24
    elif ds == "NS":
        hidden_dim = 64
    elif ds == "Router":
        hidden_dim = 64
    elif ds == "Ecoli":
        hidden_dim = 64
    elif ds == "Cora":
        hidden_dim = 32
        use_node_attr = True
        na_size = trn_ds.na.shape[1]
    elif ds == "Citeseer":
        hidden_dim = 64
        use_node_attr = True
        na_size = trn_ds.na.shape[1]
    elif ds == "Pubmed":
        hidden_dim = 128
        use_node_attr = True
        na_size = trn_ds.na.shape[1]
    else:
        raise NotImplementedError

    mod = Model_HY(torch.max(bg.x[2]), use_node_attr, na_size, *(hidden_dim, 2, 1, 0)).to(device)
    opt = Adam(mod.parameters(), lr=lr)
    train_routine(mod, opt, trn_ds, val_ds, tst_ds, verbose=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', type=str, default="USAir")
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--path', type=str, default="opt/")
    parser.add_argument('--test', action="store_true")
    parser.add_argument('--check', action="store_true")
    parser.add_argument('--reproduce', action="store_true")
    args = parser.parse_args()
    if args.device < 0:
        args.device = "cpu"
    else:
        args.device = "cuda:" + str(args.device)
    print(args.device)
    if args.test:
        testparam(args.device, args.dataset)
    elif args.reproduce:
        for i in range(10):
            reproduce(args.device, args.dataset)
    else:
        work(args.device, args.dataset)
    # main(args.device, args.dataset, lr=1e-3)