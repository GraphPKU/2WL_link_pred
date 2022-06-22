import random
from model import Net
from datasets import load_dataset, Dataset
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import roc_auc_score
import time
from ogb.linkproppred import Evaluator
from utils import idx2mask, degree
from torch_geometric.utils import negative_sampling, add_self_loops
import optuna
import numpy as np
from torch.cuda.amp import autocast as autocast, GradScaler
from datetime import datetime


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi gpu

def evaluate_hits(pos_pred, neg_pred, K):
    results = {}
    evaluator = Evaluator(name='ogbl-ddi')
    evaluator.K = K
    hits = evaluator.eval({
        'y_pred_pos': pos_pred,
        'y_pred_neg': neg_pred,
    })[f'hits@{K}']

    results[f'Hits@{K}'] = hits

    return results

perm1, perm2, pos_batchsize, neg_batchsize = None, None, None, None


def train(mod, opt, dataset, batch_size, i):
    global perm1, perm2, pos_batchsize, neg_batchsize
    mod.train()
    dataset.neg_pos1 = dataset.neg_pos1.to(torch.long)
    if i == 0:
        perm1 = torch.randperm(dataset.pos1.shape[0], device=dataset.x.device)

        pos_batchsize = batch_size // 2
        neg_batchsize = pos_batchsize

    idx1 = perm1[i * pos_batchsize:(i + 1) * pos_batchsize]

    if dataset.neg_pos1.shape[0] > 0:
        perm2 = torch.randperm(dataset.neg_pos1.shape[0], device=dataset.x.device)
        idx2 = perm2[i * neg_batchsize:(i + 1) * neg_batchsize]
        neg_pos1 = dataset.neg_pos1[idx2]
    else:
        edge_index, _ = add_self_loops(dataset.ei)
        neg_pos1 = negative_sampling(
            edge_index,
            num_nodes=dataset.x.shape[0],
            num_neg_samples=neg_batchsize,
        ).t().to(dataset.x.device)
    y = torch.cat((torch.ones(
        pos_batchsize, dtype=torch.float, device=dataset.x.device),
                   torch.zeros((neg_batchsize,), dtype=torch.float, device=dataset.x.device)),
        dim=0).unsqueeze(-1)
    delei = dataset.pos1[idx1].t()
    delei = torch.cat((delei, delei[[1, 0]]), dim=-1)
    x = dataset.x - degree(delei, dataset.x.shape[0])
    # node_label = torch.arange(dataset.x.shape[0]).to(dataset.x.device)
    eimask = torch.logical_not(idx2mask(perm1.shape[0], idx1))
    ei = dataset.pos1[eimask].t()
    ei = torch.cat((ei, ei[[1, 0]]), dim=-1)

    pos1 = torch.cat((dataset.pos1[idx1], neg_pos1), dim=0)
    opt.zero_grad()
    if args.hf:
        scaler = GradScaler()
        with autocast():
            pred = mod(x, ei, pos1)
            loss = F.binary_cross_entropy_with_logits(pred, y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    else:
        pred = mod(x, ei, pos1)
        loss = F.binary_cross_entropy_with_logits(pred, y)
        loss.backward()
        opt.step()
    result = roc_auc_score(y.cpu().numpy(), pred.detach().flatten().sigmoid().cpu().numpy())
    i += 1
    if (i + 1) * pos_batchsize > perm1.shape[0]:
        i = 0
    return loss.item(), result, i


@torch.no_grad()
def test(mod, dataset):
    mod.eval()
    #node_label = torch.arange(dataset.x.shape[0]).to(dataset.x.device)
    if False:
        with autocast():
            pred = mod(dataset.x, dataset.ei, dataset.pos1).flatten()
    else:
        pred = mod(dataset.x, dataset.ei, dataset.pos1).flatten()
    sig = pred.sigmoid().cpu().numpy()
    # print("y", torch.unique(dataset.y.reshape(-1, 2)[:, 0]), dataset.y.reshape(-1, 2)[:, 0].shape)
    if True:
        result = roc_auc_score(dataset.y.cpu().numpy(), sig)
    else:
        y = dataset.y[mask0].to(torch.bool)
        result = evaluate_hits(sig[y], sig[~y], K=20)['Hits@20']
    return result


def train_routine(mod, opt, trn_ds, val_ds, tst_ds, verbose=False):
    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    trn_ds.pos1 = trn_ds.pos1.to(torch.long)
    val_ds.pos1 = val_ds.pos1.to(torch.long)
    tst_ds.pos1 = tst_ds.pos1.to(torch.long)

    batch_size = tst_ds.y.shape[0]
    vprint(f"batch size{batch_size}")

    best_val = 0
    tst_score = 0
    early_stop = 0
    train_idx = 0
    #scaler = GradScaler()
    for i in range(1500):
        train_idx = 0
        t1 = time.time()
        loss, trn_score, train_idx = train(mod, opt, trn_ds, batch_size, train_idx)
        t2 = time.time()
        val_score = test(mod, val_ds)
        vprint(f"epoch {i}: trn: time {t2 - t1:.2f} s, loss {loss:.4f}, trn {trn_score:.4f}, val {val_score:.4f}",
               end=" ")
        if args.test:
            with open(f'./records/{args.dataset}_{time_}_auc_record.txt', 'a') as f:
                f.write(
                    'Epoch: ' + str(i) + '   ' + 'Time: ' + str(
                        round(t2 - t1, 2)) + 's' + '   ' + 'Loss: ' + str(round(loss, 4)) + '   ' +
                    'train: ' + str(round(trn_score, 4)) + '   ' + 'val: ' + str(round(val_score, 4))
                )
        early_stop += 1
        if val_score > best_val:
            early_stop = 0
            best_val = val_score
            if verbose:
                tst_score = test(mod, tst_ds)
            vprint(f"tst {tst_score:.4f}")
            if args.test:
                with open(f'./records/{args.dataset}_{time_}_auc_record.txt', 'a') as f:
                    f.write('   '+'test: ' + str(round(tst_score, 4)) + '\n')
        else:
            vprint()
            if args.test:
                with open(f'./records/{args.dataset}_{time_}_auc_record.txt', 'a') as f:
                    f.write('\n')
        if early_stop > 200:
            break
    vprint(f"end test {tst_score:.3f} time {t2-t1:.3f} s")
    with open(f'./records/{args.dataset}_{time_}_auc_record.txt', 'a') as f:
        f.write(
            'AUC: ' + str(round(tst_score, 4)) + '   ' + 'Time: ' + str(
                round(t2 - t1, 4)) + '   ' + '\n')
    return best_val


def work(device="cpu", dsname="Celegans"):
    device = torch.device(device)
    bg = load_dataset(dsname)
    bg.to(device)
    bg.preprocess()
    bg.setPosDegreeFeature()
    trn_ds = Dataset(*bg.split(0))
    val_ds = Dataset(*bg.split(1))
    tst_ds = Dataset(*bg.split(2))

    def selparam(trial):
        nonlocal bg, trn_ds, val_ds, tst_ds
        if random.random() < 0.1:
            bg = load_dataset(dsname)
            bg.to(device)
            bg.preprocess()
            bg.setPosDegreeFeature()
            trn_ds = Dataset(*bg.split(0))
            val_ds = Dataset(*bg.split(1))
            tst_ds = Dataset(*bg.split(2))
        lr = trial.suggest_categorical("lr",
                                       [0.0005, 0.001, 0.005, 0.01, 0.05])
        layer1 = trial.suggest_int("l1", 1, 3)
        layer2 = trial.suggest_int("l2", 1, 2)
        hidden_dim_1 = trial.suggest_categorical("h1", [24, 32, 64])
        hidden_dim_2 = trial.suggest_categorical("h2", [16, 24])
        dp1 = trial.suggest_float("dp1", 0.0, 0.5, step=0.1)
        dp2 = trial.suggest_float("dp2", 0.0, 0.5, step=0.1)
        dp3 = trial.suggest_float("dp3", 0.0, 0.5, step=0.1)
        pool_function = trial.suggest_categorical("pl", [True, False])
        return valparam(hidden_dim_1, hidden_dim_2, layer1, layer2, dp1, 0.05, dp3, lr, pool_function)

    def valparam(hidden_dim_1, hidden_dim_2, layer1, layer2, dp1, dp2, dp3, lr, pool_function):
        if bg.x.shape[1] > 0:
            mod = Net(bg.max_x,
                      hidden_dim_1,
                      hidden_dim_2,
                      layer1,
                      layer2,
                      dp1,
                      dp2,
                      dp3,
                      mul_pool=pool_function,
                      use_feat=True,
                      feat=bg.x).to(device)
        else:
            mod = Net(bg.x.shape[0], hidden_dim_1, hidden_dim_2, layer1, layer2, dp1, dp2,
                      dp3, mul_pool=pool_function).to(device)
        opt = Adam(mod.parameters(), lr=lr)
        return train_routine(mod, opt, trn_ds, val_ds, tst_ds)

    study = optuna.create_study(direction="maximize",
                                storage="sqlite:///" + args.path + dsname +
                                ".db",
                                study_name=dsname,
                                load_if_exists=True)
    print(
        f"storage {'sqlite:///' + args.path + dsname + '.db'} study_name {dsname}"
    )
    study.optimize(selparam, n_trials=200)


def testparam(device="cpu",
              dsname="Celegans"):  # mod_params=(32, 2, 1, 0.0), lr=3e-4
    device = torch.device(device)

    def valparam(hidden_dim_1, hidden_dim_2, layer1, layer2, dp1, dp2, dp3, lr, act=True, pool_function=1):
        if len(bg.x.shape) > 1:
            print("use_feat")
            mod = Net(bg.max_x,
                      hidden_dim_1,
                      hidden_dim_2,
                      layer1,
                      layer2,
                      dp1,
                      dp2,
                      dp3,
                      mul_pool=pool_function,
                      use_feat=True,
                      feat=bg.x,
                      act=act).to(device)
        else:
            mod = Net(bg.max_x, hidden_dim_1, hidden_dim_2, layer1, layer2, dp1, dp2,
                      dp3, act=act).to(device)
        opt = Adam(mod.parameters(), lr=lr)
        return train_routine(mod, opt, trn_ds, val_ds, tst_ds, verbose=True)
    best_params = {
        'Celegans': {
            'dp1': 0.3,
            'dp2': 0.3,
            'dp3': 0.3,
            'hidden_dim_1': 64,
            'hidden_dim_2': 64,
            'layer1': 2,
            'layer2': 2,
            'lr': 0.005,
            'act': True,
        },
        'USAir': {
            'dp1': 0.4,
            'dp2': 0.0,
            'dp3': 0.1,
            'hidden_dim_1': 24,
            'hidden_dim_2': 24,
            'layer1': 1,
            'layer2': 2,
            'lr': 0.01,
            'act': True,
        },
        'PB': {
            'dp1': 0.4,
            'dp2': 0.4,
            'dp3': 0.4,
            'hidden_dim_1': 32,
            'hidden_dim_2': 32,
            'layer1': 3,
            'layer2': 3,
            'lr': 0.005,
            'act': True,
        },
        'NS': {
            'dp1': 0.1,
            'dp2': 0.1,
            'dp3': 0.1,
            'hidden_dim_1': 16,
            'hidden_dim_2': 16,
            'layer1': 3,
            'layer2': 1,
            'lr': 0.005,
            'act': True,
        },
        'Ecoli': {
            'dp1': 0.3,
            'dp2': 0.0,
            'dp3': 0.4,
            'hidden_dim_1': 32,
            'hidden_dim_2': 32,
            'layer1': 3,
            'layer2': 1,
            'lr': 0.01,
            'act': True,
        },
        'Router': {
            'dp1': 0.4,
            'dp2': 0.00,
            'dp3': 0.00,
            'hidden_dim_1': 16,
            'hidden_dim_2': 16,
            'layer1': 2,
            'layer2': 1,
            'lr': 0.01,
            'act': True,
        },
        'Power': {
            'dp1': 0.3,
            'dp2': 0.0,
            'dp3': 0.1,
            'hidden_dim_1': 32,
            'hidden_dim_2': 8,
            'layer1': 3,
            'layer2': 3,
            'lr': 0.01,
            'act': True,
        },
        'Yeast': {
            'dp1': 0.4,
            'dp2': 0.4,
            'dp3': 0.4,
            'hidden_dim_1': 64,
            'hidden_dim_2': 32,
            'layer1': 1,
            'layer2': 2,
            'lr': 0.005,
            'act': True,
        },
        'Cora': {
            'dp1': 0.3,
            'dp2': 0.5,
            'dp3': 0.0,
            'hidden_dim_1': 64,
            'hidden_dim_2': 32,
            'layer1': 2,
            'layer2': 1,
            'lr': 0.005,
            'act': False,
        },
        'Citeseer': {
            'dp1': 0.1,
            'dp2': 0.2,
            'dp3': 0.4,
            'hidden_dim_1': 24,
            'hidden_dim_2': 16,
            'layer1': 1,
            'layer2': 1,
            'lr': 0.001,
            'act': False,
        },
        'ogbl-ddi':{
            'dp1': 0.0,
            'dp2': 0.0,
            'dp3': 0.0,
            'hidden_dim_1': 64,
            'hidden_dim_2': 64,
            'layer1': 2,
            'layer2': 1,
            'lr': 0.01
        }
    }
    params = {
        'dp1': args.dp1,
        'dp2': args.dp2,
        'dp3': args.dp3,
        'hidden_dim_1': args.hidden_1,
        'hidden_dim_2': args.hidden_2,
        'layer1': args.layer_1,
        'layer2': args.layer_2,
        'lr': args.lr,
        'pool_function':args.pl
    }
    with open(f'./records/{args.dataset}_{time_}_auc_record.txt', 'a') as f:
        f.write(
            '----------------------------------' + '\n')
        f.write(
            'layer_1: ' + str(args.layer_1) + '   ' + 'layer_2: ' + str(
                args.layer_2) + '   ' +
            'hidden_1: ' + str(args.hidden_1) + '   ' + 'hidden_2: ' + str(
                args.hidden_2) + '   ' + 'dp1: ' + str(round(args.dp1, 1)) + '   ' + 'dp2:' + str(
                round(args.dp2, 1)) + '   ' + 'dp3:' + str(
                round(args.dp2, 1)) + '   ' + 'lr: ' + str(round(args.lr, 4)) +
            '\n')
        f.write(
            '----------------------------------' + '\n')
    for i in range(10):
        set_seed(i)
        print(f"repeat {i}")
        bg = load_dataset(dsname)
        bg.to(device)
        bg.preprocess()
        bg.setPosDegreeFeature()
        trn_ds = Dataset(*bg.split(0))
        val_ds = Dataset(*bg.split(1))
        tst_ds = Dataset(*bg.split(2))

        if args.use_best:
            params = best_params[dsname]
            print("best param", params)
            valparam(**(params))
        else:
            print("Val param", params)
            valparam(**(params))

def check(dsname):
    study = optuna.create_study(direction="maximize",
                                storage="sqlite:///" + args.path + dsname +
                                        ".db",
                                study_name=dsname,
                                load_if_exists=True)
    import pandas as pd
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    df = study.trials_dataframe().drop(['state', 'datetime_start', 'datetime_complete', 'duration', 'number'],
                                       axis=1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', type=str, default="USAir")
    parser.add_argument('--dp1', default=0.0)
    parser.add_argument('--dp2', default=0.0)
    parser.add_argument('--dp3', default=0.0)
    parser.add_argument('--layer_1', default=2)
    parser.add_argument('--layer_2', default=1)
    parser.add_argument('--hidden_1', default=64)
    parser.add_argument('--hidden_2', default=32)
    parser.add_argument('--lr', default=0.01)
    parser.add_argument('--hf', action="store_true", help="Half float")

    parser.add_argument('--pl', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--path', type=str, default="Opt/")
    parser.add_argument('--test', action="store_true")
    parser.add_argument('--check', action="store_true")
    parser.add_argument('--use_best', action="store_true")
    args = parser.parse_args()

    dt = datetime.now()
    time_ = str(dt.day) + '_' + str(dt.hour) + '_' + str(dt.minute) + '_' + str(dt.second)

    if args.device < 0:
        args.device = "cpu"
    else:
        args.device = "cuda:" + str(args.device)
    print(args.dataset, args.device)
    if args.test:
        testparam(args.device, args.dataset)
    elif args.check:
        check(args.dataset)
    else:
        work(args.device, args.dataset)
