import random
from model import Net
from datasets import load_dataset, Dataset
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import roc_auc_score
import time
from utils import idx2mask, degree, compute_D
import optuna
import numpy as np


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi gpu


perm1, perm2, pos_batchsize, neg_batchsize = None, None, None, None


def train(mod, opt, dataset, batch_size, i):
    global perm1, perm2, pos_batchsize, neg_batchsize
    mod.train()
    if i == 0:
        perm1 = torch.randperm(dataset.pos1.shape[0], device=dataset.x.device)
        perm2 = torch.randperm(dataset.neg_pos1.shape[0],
                               device=dataset.x.device)
        pos_batchsize = batch_size // 2
        neg_batchsize = (perm2.shape[0]) // (dataset.ei.shape[1] //
                                             pos_batchsize)

    idx1 = perm1[i * pos_batchsize:(i + 1) * pos_batchsize]
    idx2 = perm2[i * neg_batchsize:(i + 1) * neg_batchsize]
    neg_pos1 = dataset.neg_pos1[idx2]
    y = torch.cat((torch.ones_like(
        idx1, dtype=torch.float), torch.zeros_like(idx2, dtype=torch.float)),
                  dim=0).unsqueeze(-1)
    delei = dataset.pos1[idx1].t()
    delei = torch.cat((delei, delei[[1, 0]]), dim=-1)
    x = dataset.x - degree(delei, dataset.x.shape[0])
    eimask = torch.logical_not(idx2mask(perm1.shape[0], idx1))
    ei = dataset.pos1[eimask].t()
    ei = torch.cat((ei, ei[[1, 0]]), dim=-1)
    pos1 = torch.cat((dataset.pos1[idx1], neg_pos1), dim=0)
    opt.zero_grad()
    D = compute_D(ei, x.shape[0])
    print(D.shape, D.dtype)
    pred = mod(x, ei, pos1, D)
    loss = F.binary_cross_entropy_with_logits(pred, y)
    loss.backward()
    opt.step()
    i += 1
    if (i + 1) * pos_batchsize > perm1.shape[0]:
        i = 0
    return loss.item(), i


@torch.no_grad()
def test(mod, dataset):
    mod.eval()
    pred = mod(dataset.x, dataset.ei, dataset.pos1, dataset.D).flatten()
    sig = pred.sigmoid().cpu().numpy()
    # print("y", torch.unique(dataset.y.reshape(-1, 2)[:, 0]), dataset.y.reshape(-1, 2)[:, 0].shape)
    return roc_auc_score(dataset.y.cpu().numpy(), sig)


def train_routine(mod, opt, trn_ds, val_ds, tst_ds, verbose=False):
    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    trn_ds.pos1 = trn_ds.pos1.to(torch.long)
    val_ds.pos1 = val_ds.pos1.to(torch.long)
    tst_ds.pos1 = tst_ds.pos1.to(torch.long)
    val_ds.D = compute_D(val_ds.ei, val_ds.x.shape[0])
    tst_ds.D = compute_D(tst_ds.ei, tst_ds.x.shape[0])
    batch_size = tst_ds.y.shape[0]
    vprint(f"batch size{batch_size}")

    best_val = 0
    tst_score = 0
    early_stop = 0
    train_idx = 0
    for i in range(1500):
        t1 = time.time()
        loss, train_idx = train(mod, opt, trn_ds, batch_size, train_idx)
        t2 = time.time()
        val_score = test(mod, val_ds)
        vprint(f"trn: time {t2-t1:.2f} s, loss {loss:.4f} val {val_score:.4f}",
               end=" ")
        early_stop += 1
        if val_score > best_val:
            early_stop = 0
            best_val = val_score
            if verbose:
                tst_score = test(mod, tst_ds)
            vprint(f"tst {tst_score:.4f}")
        else:
            vprint()
    vprint(f"end test {tst_score:.3f} time {t2-t1:.3f} s")
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
        layer1 = trial.suggest_int("layer1", 1, 3)
        layer2 = trial.suggest_int("layer2", 1, 2)
        hidden_dim = trial.suggest_categorical("hidden_dim", [8, 16, 24, 32])
        dp1 = trial.suggest_float("dp1", 0.0, 0.7, step=0.05)
        dp2 = trial.suggest_float("dp2", 0.0, 0.5, step=0.05)
        dp3 = trial.suggest_float("dp3", 0.0, 0.5, step=0.05)
        return valparam(hidden_dim, layer1, layer2, dp1, dp2, dp3, lr)

    def valparam(hidden_dim, layer1, layer2, dp1, dp2, dp3, lr):
        if bg.x.shape[1] > 0:
            mod = Net(bg.max_x,
                      hidden_dim,
                      layer1,
                      layer2,
                      dp1,
                      dp2,
                      dp3,
                      use_feat=True,
                      feat=bg.x).to(device)
        else:
            mod = Net(bg.max_x, hidden_dim, layer1, layer2, dp1, dp2,
                      dp3).to(device)
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
    study.optimize(selparam, n_trials=100)


def testparam(device="cpu",
              dsname="Celegans"):  # mod_params=(32, 2, 1, 0.0), lr=3e-4
    device = torch.device(device)

    def valparam(hidden_dim, layer1, layer2, dp1, dp2, dp3, lr):
        if bg.x.shape[1] > 0:
            print("use_feat")
            mod = Net(bg.max_x,
                      hidden_dim,
                      layer1,
                      layer2,
                      dp1,
                      dp2,
                      dp3,
                      use_feat=True,
                      feat=bg.x).to(device)
        else:
            mod = Net(bg.max_x, hidden_dim, layer1, layer2, dp1, dp2,
                      dp3).to(device)
        opt = Adam(mod.parameters(), lr=lr)
        return train_routine(mod, opt, trn_ds, val_ds, tst_ds, verbose=True)

    best_params = {
        'Celegans': {
            'dp1': 0.2,
            'dp2': 0.1,
            'dp3': 0.3,
            'hidden_dim': 32,
            'layer1': 3,
            'layer2': 2,
            'lr': 0.01
        },
        'USAir': {
            'dp1': 0.1,
            'dp2': 0.0,
            'dp3': 0.3,
            'hidden_dim': 24,
            'layer1': 3,
            'layer2': 1,
            'lr': 0.005
        },
        'PB': {
            'dp1': 0.55,
            'dp2': 0.0,
            'dp3': 0.35,
            'hidden_dim': 24,
            'layer1': 3,
            'layer2': 2,
            'lr': 0.05
        },
        'NS': {
            'dp1': 0.7,
            'dp2': 0.0,
            'dp3': 0.1,
            'hidden_dim': 32,
            'layer1': 3,
            'layer2': 1,
            'lr': 0.005
        },
        'Ecoli': {
            'dp1': 0.3,
            'dp2': 0.0,
            'dp3': 0.4,
            'hidden_dim': 32,
            'layer1': 3,
            'layer2': 1,
            'lr': 0.01
        },
        'Router': {
            'dp1': 0.35,
            'dp2': 0.05,
            'dp3': 0.05,
            'hidden_dim': 24,
            'layer1': 2,
            'layer2': 1,
            'lr': 0.005
        },
        'Power': {
            'dp1': 0.65,
            'dp2': 0.25,
            'dp3': 0.1,
            'hidden_dim': 8,
            'layer1': 3,
            'layer2': 2,
            'lr': 0.01
        },
        'Yeast': {
            'dp1': 0.0,
            'dp2': 0.0,
            'dp3': 0.0,
            'hidden_dim': 32,
            'layer1': 3,
            'layer2': 1,
            'lr': 0.05
        },
        'Cora': {
            'dp1': 0.4,
            'dp2': 0.05,
            'dp3': 0.3,
            'hidden_dim': 24,
            'layer1': 1,
            'layer2': 1,
            'lr': 0.005
        },
        'Citeseer': {
            'dp1': 0.65,
            'dp2': 0.0,
            'dp3': 0.1,
            'hidden_dim': 24,
            'layer1': 2,
            'layer2': 1,
            'lr': 0.05
        }
    }
    # {'dp1': 0.2, 'dp2': 0.0, 'dp3': 0.2, 'hidden_dim': 32, 'layer1': 3, 'layer2': 1, 'lr': 0.005}
    params = best_params[dsname]
    print("best param", params)
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
        valparam(**(params))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', type=str, default="Celegans")
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--path', type=str, default="Opt/")
    parser.add_argument('--test', action="store_true", default=True)
    args = parser.parse_args()
    if args.device < 0:
        args.device = "cpu"
    else:
        args.device = "cuda:" + str(args.device)
    print(args.device)
    if args.test:
        testparam(args.device, args.dataset)
    else:
        work(args.device, args.dataset)
