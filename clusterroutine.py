import enum
from utils import tensorDataloader
import torch.nn.functional as F
from torch_geometric.data import ClusterData, ClusterLoader
import torch
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from torch_geometric.utils import degree
from torch import Tensor
from utils import reverseperm
import torch.nn as nn
from torch_geometric.data import Data
from torch.optim import Optimizer, Adam
from torch_geometric.utils import negative_sampling
import numpy as np
from torch_sparse import SparseTensor
from sklearn.linear_model import LogisticRegression


def to_directed(ei):
    return ei[:, ei[0] <= ei[1]]


def to_undirected(ei):
    return torch.cat((ei, ei[[1, 0]]), dim=-1)


def slicespm(spm: SparseTensor, tar: Tensor):
    spm.coalesce()
    row, col, val = spm.coo()
    size1 = spm.sizes()[1] + 5
    hashsp = row * size1 + col
    #hashsp, ids = torch.sort(hashsp)
    #val = val[ids]
    assert not (torch.diff(hashsp) < 0).any()
    hashtar = tar[0] * size1 + tar[1]
    searchret = torch.searchsorted(hashsp[:-1], hashtar)
    ret = val[searchret]
    ret[hashsp[searchret] != hashtar] = 0
    return ret


def computeCNAA(ei, tarei, num_nodes):
    adj = SparseTensor(row=ei[0],
                       col=ei[1],
                       value=torch.ones_like(ei[1], dtype=torch.float),
                       sparse_sizes=(num_nodes, num_nodes)).cuda().coalesce()
    cn = slicespm(adj @ adj, tarei)
    deg = adj.sum(dim=1).flatten()
    deg[deg < 1.5] = 2
    idx = torch.arange(num_nodes, device=device)
    Dinv = SparseTensor(row=idx,
                        col=idx,
                        value=1 / deg,
                        sparse_sizes=(num_nodes, num_nodes)).cuda()
    ra = slicespm(adj @ Dinv @ adj, tarei)
    Dloginv = SparseTensor(row=idx,
                        col=idx,
                        value=1 /torch.log(deg),
                        sparse_sizes=(num_nodes, num_nodes)).cuda()
    aa = slicespm(adj @ Dloginv @ adj, tarei)
    return torch.stack((ra, aa, cn), dim=-1)


ds_name = "ogbl-collab"
dataset = PygLinkPropPredDataset(name=ds_name, root="./data/")
data = dataset[0]
es = dataset.get_edge_split()
device = torch.device("cuda")


@torch.no_grad()
def test(num_clu,
         nnodes,
         partptr,
         nbin,
         nes,
         datalist,
         mod: nn.Module,
         evaluator,
         split="valid",
         cnaapred=None):
    mod.eval()
    num_pos, num_neg = nbin[split]["edge"].shape[0], nbin[split][
        "edge_neg"].shape[0]
    tb = torch.cat((nbin[split]["edge"], nbin[split]["edge_neg"]))
    te = torch.cat((nes[split]["edge"], nes[split]["edge_neg"]), dim=-1)
    pred = torch.zeros((num_pos + num_neg), device=datalist[0].device)
    for i in range(num_clu):
        ei = datalist[i]
        mask = tb == i
        tar_ei = te[:, mask] - partptr[i]
        x = torch.ones(nnodes[i], device=ei.device, dtype=torch.long)
        pred[mask] = mod(x, to_undirected(ei), tar_ei).flatten()
    if cnaapred is not None:
        pred[tb == -1] = cnaapred
    return evaluator.eval({
        'y_pred_pos': pred[:num_pos],
        'y_pred_neg': pred[num_pos:]
    })["hits@50"]


def train(num_clu, nnodes, datalist, negdatalist, max_iter, mod: nn.Module,
          opt: Optimizer, batch_size: int):
    losss = []
    mod.train()
    for i in range(num_clu):
        dl = tensorDataloader(datalist[i], batch_size, ret_rev=True)
        dlneg = tensorDataloader(negdatalist[i], batch_size, ret_rev=False)
        dlneg = iter(dlneg)
        for j, batch in enumerate(dl):
            if j > max_iter:
                break
            ei, tar_ei = batch
            opt.zero_grad(set_to_none=True)
            negedge = next(dlneg)
            x = torch.ones(nnodes[i], dtype=torch.long, device=ei.device)
            pred = mod(x, to_undirected(ei),
                       torch.cat((tar_ei, negedge), dim=-1))
            loss = -F.logsigmoid(pred[:tar_ei.shape[1]]).mean() - F.logsigmoid(
                -pred[tar_ei.shape[1]:]).mean()
            loss.backward()
            opt.step()
            losss.append(loss)
    return np.average([i.item() for i in losss])


def buildmodel(**kwargs) -> nn.Module:
    from model import WXYFWLNet
    return WXYFWLNet(**kwargs)


def routine(num_clu: int, num_epoch: int, lr: float, batch_size: int,
            **kwargs):
    cld = ClusterData(data, num_clu)
    cld.partptr = cld.partptr.to(device, non_blocking=True)
    cld.perm = cld.perm.to(device, non_blocking=True)
    num_nodes = data.num_nodes
    nnodes = torch.diff(cld.partptr).cpu().numpy()
    revperm = reverseperm(cld.perm)
    nei = revperm[data.edge_index].to(device, non_blocking=True)
    neibin = torch.searchsorted(cld.partptr, nei, right=True) - 1
    neibin[0, neibin[0] != neibin[1]] = -1
    neibin = neibin[0]
    print("neibin end", flush=True)
    datalist = [nei[:, neibin == i] - cld.partptr[i] for i in range(num_clu)]
    negdatalist = [
        negative_sampling(datalist[i], nnodes[i], 10 * datalist[i].shape[1])
        for i in range(num_clu)
    ]
    datalist = [to_directed(_) for _ in datalist]

    exedge = {"train": {}, "valid": {}, "test": {}}
    exedge["train"]["edge"] = to_directed(nei[:, neibin == -1])
    exedge["train"]["edge_neg"] = negative_sampling(nei,
                                                num_neg_samples=10 *
                                                nei.shape[1])

    print("datalist end", flush=True)

    nes = {"valid": {}, "test": {}}
    nes["valid"]["edge"] = revperm[es["valid"]["edge"].t()].to(
        device, non_blocking=True)
    nes["valid"]["edge_neg"] = revperm[es["valid"]["edge_neg"].t()].to(
        device, non_blocking=True)
    nes["test"]["edge"] = revperm[es["test"]["edge"].t()].to(device,
                                                             non_blocking=True)
    nes["test"]["edge_neg"] = revperm[es["test"]["edge_neg"].t()].to(
        device, non_blocking=True)
    nbin = {"valid": {}, "test": {}}
    for key1 in nes:
        for key2 in nes[key1]:
            tb = torch.searchsorted(cld.partptr, nes[key1][key2],
                                    right=True) - 1
            tb[0, tb[0] != tb[1]] = -1
            nbin[key1][key2] = tb[0]
            exedge[key1][key2] = nes[key1][key2][:, nbin[key1][key2] == -1]

    texedge = torch.cat([
        exedge[key1][key2] for key1 in ["train", "valid", "test"]
        for key2 in ["edge", "edge_neg"]
    ], dim=-1)
    texedge_idx = [0] + [
        exedge[key1][key2].shape[1] for key1 in ["train", "valid", "test"]
        for key2 in ["edge", "edge_neg"]
    ]
    texedge_idx = np.cumsum(texedge_idx)
    cnaa = computeCNAA(nei, texedge, num_nodes)
    for i1, key1 in enumerate(["train", "valid", "test"]):
        for i2, key2 in enumerate(["edge", "edge_neg"]):
            exedge[key1][key2] = cnaa[texedge_idx[i1 * 2 +
                                                  i2]:texedge_idx[i1 * 2 + i2 +
                                                                  1]]

    exmodel = LogisticRegression()
    X = torch.cat(
        (exedge["train"]["edge"], exedge["train"]["edge_neg"])).cpu().numpy()
    Y = np.zeros_like(X[:, 0])
    Y[:exedge["train"]["edge"].shape[1]] = 1
    exmodel.fit(X, Y)
    valexpred = torch.from_numpy(
        exmodel.predict(
            torch.cat((exedge["valid"]["edge"],
                      exedge["valid"]["edge_neg"])).cpu().numpy())).flatten().to(device)
    tstexpred = torch.from_numpy(
        exmodel.predict(
            torch.cat((exedge["test"]["edge"],
                      exedge["test"]["edge_neg"])).cpu().numpy())).flatten().to(device)
    del exedge, texedge, texedge_idx, exmodel, X, Y
    print("nes end", flush=True)

    model = buildmodel(max_x=2, layer1=0).to(device, non_blocking=True)
    opt = Adam(model.parameters(), lr=lr)
    from time import time
    for i in range(num_epoch):
        t0 = time()
        loss = train(num_clu, nnodes, datalist, negdatalist, 5, model, opt,
                     batch_size)
        t1 = time()
        validscore = test(num_clu, nnodes, cld.partptr, nbin, nes, datalist,
                          model, Evaluator(ds_name), "valid", valexpred)
        t2 = time()
        testscore = test(num_clu, nnodes, cld.partptr, nbin, nes, datalist,
                         model, Evaluator(ds_name), "test", tstexpred)
        t3 = time()
        print(
            f"{i}: time {t1-t0:.1f} {t2-t1:.1f} {t3-t2:.1f} {loss} {validscore} {testscore}",
            flush=True)


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--nc", type=int)
args = parser.parse_args()

for nc in [args.nc]:
    for bs in [16, 32, 64]:
        print("ncbs", nc, bs)
        routine(nc, 50, 3e-4, bs)
