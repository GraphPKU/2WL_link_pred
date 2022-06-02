import enum
import os
import os.path as osp
import sys
import argparse
from torch.functional import Tensor, _return_counts
from torch.serialization import validate_cuda_device
import torch_geometric

import scipy.io as sio
import scipy.sparse as ssp
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.nn import Embedding, BCEWithLogitsLoss
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import dropout_adj
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv  # noqa
from torch_scatter import scatter_mean
#from matmul_xy import matmul
import warnings
import time
from torch_geometric.data import Data
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

warnings.filterwarnings("ignore")

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

def node2link(p, q, s):
    if p <= q:
        return s * (s - 1) - (s - p - 2) * (s - p - 1) - 2 * (s - q)
    else:
        return s * (s - 1) - (s - q - 2) * (s - q - 1) - 2 * (s - p) + 1


def node2link_vec(p, q, s):
    adder = (p > q).to(torch.long)
    m1 = torch.max(torch.cat([p.unsqueeze(0), q.unsqueeze(0)], 0), axis=0)[0]
    m2 = torch.min(torch.cat([p.unsqueeze(0), q.unsqueeze(0)], 0), axis=0)[0]
    return s * (s - 1) - (s - m2 - 2) * (s - m2 - 1) - 2 * (s - m1) + adder


def link2node(l, s):
    rev = True if l > l // 2 * 2 else False
    l = l // 2
    tem = s * (s - 1) // 2 - l - 1
    t = int(np.sqrt(2 * tem))
    if tem < t * (t + 1) // 2:
        t -= 1
    p = s - t - 2
    q = s - 1 - tem + t * (t + 1) // 2
    if rev:
        return q, p
    else:
        return p, q


@torch.jit.script
def link2node_vec(l: Tensor, s: int):
    rev = l & 1
    l = l >> 1
    tem = ((s * (s - 1)) >> 1) - l - 1
    t = torch.sqrt((tem << 1).to(torch.float)).to(torch.int)
    t -= (tem < ((t * (t + 1)) >> 1)).to(t.dtype)
    p = s - t - 2
    q = s - 1 - tem + ((t * (t + 1)) >> 1)
    ret = torch.stack((p, q))
    retinx = torch.arange(ret.shape[1])
    return ret[rev, retinx], ret[1 - rev, retinx]


def avg_pool(x, assignment):
    row, col = assignment
    return scatter_mean(x[row], col, dim=0)


def random_split(data, train_size, val_size, test_size):
    n = tailed_size * (tailed_size - 1) // 2
    train_mask = torch.zeros(num_new_node, dtype=torch.bool)
    val_mask = torch.zeros(num_new_node, dtype=torch.bool)
    test_mask = torch.zeros(num_new_node, dtype=torch.bool)
    train_size = train_size // 2
    val_size = val_size // 2
    test_size = test_size // 2
    l1 = torch.arange(num_new_node // 2)[(B == 1)[even_mask]]
    l0 = torch.arange(num_new_node // 2)[(B == 0)[even_mask]]
    l1 = l1[torch.randperm(l1.shape[0])]
    l0 = l0[torch.randperm(l0.shape[0])]
    train_label = torch.cat((l1[:train_size], l0[:train_size]))
    val_label = torch.cat((l1[train_size:train_size + val_size],
                           l0[train_size:train_size + val_size]))
    test_label = torch.cat(
        (l1[train_size + val_size:train_size + val_size + test_size],
         l0[train_size + val_size:train_size + val_size + test_size]))

    train_mask[train_label * 2] = 1
    train_mask[train_label * 2 + 1] = 1
    val_mask[val_label * 2] = 1
    val_mask[val_label * 2 + 1] = 1
    test_mask[test_label * 2] = 1
    test_mask[test_label * 2 + 1] = 1

    return train_mask, val_mask, test_mask


def random_train_mask(train_size, pool):
    train_size = train_size // 2
    n = len(pool)
    train_mask = torch.zeros(num_new_node, dtype=bool)
    l1 = l1_init[torch.randperm(l1_init.shape[0])]
    l0 = l0_init[torch.randperm(l0_init.shape[0])]

    train_label = torch.cat((l1[:train_size], l0[:train_size]))
    train_mask[train_label] = 1
    train_mask[train_label + 1] = 1
    return train_mask


@torch.jit.script
def setmul(a, b, c):
    '''
    shape:
        a : (m)
        b : (m)
        c : (n)
    '''
    m = a.shape[0]
    n = c.shape[0]
    na = a.unsqueeze(1).expand(-1, n).flatten()
    nb = b.unsqueeze(1).expand(-1, n).flatten()
    nc = c.unsqueeze(0).expand(m, -1).flatten()
    return torch.stack((na, nb, nc))

def set_mul(a: Tensor, b: Tensor):
    a = a.reshape(-1, 1)
    b = b.reshape(1, -1)
    a = a.expand(-1, b.shape[1])
    b = b.expand(a.shape[0], -1)
    return torch.cat((a.reshape(-1, 1), b.reshape(-1, 1)), dim=-1)


def to_directed(attr):
    pool = entire_pool[attr[:, 0] != 0] if len(
        attr.shape) > 1 else entire_pool[attr != 0]
    attr0 = torch.zeros(
        (num_new_node0,
         attr.shape[-1]), dtype=attr.dtype) if len(attr.shape) > 1 else torch.zeros(
             (num_new_node0, ), dtype=attr.dtype)
    attr0 = attr0.to(attr.device)
    #print(len(mask))
    vec_e = pool[torch.logical_not((pool & 1).to(torch.bool))].to(attr.device)
    vec_p, vec_q = link2node_vec(vec_e, tailed_size)
    attr0[vec_p * tailed_size + vec_q] = attr[vec_e]
    attr0[vec_q * tailed_size + vec_p] = attr[vec_e]
    return attr0

def double(x, for_index = False):
    if not for_index:
        row, col = x[0].reshape(1, x.shape[1]), x[1].reshape(1, x.shape[1])
        x = torch.cat([row, col, col, row], 0).t()
        x = x.reshape(-1, 2).t()
    else:
        x = x.reshape(1, x.shape[0])
        x = torch.cat([2 * x, 2 * x + 1], 0).t()
        x = x.reshape(-1, 1).t().squeeze()
    return x

def cut_edge(initial_edge_attr, initial_x, initial_edge, A, B_ob, node_mask,
             pruning, device):
    if two_wl_flow == 'local_dir':
        t0 = time.time()

        edge_mask = torch.zeros((initial_edge.shape[1], ), dtype=torch.bool)
        output_A = A.detach().clone()

        t1 = time.time()

        vec_p = initial_edge[0]
        vec_q = initial_edge[1]
        edge_mask[node_mask[node2link_vec(vec_p, vec_q, tailed_size)]] = 1

        edges = entire_pool[node_mask]
        output_x = initial_x.detach().clone().cpu()
        t2 = time.time()

        if True:
            vec_tem = torch.arange(0, len(edges), 2)
            vec_tem = vec_tem[B_ob[edges[vec_tem]] != 0]
            vec_p, vec_q = link2node_vec(edges[vec_tem], tailed_size)
            puni, pcount = torch.unique(vec_p, return_counts=True)
            quni, qcount = torch.unique(vec_q, return_counts=True)
            output_x[puni] -= pcount
            output_x[quni] -= qcount
            output_A[vec_p, vec_q] = 0
            output_A[vec_q, vec_p] = 0

            if torch.any(output_x < 0):
                print("output_x < 0")
                print(output_x[output_x < 0])
                sys.exit(0)

        output_edge_attr = initial_edge_attr.detach().clone()
        t3 = time.time()
        if use_edge_attr:
            output_edge_attr[edges] = 0
        output_x = output_x.to(device)
        t4 = time.time()
        if output_edge_attr != None:
            output_edge_attr = output_edge_attr.to(device)
        output_edge_index = initial_edge[:, ~edge_mask].to(device)
        #output_edge_index_2 = initial_edge_2[:, ~edge_mask_2].to(device)
        t5 = time.time()
        #print(t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4)
        node_mask0 = to_directed(node_mask)
        mask = ~node_mask0 & (A.flatten() != 0)

    return output_edge_attr, output_x, output_edge_index, output_A

def init_edge_attr(n, A):
    device = A.device
    C = torch.eye(n).to(device)
    return torch.cat([C,A],-1)

def mataggr(A, h, g):
    '''
    A (n, n, d). n is number of node, d is latent dimension
    h, g are mlp
    '''
    B = h(A)
    #C = f(A)
    n, d = A.shape[0], A.shape[1]
    vec_p = (torch.sum(B, dim=1, keepdim=True)).expand(-1, n, -1)
    vec_q = (torch.sum(B, dim=0, keepdim=True)).expand(n, -1, -1)
    D = torch.cat([A, vec_p, vec_q], -1)
    return g(D)

class Seq(torch.nn.Module):
    def __init__(self, modlist):
        super().__init__()
        self.modlist = torch.nn.ModuleList(modlist)

    def forward(self, *args, **kwargs):
        out = self.modlist[0](*args, **kwargs)
        for i in range(1, len(self.modlist)):
            out = self.modlist[i](out)
        return out

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', type=str, default="USAir")
parser.add_argument('--dp1', type=float, default=0.0)
parser.add_argument('--dp2', type=float, default=0.0)
parser.add_argument('--dp3', type=float, default=0.0)
parser.add_argument('--layer_1', default=2)
parser.add_argument('--layer_2', default=1)
parser.add_argument('--hidden_1', type=int, default=64)
parser.add_argument('--hidden_2', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--hf', action="store_true", help="Half float")

parser.add_argument('--pl', type=bool, default=True)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--path', type=str, default="Opt/")
parser.add_argument('--test', action="store_true")
parser.add_argument('--check', action="store_true")
parser.add_argument('--use_best', action="store_true")
args = parser.parse_args()

if True:
    #np.random.seed(100)
    #torch.manual_seed(100)
    datatype = torch.float32
    torch.set_default_dtype(datatype)
    path = osp.join('dataset', 'Cora')
    datasets = Planetoid(path, 'Cora', transform=T.NormalizeFeatures())
    data = datasets[0]

    dataset = args.dataset
    print(dataset)

    two_wl_flow = 'local_dir'  # 'global', 'local_k', 'local', 'local_undir', 'local_dir'

    node_undirect = False
    edge_undirect = False

    if dataset == 'USAir':
        f = open("./dataset/USAir.txt")
        line = f.readlines()
        f.close()
        usair_edge_index = torch.zeros((2, len(line)), dtype=torch.long)
        edge_feature = torch.zeros(len(line, ), dtype=datatype)
        for i in range(len(line)):
            usair_edge_index[0, i] = int(line[i][:7])
            usair_edge_index[1, i] = int(line[i][7:14])
            edge_feature[i] = float(line[i][14:24])
        f.close()

        undirect = True
        use_node_feature = False
        use_edge_feature = True
        use_edge_exist = True
        use_node_degree = True
        use_edge_dist = False
        use_node_attr = use_node_feature | use_node_degree
        use_edge_attr = use_edge_feature | use_edge_exist
        batch_size = 500
        val_size = 212
        test_size = 424
        latent_size = 24
        embedding_size = 16
        # num_layer = 4

        tailed_size = int(usair_edge_index.max())
        data.edge_index = usair_edge_index - 1
        data.edge_attr = None
        data.x = torch.zeros(tailed_size)
        compute_new_edge = False
        compute_all_edge = False
        compute_new_D = True
        remove_edge = False

    if dataset == 'Celegans':
        f = open("./dataset/Cele.txt")
        line = f.readlines()
        f.close()
        edge_index = torch.zeros((2, len(line)), dtype=torch.long)
        for i in range(len(line)):
            if line[i][1] == ' ':
                edge_index[0, i] = int(line[i][:1])
                edge_index[1, i] = int(line[i][2:5])
            elif line[i][2] == ' ':
                edge_index[0, i] = int(line[i][:2])
                edge_index[1, i] = int(line[i][3:6])
            else:
                edge_index[0, i] = int(line[i][:3])
                edge_index[1, i] = int(line[i][4:7])
        f.close()

        undirect = True
        use_node_feature = False
        use_edge_feature = False
        use_edge_exist = True
        use_node_degree = True
        use_node_attr = use_node_feature | use_node_degree
        use_edge_attr = use_edge_feature | use_edge_exist
        batch_size = 450
        val_size = 236
        test_size = 472
        latent_size = 32
        embedding_size = 16

        tailed_size = int(edge_index.max()) + 1
        data.edge_index = edge_index
        data.edge_attr = None
        data.x = torch.zeros(tailed_size)
        compute_new_D = True
        remove_edge = False

    if dataset == 'Cora':
        undirect = False
        use_node_feature = True
        use_edge_feature = False
        use_edge_exist = True
        use_node_degree = False
        use_node_attr = use_node_feature | use_node_degree
        use_edge_attr = use_edge_feature | use_edge_exist
        batch_size = 1000
        val_size = 526
        test_size = 1054
        latent_size = 64
        embedding_size = 16

        tailed_size = len(data.x)
        #tailed_size = 500
        #data = tail(data, tailed_size)
        data.feature = data.x
        data.x = torch.zeros(tailed_size)
        compute_new_edge = False
        compute_all_edge = False
        remove_edge = False

    if dataset == 'Citeseer':
        path = osp.join('dataset', dataset)
        datasets = Planetoid(path, dataset, transform=T.NormalizeFeatures())
        data = datasets[0]
        print(data)
        undirect = False
        use_node_feature = True
        use_edge_feature = False
        use_edge_exist = True
        use_node_degree = False
        use_node_attr = use_node_feature | use_node_degree
        use_edge_attr = use_edge_feature | use_edge_exist
        batch_size = 1000
        val_size = 455
        test_size = 910
        latent_size = 64
        embedding_size = 16

        tailed_size = len(data.x)
        #tailed_size = 500
        #data = tail(data, tailed_size)
        data.feature = data.x
        data.x = torch.zeros(tailed_size)
        compute_new_edge = False
        compute_all_edge = False
        remove_edge = False

    if dataset == 'Pubmed':
        path = osp.join('dataset', dataset)
        datasets = Planetoid(path, dataset, transform=T.NormalizeFeatures())
        data = datasets[0]
        undirect = False
        use_node_feature = False
        use_edge_feature = False
        use_edge_exist = True
        use_node_degree = True
        use_node_attr = use_node_feature | use_node_degree
        use_edge_attr = use_edge_feature | use_edge_exist
        batch_size = 1000
        val_size = 526
        test_size = 1054
        latent_size = 64
        embedding_size = 16

        tailed_size = len(data.x)
        #tailed_size = 500
        #data = tail(data, tailed_size)
        data.feature = data.x
        data.x = torch.zeros(tailed_size)
        compute_new_edge = False
        compute_all_edge = False
        remove_edge = False

    if dataset == 'NS':
        f = open("./dataset/NS.txt")
        line = f.readlines()
        f.close()
        edge_index = torch.zeros((2, len(line)), dtype=torch.long)
        edge_feature = torch.zeros(len(line, ), dtype=datatype)
        for i in range(len(line)):
            p = 0
            while line[i][p] != ' ':
                p += 1
            q = p + 1
            while line[i][q] != ' ':
                q += 1
            edge_index[0, i] = int(line[i][0:p]) - 1
            edge_index[1, i] = int(line[i][p:q]) - 1
            edge_feature[i] = float(line[i][q:])
        f.close()

        undirect = True
        use_node_feature = False
        use_edge_feature = False
        use_edge_exist = True
        use_node_degree = True
        use_node_attr = use_node_feature | use_node_degree
        use_edge_attr = use_edge_feature | use_edge_exist
        batch_size = 500
        val_size = 274
        test_size = 548
        latent_size = 64
        embedding_size = 16

        tailed_size = int(edge_index.max()) + 1
        data.edge_index = edge_index
        data.edge_attr = None
        data.x = torch.zeros(tailed_size)
        compute_new_edge = True
        compute_all_edge = False
        compute_new_D = True
        remove_edge = False

    if dataset == 'Power':
        f = open("./dataset/Power.txt")
        line = f.readlines()
        f.close()
        edge_index = torch.zeros((2, len(line)), dtype=int)
        edge_feature = torch.zeros(len(line, ), dtype=float)

        for i in range(len(line)):
            p = 0
            while line[i][p] != '\t': p += 1
            edge_index[0, i] = int(line[i][:p])
            edge_index[1, i] = int(line[i][p + 1:])

        undirect = True
        use_node_feature = False
        use_edge_feature = False
        use_edge_exist = True
        use_node_degree = True
        use_node_attr = use_node_feature | use_node_degree
        use_edge_attr = use_edge_feature | use_edge_exist
        batch_size = 1300
        val_size = 658
        test_size = 1316
        latent_size = 32
        embedding_size = 16
        # num_layer = 4

        tailed_size = int(edge_index.max()) + 1
        # data = tail(data, tailed_size)
        # data.edge_index = np.array([[0,1,1,3,3,1,2,3,4,5],[1,2,3,4,5,0,1,1,3,3]], dtype = int)
        data.edge_index = edge_index
        data.edge_attr = None
        data.x = np.zeros(tailed_size)
        compute_new_edge = False
        compute_all_edge = False
        remove_edge = False

    if dataset == 'PB':
        f = open("./dataset/PB.txt")
        line = f.readlines()
        f.close()
        edge_index = torch.zeros((2, len(line)), dtype=int)
        edge_feature = torch.zeros(len(line, ), dtype=float)

        for i in range(len(line)):
            p = 0
            while line[i][p] != '\t': p += 1
            edge_index[0, i] = int(line[i][:p]) - 1
            edge_index[1, i] = int(line[i][p + 1:]) - 1

        undirect = True
        use_node_feature = False
        use_edge_feature = False
        use_edge_exist = True
        use_node_degree = True
        use_node_attr = use_node_feature | use_node_degree
        use_edge_attr = use_edge_feature | use_edge_exist
        batch_size = 1000
        val_size = 1902
        test_size = 3804
        latent_size = 96
        embedding_size = 32
        # num_layer = 4

        tailed_size = int(edge_index.max()) + 1
        # data = tail(data, tailed_size)
        # data.edge_index = np.array([[0,1,1,3,3,1,2,3,4,5],[1,2,3,4,5,0,1,1,3,3]], dtype = int)
        data.edge_index = edge_index
        data.edge_attr = None
        data.x = np.zeros(tailed_size)
        compute_new_edge = False
        compute_all_edge = False
        remove_edge = False

    if dataset == 'Router':
        f = open("./dataset/Router.txt")
        line = f.readlines()
        f.close()
        edge_index = torch.zeros((2, len(line)), dtype=int)
        edge_feature = torch.zeros(len(line, ), dtype=float)

        for i in range(len(line)):
            p = 0
            while line[i][p] != '\t': p += 1
            edge_index[0, i] = int(line[i][:p])
            edge_index[1, i] = int(line[i][p + 1:])

        undirect = True
        use_node_feature = False
        use_edge_feature = False
        use_edge_exist = True
        use_node_degree = True
        use_node_attr = use_node_feature | use_node_degree
        use_edge_attr = use_edge_feature | use_edge_exist
        batch_size = 1200
        val_size = 624
        test_size = 1248
        latent_size = 32
        embedding_size = 16
        # num_layer = 4

        tailed_size = int(edge_index.max()) + 1
        # data = tail(data, tailed_size)
        # data.edge_index = np.array([[0,1,1,3,3,1,2,3,4,5],[1,2,3,4,5,0,1,1,3,3]], dtype = int)
        data.edge_index = edge_index
        data.edge_attr = None
        data.x = np.zeros(tailed_size)
        compute_new_edge = False
        compute_all_edge = False
        remove_edge = False

    if dataset == 'Yeast':
        f = open("./dataset/Yeast.txt")
        line = f.readlines()
        f.close()
        edge_index = torch.zeros((2, len(line)), dtype=int)
        edge_feature = torch.zeros(len(line, ), dtype=float)

        for i in range(len(line)):
            p = 0
            while line[i][p] != '\t': p += 1
            edge_index[0, i] = int(line[i][:p])
            edge_index[1, i] = int(line[i][p + 1:])

        undirect = True
        use_node_feature = False
        use_edge_feature = False
        use_edge_exist = True
        use_node_degree = True
        use_node_attr = use_node_feature | use_node_degree
        use_edge_attr = use_edge_feature | use_edge_exist
        batch_size = 2400
        val_size = 1168
        test_size = 2336
        latent_size = 64
        embedding_size = 16
        # num_layer = 4

        tailed_size = int(edge_index.max()) + 1
        # data = tail(data, tailed_size)
        # data.edge_index = np.array([[0,1,1,3,3,1,2,3,4,5],[1,2,3,4,5,0,1,1,3,3]], dtype = int)
        data.edge_index = edge_index
        data.edge_attr = None
        data.x = np.zeros(tailed_size)
        compute_new_edge = False
        compute_all_edge = False
        remove_edge = False

    if dataset == 'Ecoli':
        data = sio.loadmat('dataset/{}.mat'.format(dataset))
        net = data['net']
        net_triu = ssp.triu(net, k=1)
        row, col, _ = ssp.find(net_triu)
        edge_index = torch.stack(
            (torch.tensor(row).flatten(), torch.tensor(col).flatten())).to(torch.long)
        # edge_index = to_undirected(edge_index)
        data = Data(edge_index=edge_index)
        edge_feature = torch.zeros(edge_index.shape[1], dtype=float)
        undirect = True
        use_node_feature = False
        use_edge_feature = False
        use_edge_exist = True
        use_node_degree = True
        use_node_attr = use_node_feature | use_node_degree
        use_edge_attr = use_edge_feature | use_edge_exist
        batch_size = 2500
        val_size = 1466
        test_size = 2932
        latent_size = 64
        embedding_size = 16
        # num_layer = 4

        tailed_size = int(edge_index.max()) + 1
        # data = tail(data, tailed_size)
        # data.edge_index = np.array([[0,1,1,3,3,1,2,3,4,5],[1,2,3,4,5,0,1,1,3,3]], dtype = int)
        data.edge_index = edge_index
        data.edge_attr = None
        data.x = np.zeros(tailed_size)
        compute_new_edge = False
        compute_all_edge = False
        remove_edge = False

    if dataset.startswith('ogbl'):
        datasets = PygLinkPropPredDataset(name=dataset)
        data = datasets[0]
        print(data.edge_index.shape)
        tailed_size = data.num_nodes
        use_node_feature = False
        use_edge_feature = False
        use_edge_exist = True
        use_node_degree = True
        use_node_attr = use_node_feature | use_node_degree
        use_edge_attr = use_edge_feature | use_edge_exist
        latent_size = 64
        embedding_size = 16
        split_edge = datasets.get_edge_split()
        #import pdb
        #pdb.set_trace()

        split_edge['train']['edge'] = split_edge['train']['edge'].t()
        split_edge['valid']['edge'] = split_edge['valid']['edge'].t()
        split_edge['valid']['edge_neg'] = split_edge['valid']['edge_neg'].t()
        split_edge['test']['edge'] = split_edge['test']['edge'].t()
        split_edge['test']['edge_neg'] = split_edge['test']['edge_neg'].t()
        train_size, val_size, test_size = 2 * split_edge['train']['edge'].shape[1], \
                                          2 * split_edge['valid']['edge'].shape[1], \
                                          2 * split_edge['test']['edge'].shape[1]
        batch_size = test_size
        num_edge = train_size + val_size + test_size
        split_edge['train']['edge'] = double(split_edge['train']['edge'])
        split_edge['valid']['edge'] = double(split_edge['valid']['edge'])
        split_edge['test']['edge'] = double(split_edge['test']['edge'])
        data.edge_index = torch.cat([split_edge['train']['edge'], split_edge['valid']['edge'], split_edge['test']['edge']], 1)
        undirect = False
        remove_edge = False

    latent_size = args.hidden_1
    embedding_size = args.hidden_1

    if undirect:
        edge_index_r = data.edge_index[[1, 0]]
        data.edge_index = torch.cat((data.edge_index, edge_index_r), axis=1)
        if use_edge_feature:
            edge_feature = torch.cat((edge_feature, edge_feature), axis=0)

best_params = {
        'Celegans': {
            'dp1': 0.0,
            'dp2': 0.2,
            'dp3': 0.3,
            'hidden_dim_1wl': 32,
            'hidden_dim_2wl': 24,
            'lr': 0.01
        },
        'USAir': {
            'dp1': 0.0,
            'dp2': 0.3,
            'dp3': 0.0,
            'hidden_dim_1wl': 24,
            'hidden_dim_2wl': 24,
            'lr': 0.01
        },
        'PB': {
            'dp1': 0.0,
            'dp2': 0.0,
            'dp3': 0.3,
            'hidden_dim_1wl': 24,
            'hidden_dim_2wl': 24,
            'lr': 0.01
        },
        'NS': {
            'dp1': 0.0,
            'dp2': 0.0,
            'dp3': 0.3,
            'hidden_dim_1wl': 16,
            'hidden_dim_2wl': 16,
            'lr': 0.01
        },
        'Ecoli': {
            'dp1': 0.0,
            'dp2': 0.0,
            'dp3': 0.3,
            'hidden_dim_1wl': 32,
            'hidden_dim_2wl': 24,
            'lr': 0.01
        },
        'Router': {
            'dp1': 0.0,
            'dp2': 0.3,
            'dp3': 0.0,
            'hidden_dim_1wl': 16,
            'hidden_dim_2wl': 16,
            'lr': 0.01
        },
        'Power': {
            'dp1': 0.0,
            'dp2': 0.0,
            'dp3': 0.3,
            'hidden_dim_1wl': 16,
            'hidden_dim_2wl': 16,
            'lr': 0.01
        },
        'Yeast': {
            'dp1': 0.0,
            'dp2': 0.0,
            'dp3': 0.3,
            'hidden_dim_1wl': 32,
            'hidden_dim_2wl': 24,
            'lr': 0.01
        },
        'Cora': {
            'dp1': 0.0,
            'dp2': 0.1,
            'dp3': 0.3,
            'hidden_dim_1wl': 32,
            'hidden_dim_2wl': 24,
            'lr': 0.001
        },
        'Citeseer': {
            'dp1': 0.1,
            'dp2': 0.2,
            'dp3': 0.3,
            'hidden_dim_1wl': 32,
            'hidden_dim_2wl': 24,
            'lr': 0.01
        }}
params = best_params[dataset]
if True:
    if True:
        p = 0
        q = 0
        num_new_node = tailed_size * (tailed_size - 1)
        num_new_node0 = tailed_size * tailed_size
        entire_pool = torch.arange(num_new_node)
        initial_edge = data.edge_index
        #num_new_edge = len(data.x) * (len(data.x) - 1) * (len(data.x) - 2)
        x_new = torch.zeros(tailed_size, dtype=torch.int)
        y_new = torch.zeros(num_new_node, dtype=torch.bool)
        y_new0 = torch.zeros(num_new_node0, dtype=torch.bool)
        deg = torch.zeros(tailed_size, dtype=torch.long)
        A = torch.zeros((tailed_size, tailed_size), dtype=torch.bool)
        B = torch.zeros((num_new_node, ), dtype=torch.bool)
        t = 0

        one_mask = torch.ones((num_new_node, ), dtype=torch.bool)
        if not node_undirect:
            even_mask = torch.zeros((num_new_node, ), dtype=torch.bool)
            even_idx = torch.arange(0, num_new_node, 2)
            even_mask[even_idx] = True
            odd_mask = torch.logical_not(even_mask)

        #print('!!')
        vec_p = data.edge_index[0]
        vec_q = data.edge_index[1]
        A[vec_p, vec_q] = 1
        B[node2link_vec(vec_p, vec_q, tailed_size)] = 1


        row = torch.ones([1, tailed_size], dtype=torch.long)
        col = torch.ones([tailed_size, 1], dtype=torch.long)
        mul = torch.arange(tailed_size)
        row = mul.unsqueeze(1) * row
        col = col * mul.unsqueeze(0)

        all_edges = torch.cat([row.unsqueeze(0), col.unsqueeze(0)], 0)
        all_edges_res = all_edges.reshape(2, -1)

        tidx = torch.arange(tailed_size)
        A[tidx, tidx] = 2

        if remove_edge:
            flag = torch.zeros((len(data.x), len(data.x)), dtype=bool)
            mask = torch.ones((data.edge_index.shape[1], ), dtype=bool)
            vec_p, vec_q = data.edge_index
            mask[flag[vec_p, vec_q]] = 0
            flag[vec_p, vec_q] = 1
            data.edge_index = data.edge_index[:, mask]


        vec_p, vec_q = data.edge_index
        tmask = vec_p < vec_q
        vec_p = vec_p[tmask]
        vec_q = vec_q[tmask]
        tidx = node2link_vec(vec_p, vec_q, tailed_size)
        if use_edge_feature:
            if 'edge_attr' not in data.keys:
                data.edge_attr = torch.zeros(
                    (num_new_node,
                     1), dtype = datatype) if len(edge_feature.shape) == 1 else torch.zeros(
                         (num_new_node, edge_feature.shape[1]), dtype = datatype)

            data.edge_attr[tidx] = edge_feature[tmask].unsqueeze(1)
            data.edge_attr[tidx + 1] = edge_feature[tmask].unsqueeze(1)
        y_new[tidx] = 1
        y_new[tidx + 1] = 1
        y_new0[vec_p * tailed_size + vec_q] = 1
        y_new0[vec_q * tailed_size + vec_p] = 1
        puni, pcount = torch.unique(vec_p, return_counts=True)
        quni, qcount = torch.unique(vec_q, return_counts=True)
        deg[puni] += pcount
        deg[quni] += qcount


        if use_edge_exist:
            if 'edge_attr' not in data.keys:
                data.edge_attr = B.unsqueeze(1)
            else:
                data.edge_attr = torch.cat([data.edge_attr, B.unsqueeze(1)], 1)

    x_new[:tailed_size] = deg[:tailed_size]

    # data.feature = torch.tensor(data.x) if use_node_feature else None
    data.edge_attr = torch.tensor(data.edge_attr,
                                  dtype=datatype) if use_edge_attr else None

    data.x = torch.tensor(x_new, dtype=torch.long)
    data.edge_index = torch.tensor(data.edge_index, dtype=torch.long)
    data.y = torch.tensor(y_new0, dtype=torch.long)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    initial_edge = data.edge_index
    #initial_edge_2 = data.edge_index_2
    initial_x = data.x.detach().clone()
    initial_edge_attr = data.edge_attr

    for i in range(10):

        if dataset.startswith('ogbl'):
            p, q = torch.cat([split_edge['valid']['edge'], split_edge['valid']['edge_neg']], 1)
            val_pool = node2link_vec(p, q, tailed_size)
            p, q = torch.cat([split_edge['test']['edge'], split_edge['test']['edge_neg']], 1)
            test_pool = node2link_vec(p, q, tailed_size)
            train_mask = torch.zeros([num_new_node, ], dtype=torch.bool)
            val_mask = torch.zeros([num_new_node,], dtype=torch.bool)
            val_mask[val_pool] = 1
            test_mask = torch.zeros([num_new_node, ], dtype=torch.bool)
            test_mask[test_pool] = 1
        else:
            train_mask, val_mask, test_mask = random_split(data, 0, val_size,
                                                       test_size)

        data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
        data.val_mask = torch.tensor(val_mask, dtype=torch.bool)
        data.test_mask = torch.tensor(test_mask, dtype=torch.bool)
        train_pool = torch.arange(num_new_node)[~val_mask & ~test_mask]

        l1_init = torch.arange(num_new_node)[~val_mask & ~test_mask & y_new & even_mask]
        l0_init = torch.arange(num_new_node)[~val_mask & ~test_mask & ~y_new & even_mask]
        # node_mask = train_mask | val_mask | test_mask

        # print(data.x.sum())
        node_mask = val_mask | test_mask
        B_train = (B == 1) & ~node_mask

        if use_node_feature:
            node_feature_size = 1 if len(
                data.feature.shape) == 1 else data.feature.shape[1]
        if use_edge_attr:
            edge_attr_size = data.edge_attr.shape[1]+1

        device = torch.device('cuda')
        initial_edge = initial_edge.to(device)
        initial_x = initial_x.to(device)
        x_max = initial_x.max()
        if use_edge_attr:
            initial_edge_attr = initial_edge_attr.to(device)

        #print('!!!')
        class Net(torch.nn.Module):
            def __init__(self, embedding_size=20, latent_size_1=20, latent_size_2=20, dp1=0.0,dp2=0.0,dp3=0.0):
                super(Net, self).__init__()
                use_affine = False

                block_fn = lambda insize, outsize: Seq([
                    torch.nn.Linear(insize, outsize),
                    nn.LayerNorm(outsize, elementwise_affine=use_affine),
                    torch.nn.Dropout(p=dp3, inplace=True),
                    torch.nn.ReLU(inplace=True)
                ])

                block_lfn = lambda insize, outsize: Seq([
                    torch.nn.Linear(insize, outsize),
                    nn.LayerNorm(outsize, elementwise_affine=use_affine),
                    torch.nn.Dropout(p=dp3, inplace=True),
                ])

                Convs = lambda insize, outsize: Seq([
                    SAGEConv(insize, outsize),
                    nn.LayerNorm(outsize, elementwise_affine=use_affine),
                    torch.nn.Dropout(p=dp2, inplace=True),
                ])

                input_node_size = 0
                if use_node_feature: input_node_size += node_feature_size
                if use_node_degree: input_node_size += embedding_size

                self.embedding = torch.nn.Sequential(torch.nn.Embedding(x_max + 1, latent_size),
                                               torch.nn.Dropout(p=dp1))
                self.embedding = Embedding(x_max + 1, embedding_size)

                self.nconv1 = Convs(input_node_size, latent_size)
                self.nconv2 = Convs(latent_size, latent_size)


                input_edge_size = latent_size
                if use_edge_attr: input_edge_size += edge_attr_size

                self.h_1 = block_fn(input_edge_size + 1, latent_size_2)

                self.g_1 = block_fn(latent_size_2 * 2 + input_edge_size + 1, latent_size_2)

                self.lin_dir = torch.nn.Linear(latent_size_2, 1)
                #import pdb
                #pdb.set_trace()
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        print(m)
                        nn.init.xavier_normal_(m.weight)

            def forward(self, edge_mask, use_logsoftmax=False):
                x, edge_index = data.x, data.edge_index
                n = x.shape[0]
                if use_node_feature: feature = data.feature
                if use_edge_attr: edge_attr = data.edge_attr
                edge_weight = None

                if use_node_degree: x = self.embedding(x)

                if use_node_feature:
                    feature = feature.unsqueeze(1) if len(
                        feature.shape) == 1 else feature
                    x = torch.cat([x, feature.to(x.dtype)],
                                  1) if use_node_degree else feature.to(
                                      datatype)
                # x = F.relu(self.nlin1(x))

                x = F.relu(self.nconv1(x, edge_index))

                # x = F.dropout(x, training=self.training)
                x = self.nconv2(x, edge_index)

                x = x[all_edges_res[0]] * x[all_edges_res[1]]

                #if use_edge_attr: x = torch.cat([x, edge_attr], 1)
                #x = F.relu(self.conv1_0(x, data.edge_index_2))
                x = x.reshape(tailed_size, tailed_size, -1)

                if use_edge_attr:
                    edge_attr = edge_attr.reshape(tailed_size, tailed_size, -1)
                    x = torch.cat([x, edge_attr], -1)
                    x = torch.cat([x, torch.eye(tailed_size).unsqueeze(-1).to(device)], -1)

                eim = torch.zeros((n * n,), device=x.device)
                eim[edge_index[0] * n + edge_index[1]] = 1
                eim = eim.reshape(n, n, 1)
                x = torch.cat((x, eim), dim=-1)

                x = mataggr(x, self.h_1, self.g_1)

                x = (x * x.permute(1, 0, 2)).reshape(tailed_size * tailed_size, -1)

                x = x[edge_mask]

                x = self.lin_dir(x)

                return x

        node_mask = val_mask | test_mask
        initial_train_edge_attr, initial_train_x, initial_train_edge, A_train = cut_edge(
            initial_edge_attr, initial_x, initial_edge, A, B, node_mask, False,
            device)  # , edge2)
        initial_val_edge_attr, initial_val_x, initial_val_edge, A_val = cut_edge(
            initial_edge_attr, initial_x, initial_edge, A, B, node_mask, True,
            device)  # , edge2)
        initial_test_edge_attr, initial_test_x, initial_test_edge, A_test = cut_edge(
            initial_edge_attr, initial_x, initial_edge, A, B, test_mask, True,
            device)  # , edge2)

        model, data = Net(embedding_size, params['hidden_dim_1wl'], params['hidden_dim_2wl'], params['dp1'], params['dp2'], params['dp3']).to(device), data.to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=params['lr'])

        Criteria = BCEWithLogitsLoss()

        #for n, i in model.named_parameters():
        #    print(n)
        #print(model.nconv1.lin_l.weight)

        def train():
            model.train()
            optimizer.zero_grad()
            output = model(train_mask0).view(-1)
            loss = Criteria(output, data.y[train_mask0].to(datatype))
            loss.backward()
            optimizer.step()
            return loss

        def initialize_val():
            data.x = initial_val_x.detach().clone().to(device)
            if initial_edge_attr != None:
                data.edge_attr = initial_val_edge_attr.detach().clone().to(device)
            data.edge_index = initial_val_edge.detach().clone().to(device)
            data.A = A_val
            #data.D = D_val.detach().clone().to(device)

        def initialize_test():
            data.x = initial_test_x.detach().clone().to(device)
            if initial_edge_attr != None:
                data.edge_attr = initial_test_edge_attr.detach().clone().to(device)
            data.edge_index = initial_test_edge.detach().clone().to(device)
            data.A = A_test
            #data.D = D_test.detach().clone().to(device)

        @torch.no_grad()
        def test():
            model.eval()
            aucs = []

            # print('train')
            logits = model(train_mask0)
            #print(logits)
            sig = logits.sigmoid().cpu().numpy()
            label = data.y[train_mask0].cpu().numpy()
            mask1 = (B0 == 1)[train_mask0]
            train1, train0 = sig[mask1].mean(), sig[~mask1].mean()

            aucs.append(roc_auc_score(label, sig))

            # print('val')
            initialize_val()
            if use_edge_exist:
                data.edge_attr = to_directed(data.edge_attr).to(device)

            logits = model(val_mask0)
            sig = logits.sigmoid().squeeze().cpu().numpy()
            label = data.y[val_mask0].to(torch.long).cpu().numpy()
            # mask1 = (B0 == 1)[val_mask0]
            # print(sig[mask1].mean(), sig[~mask1].mean())
            if dataset.startswith('ogbl'):
                result = evaluate_hits(sig[label], sig[~label], K=20)['Hits@20']
                aucs.append(result)
            else:
                aucs.append(roc_auc_score(label, sig))

            # logits = model(one_mask).sigmoid()
            # print(logits[(B==1)&~val_mask].mean().cpu().numpy(), logits[(B==0)&~val_mask].mean().cpu().numpy())

            # print('test')
            initialize_test()
            if use_edge_exist:
                data.edge_attr = to_directed(data.edge_attr).to(device)

            logits = model(test_mask0)
            sig = logits.sigmoid().squeeze().cpu().numpy()
            label = data.y[test_mask0].to(torch.long).cpu().numpy()
            mask1 = (B0 == 1)[test_mask0]
            result = evaluate_hits(sig[label], sig[~label], K=20)['Hits@20']
            if dataset.startswith('ogbl'):
                result = evaluate_hits(sig[label], sig[~label], K=20)['Hits@20']
                aucs.append(result)
            else:
                aucs.append(roc_auc_score(label, sig))
            aucs.append(sig[mask1].mean())
            aucs.append(sig[~mask1].mean())

            # logits = model(one_mask).sigmoid()
            # print(logits[(B==1)&~test_mask].mean().cpu().numpy(), logits[(B==0)&~test_mask].mean().cpu().numpy())

            return aucs

        if True:
            # model.load_state_dict(torch.load('./model/model_1221109.pt'))
            best_val_auc = test_auc = 0
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
            val_mask0, test_mask0, B0 = to_directed(
                val_mask), to_directed(test_mask), to_directed(B)
            early_stop = 0
            for epoch in range(1, 10001):
                t00 = time.time()
                if True:
                    train_mask = random_train_mask(batch_size, train_pool)
                    train_mask0 = to_directed(train_mask)
                data.edge_attr, data.x, data.edge_index, data.A = cut_edge(
                    initial_train_edge_attr, initial_train_x, initial_train_edge,
                    A_train, B_train, train_mask, True, device)

                if use_edge_exist:
                    data.edge_attr = to_directed(data.edge_attr).to(device)

                data.train_mask = train_mask

                t0 = time.time()
                loss = train()
                t1 = time.time()
                if (epoch % 1==0):
                    early_stop += 1
                    training_auc, val_auc, tmp_test_auc, pred1, pred0 = test()
                    t2 = time.time()
                    if (val_auc > best_val_auc) and (epoch > 30):
                        best_val_auc = val_auc
                        test_auc = tmp_test_auc
                        early_stop = 0
                        print('Best!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    log1 = 'Epoch: {:03d}, Time:{:.2f}, Loss: {:.4f}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
                    # print(log1.format(epoch, training_auc, train_auc, best_val_auc, test_auc))
                    print(log1.format(epoch, t2-t00, loss, training_auc, val_auc,
                                      tmp_test_auc))

                if early_stop > 600:
                    break

            with open(f'./records/{dataset}_auc_record.txt', 'a') as f:
                f.write(
                    'AUC:' + str(round(test_auc, 4)) + '   ' + 'Time:' + str(
                        round(t1-t0, 4)) + '   ' + '\n')