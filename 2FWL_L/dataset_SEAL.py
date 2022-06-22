'''
使用方法
train_pos, train_neg, test_pos, test_neg = load({
            "data_name":
            "Celegans",
            "train_name":
            None,
            "test_name":
            None,
            "test_ratio":
            0.1,
            "max_train_num":
            1000000000
        })
train_pos, train_neg, test_pos, test_neg 分别为训练集中正边，负边，测试集中正边，负边
'''
import os.path as osp
import numpy as np
import torch
import scipy.io as sio
import scipy.sparse as ssp
from torch.functional import split
from torch_geometric.utils import negative_sampling, add_self_loops
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from ogb.linkproppred import PygLinkPropPredDataset
from tqdm import tqdm
from utils import random_split_edges

def to_undirected(edge_index):
    n = edge_index.max()+1
    A = torch.zeros((n,n), dtype = torch.bool)
    index = torch.empty((2, edge_index.shape[1]//2), dtype = torch.long)
    t = 0
    for i in tqdm(range(edge_index.shape[1])):
        p, q = edge_index[0,i], edge_index[1,i]
        if not A[q, p]:
            index[0, t] = p
            index[1, t] = q
            A[p, q] = 1
            t += 1
    return index

def tail(edge_index, s):
    mask = (edge_index[0, :] < s) & (edge_index[1, :] < s)
    return edge_index[:, mask]

def load(args):
    # check whether train and test links are provided
    if args["data_name"].startswith('ogbl'):
        dataset = PygLinkPropPredDataset(name=args['data_name'])
        data = dataset[0]

    elif args["data_name"] in ["Cora", "Citeseer", "Pubmed"]:
        dataset = Planetoid(osp.join('dataset', args["data_name"]), args["data_name"],
                         transform=T.NormalizeFeatures())
        data = dataset[0]
        data.edge_index = to_undirected(data.edge_index)
        neg_pool_max = True
    else:
        train_pos, test_pos = None, None
        if args["train_name"] is not None:
            train_idx = np.loadtxt('data_SEAL/{}'.format(args["train_name"]),
                                   dtype=int)
            train_pos = (train_idx[:, 0], train_idx[:, 1])
        if args["test_name"] is not None:
            test_idx = np.loadtxt('data_SEAL/{}'.format(args["test_name"]),
                                  dtype=int)
            test_pos = (test_idx[:, 0], test_idx[:, 1])

        if args["data_name"] is not None:
            data = sio.loadmat('data_SEAL/Celegans.mat')
            data = sio.loadmat('data_SEAL/{}.mat'.format(args["data_name"]))
            net = data['net']
            if 'group' in data:
                # load node attributes (here a.k.a. node classes)
                attributes = data['group'].toarray().astype('float32')
                print('attributes')
                print(attributes.shape)
            else:
                attributes = None
        else:  # build network from train links
            assert (args["train_name"]
                    is not None), "must provide train links if not using .mat"
            if args["train_name"].endswith('_train.txt'):
                args["data_name"] = args["train_name"][:-10]
            else:
                args["data_name"] = args["train_name"].split('.')[0]
            max_idx = np.max(train_idx)
            if args["test_name"] is not None:
                max_idx = max(max_idx, np.max(test_idx))
            net = ssp.csc_matrix(
                (np.ones(len(train_idx)), (train_idx[:, 0], train_idx[:, 1])),
                shape=(max_idx + 1, max_idx + 1))
            net[train_idx[:, 1], train_idx[:, 0]] = 1  # add symmetric edges
            net[np.arange(max_idx + 1),
                np.arange(max_idx + 1)] = 0  # remove self-loops
        # get upper triangular matrix
        net_triu = ssp.triu(net, k=1)
        row, col, _ = ssp.find(net_triu)
        edge_index = torch.stack(
            (torch.tensor(row).flatten(), torch.tensor(col).flatten())).to(torch.long)
        #edge_index = to_undirected(edge_index)
        data = Data(edge_index=edge_index)
        neg_pool_max = False

    print(data.edge_index.shape)
    if args["data_name"].startswith('ogbl'):
        split_edge = dataset.get_edge_split()
        split_edge['train']['edge_neg'] = negative_sampling(
            data.edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=split_edge['train']['edge'].shape[0])
        split_edge['train']['edge'] = split_edge['train']['edge'].t()
        split_edge['valid']['edge'] = split_edge['valid']['edge'].t()
        split_edge['valid']['edge_neg'] = split_edge['valid']['edge_neg'].t()
        split_edge['test']['edge'] = split_edge['test']['edge'].t()
        split_edge['test']['edge_neg'] = split_edge['test']['edge_neg'].t()
        print(split_edge['train']['edge'].shape)
        print(split_edge['train']['edge_neg'].shape)
        print(split_edge['valid']['edge_neg'].shape)
        print(split_edge['test']['edge_neg'].shape)
    else:
        split_edge = do_edge_split(data, args["val_ratio"], args["test_ratio"], neg_pool_max)
    x = data.x if 'x' in data.keys else None

    if False:
        size = data.x.shape[0]//10
        data.num_nodes = size
        data.x = data.x[:size,:]
        x = x[:size,:]
        data.edge_index = tail(data.edge_index, size)
        split_edge['train']['edge'] = tail(split_edge['train']['edge'], size)
        split_edge['train']['edge_neg'] = negative_sampling(
            data.edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=split_edge['train']['edge'].shape[1])
        split_edge['valid']['edge'] = tail(split_edge['valid']['edge'], size)
        split_edge['valid']['edge_neg'] = tail(split_edge['valid']['edge_neg'], size)
        split_edge['test']['edge'] = tail(split_edge['test']['edge'], size)
        split_edge['test']['edge_neg'] = tail(split_edge['test']['edge_neg'], size)
    return split_edge, x


def do_edge_split(data, val_ratio=0.05, test_ratio=0.1, neg_pool_max = False):
    edge_index, _ = add_self_loops(data.edge_index)
    data = random_split_edges(data, val_ratio, test_ratio)
    #data.train_pos_edge_index = to_undirected(data.train_pos_edge_index)
    #data.val_pos_edge_index = to_undirected(data.val_pos_edge_index)
    #data.test_pos_edge_index = to_undirected(data.test_pos_edge_index)

    if not neg_pool_max:
        data.train_neg_edge_index = negative_sampling(
            edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=data.train_pos_edge_index.shape[1],
        )
    else:
        data.train_neg_edge_index = torch.zeros((2, 0))
        #else:
    #    data.train_neg_edge_index = all_neg_edges(
    #        data.train_pos_edge_index,
    #        data.num_nodes)

    data.val_neg_edge_index = negative_sampling(
        torch.cat((edge_index, data.val_pos_edge_index), dim=-1),
        num_nodes=data.num_nodes,
        num_neg_samples=data.val_pos_edge_index.shape[1],
        force_undirected=True
    )

    data.test_neg_edge_index = negative_sampling(
        torch.cat(
            (edge_index, data.val_pos_edge_index, data.test_pos_edge_index),
            dim=-1),
        num_nodes=data.num_nodes,
        num_neg_samples=data.test_pos_edge_index.shape[1],
        force_undirected=True
    )

    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    split_edge['train']['edge'] = data.train_pos_edge_index
    split_edge['train']['edge_neg'] = data.train_neg_edge_index
    split_edge['valid']['edge'] = data.val_pos_edge_index
    split_edge['valid']['edge_neg'] = data.val_neg_edge_index
    split_edge['test']['edge'] = data.test_pos_edge_index
    split_edge['test']['edge_neg'] = data.test_neg_edge_index
    return split_edge