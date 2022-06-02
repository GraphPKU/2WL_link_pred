from dataset_SEAL import do_edge_split, load
from utils import edgegraph, degree
import torch
import numpy as np
from torch_geometric.utils import to_undirected, is_undirected
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T


class Dataset:
    def __init__(self, x, ei, pos1, y, neg_pos1):
        self.x = x
        self.ei = ei
        self.pos1 = pos1
        self.y = y
        self.neg_pos1 = neg_pos1


class BaseGraph:
    def __init__(self, x, edge_pos, edge_neg, num_pos, num_neg):
        self.x = x
        self.xs = None
        self.edge_pos = edge_pos
        self.edge_neg = edge_neg
        self.num_pos = num_pos
        self.num_neg = num_neg
        self.num_nodes = x.shape[0]
        self.max_x = None

    def preprocess(self):


        self.edge_indexs = [
            self.edge_pos[:, :self.num_pos[0]],
            self.edge_pos[:, :self.num_pos[0]],
            self.edge_pos[:, :self.num_pos[0] + self.num_pos[1]]
        ]


        pos_edges = [
            self.edge_pos[:, :self.num_pos[0]],
            self.edge_pos[:,
                          self.num_pos[0]:self.num_pos[0] + self.num_pos[1]],
            self.edge_pos[:, -self.num_pos[2]:]
        ]
        neg_edges = [
            self.edge_neg[:, :self.num_neg[0]],
            self.edge_neg[:,
                          self.num_neg[0]:self.num_neg[0] + self.num_neg[1]],
            self.edge_neg[:, -self.num_neg[2]:]
        ]

        pred_edges = [
            torch.cat((pos_edges[i], neg_edges[i]), dim=1) for i in range(3)
        ]

        #self.edge_indexs = [pos_edges[0], pos_edges[0], pos_edges[0]]

        self.pos1s = [pos_edges[0].t()
                      ] + [pred_edges[i].t() for i in range(1, 3)]

        self.ys = [
            torch.cat((torch.ones((pos_edges[i].shape[1], 1),
                                  dtype=torch.float,
                                  device=self.edge_pos.device),
                       torch.zeros((neg_edges[i].shape[1], 1),
                                   dtype=torch.float,
                                   device=self.edge_pos.device)))
            for i in range(3)
        ]
        self.edge_indexs = [to_undirected(i) for i in self.edge_indexs]

    def split(self, split: int):
        if self.xs is None:
            self.xs = [self.x for i in range(3)]
        neg_pos1 = self.edge_neg.t()[:self.num_neg[0]] if split == 0 else None
        return self.xs[split], self.edge_indexs[split], self.pos1s[
            split], self.ys[split], neg_pos1

    def setPosDegreeFeature(self):
        self.xs = [
                      degree(self.edge_indexs[0], self.num_nodes) for i in range(0, 2)
                  ] + [
                      degree(self.edge_indexs[1], self.num_nodes) for i in range(2, 3)
                  ]

        self.max_x = max([torch.max(_).item() for _ in self.xs])

    def setPosOneFeature(self):
        self.x = torch.ones((self.num_nodes,), dtype=torch.int64)
        self.max_x = 1

    def setPosNodeIdFeature(self):
        self.x = torch.arange(self.num_nodes,
                              dtype=torch.int64).to(self.pos1s[0].device)
        self.max_x = self.num_nodes - 1

    def to_undirected(self):
        if not is_undirected(self.edge_pos):
            self.edge_pos = to_undirected(self.edge_pos)

    def to(self, device):
        self.x = self.x.to(device)
        self.edge_pos = self.edge_pos.to(device)
        self.edge_neg = self.edge_neg.to(device)
        return self


def load_dataset(name, trn_ratio=0.85, val_ratio=0.05, test_ratio=0.1):
    if name in [
        "arxiv", "Celegans", "Ecoli", "NS", "PB", "Power", "Router",
        "USAir", "Yeast", "Wikipedia", "Cora", "Citeseer", "Pubmed",
        "ogbl-ddi", "ogbl-collab"
    ]:
        split_edge, node_attr = load({
            "data_name": name,
            "train_name": None,
            "test_name": None,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "max_train_num": 1000000000
        })

        train_pos = split_edge['train']['edge']
        train_neg = split_edge['train']['edge_neg']
        val_pos = split_edge["valid"]["edge"]
        val_neg = split_edge["valid"]["edge_neg"]
        test_pos = split_edge["test"]["edge"]
        test_neg = split_edge["test"]["edge_neg"]

        edge_pos = torch.cat((train_pos, val_pos, test_pos), dim=-1)
        edge_neg = torch.cat((train_neg, val_neg, test_neg), dim=-1)
        num_pos = torch.tensor(
            [train_pos.shape[1], val_pos.shape[1], test_pos.shape[1]])
        num_neg = torch.tensor(
            [train_neg.shape[1], val_neg.shape[1], test_neg.shape[1]])
        n_node = max(torch.max(edge_pos), torch.max(edge_neg)) + 1
        if node_attr == None:
            node_attr = torch.empty((n_node, 0))
        return BaseGraph(node_attr, edge_pos, edge_neg, num_pos, num_neg)
    else:
        raise NotImplementedError
