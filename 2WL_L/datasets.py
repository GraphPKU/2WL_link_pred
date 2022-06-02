from dataset_SEAL import load
from utils import get_ei2, idx2mask, blockei2
import torch
import numpy as np
from torch_geometric.utils import to_undirected, is_undirected
from utils import double, degree


class dataset:
    def __init__(self, x, na, ei, ea, pos1, y, ei2):
        self.x = x
        self.na = na
        self.ei = ei
        self.ea = ea
        self.pos1 = pos1
        self.y = y
        self.ei2 = ei2


class BaseGraph:
    def __init__(self, x, node_attr, edge_pos, edge_neg, num_pos, num_neg):
        self.x = x
        self.node_attr = node_attr
        self.edge_pos = edge_pos
        self.edge_neg = edge_neg
        #self.edge_attr = edge_attr
        self.num_pos = num_pos
        self.num_neg = num_neg
        self.num_nodes = x.shape[0]

    def preprocess(self):
        self.edge_indexs = [
            self.edge_pos[:, :self.num_pos[0]],
            self.edge_pos[:, :self.num_pos[0]],
            self.edge_pos[:, :self.num_pos[0] + self.num_pos[1]]
        ]

        self.edge_attrs = [
            torch.ones_like(self.edge_indexs[i][0], dtype=torch.float)
            for i in range(3)
        ]
        '''if self.attr == None else [
            self.edge_attr[0],
            self.edge_attr[1],
            self.edge_attr[2]
        ]
        '''
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

        pred_edges = [neg_edges[0]] + [
            torch.cat((pos_edges[i], neg_edges[i]), dim=1)
            for i in range(1, 3)
        ]

        self.pos1s = [
            torch.cat((self.edge_indexs[i].t(), pred_edges[i].t()), dim=0)
            for i in range(3)
        ]

        self.ys = [torch.zeros((neg_edges[0].shape[1], 1),
                                 device=self.edge_pos.device)]+[
            torch.cat((torch.ones((pos_edges[i].shape[1], 1),
                                  dtype=torch.float,
                                  device=self.edge_pos.device),
                       torch.zeros((neg_edges[i].shape[1], 1),
                                   dtype=torch.float,
                                   device=self.edge_pos.device)))
            for i in range(1, 3)
        ]
        if True:
            self.ei2s = [
                get_ei2(self.num_nodes, self.edge_indexs[0], pred_edges[0]),
                get_ei2(self.num_nodes, self.edge_indexs[1], pred_edges[1]),
                get_ei2(self.num_nodes, self.edge_indexs[2], pred_edges[2])
            ]
            #np.save('2wl_train_edge_{}.npy'.format(self.num_neg[0]), ei2_0.cpu().numpy())
            #np.save('2wl_test_edge_{}.npy'.format(self.num_neg[0]), ei2_2.cpu().numpy())
        else:
            ei2_0 = torch.from_numpy(np.load('2wl_train_edge_{}.npy'.format(self.num_neg[0]))).to(self.x.device)
            ei2_0 = torch.from_numpy(np.load('2wl_test_edge_{}.npy'.format(self.num_neg[0]))).to(self.x.device)


    def split(self, split: int):
        return self.x[split], self.node_attr, self.edge_indexs[split], self.edge_attrs[
            split], self.pos1s[split], self.ys[split], self.ei2s[split]

    def setPosDegreeFeature(self):
        #print(self.edge_indexs[0], self.edge_attrs[0], (self.x.shape[0], self.x.shape[0]))

        self.x = [
                      degree(self.edge_indexs[0], self.num_nodes) for i in range(0, 2)
                  ] + [
                      degree(self.edge_indexs[2], self.num_nodes) for i in range(2, 3)
                  ]

        self.max_x = max([torch.max(_).item() for _ in self.x])

    def setPosOneFeature(self):
        self.x = torch.ones((self.x.shape[0], 1), dtype=torch.int64)

    def setPosNodeIdFeature(self):
        self.x = torch.arange(self.x.shape[0],
                              dtype=torch.int64).reshape(self.x.shape[0], 1)

    def to_undirected(self):
        if not is_undirected(self.edge_pos):
            self.edge_pos = to_undirected(self.edge_pos)

    def to(self, device):
        self.x = self.x.to(device)
        self.edge_pos = self.edge_pos.to(device)
        self.edge_neg = self.edge_neg.to(device)
        return self


def load_dataset(name, trn_ratio=0.8, val_ratio=0.05, test_ratio=0.1):
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


        train_pos = double(split_edge['train']['edge'])
        train_neg = double(split_edge['train']['edge_neg'])
        val_pos = double(split_edge["valid"]["edge"])
        val_neg = double(split_edge["valid"]["edge_neg"])
        test_pos = double(split_edge["test"]["edge"])
        test_neg = double(split_edge["test"]["edge_neg"])


        edge_pos = torch.cat((train_pos, val_pos, test_pos), dim=-1)
        edge_neg = torch.cat((train_neg, val_neg, test_neg), dim=-1)
        #edge_attr = torch.cat((train_attr, val_attr, test_attr), dim=-1)
        num_pos = torch.tensor(
            [train_pos.shape[1], val_pos.shape[1], test_pos.shape[1]])
        num_neg = torch.tensor(
            [train_neg.shape[1], val_neg.shape[1], test_neg.shape[1]])
        n_node = max(torch.max(edge_pos), torch.max(edge_neg)) + 1
        x = torch.zeros((n_node, 0))
        print(num_pos)
        print(num_neg)
        return BaseGraph(x, node_attr, edge_pos, edge_neg, num_pos, num_neg)
    else:
        raise NotImplementedError
