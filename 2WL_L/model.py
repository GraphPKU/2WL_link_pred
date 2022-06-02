from torch import nn
import torch
import torch.nn.functional as F
from torch.nn.modules.dropout import Dropout
from torch_geometric.nn import GCNConv, GraphNorm, SAGEConv


class Seq(nn.Module):
    def __init__(self, modlist):
        super().__init__()
        self.modlist = nn.ModuleList(modlist)

    def forward(self, *args, **kwargs):
        out = self.modlist[0](*args, **kwargs)
        for i in range(1, len(self.modlist)):
            out = self.modlist[i](out)
        return out


class WLGNN(nn.Module):
    def __init__(self, max_x, use_node_attr, na_size, latent_size=32, depth1=1, depth2=1, dropout_0 = 0.5, dropout_1=0.5, dropout_2=0.5, dropout_3=0.5, act0 = True, act1 = True):
        super().__init__()
        block_fn_1 = lambda insize, outsize: Seq([
            GCNConv(insize, outsize),
            GraphNorm(outsize),
            Dropout(p=dropout_1, inplace=True),
            nn.ReLU(inplace=True) if act0 else nn.Identity()
        ])
        block_fn_1_end = lambda insize, outsize: Seq([
            GCNConv(insize, outsize),
            GraphNorm(outsize),
            Dropout(p=dropout_3, inplace=True),
            nn.ReLU(inplace=True) if act1 else nn.Identity()
        ])
        block_fn_2 = lambda insize, outsize: Seq([
            GCNConv(insize, outsize),
            GraphNorm(outsize),
            Dropout(p=dropout_2, inplace=True),
            nn.ReLU(inplace=True)
        ])
        self.max_x = max_x
        self.na = use_node_attr
        if not use_node_attr:
            self.emb = nn.Sequential(nn.Embedding(max_x + 1, latent_size),
                                     GraphNorm(latent_size),
                                     Dropout(p=dropout_0, inplace=True))
        input_size = latent_size if not use_node_attr else na_size

        self.conv1s = nn.ModuleList(
            [block_fn_1(input_size, latent_size)] +
            [block_fn_1_end(latent_size, latent_size) for _ in range(depth1 - 1)])


        self.conv2s = nn.ModuleList(
            [block_fn_2(latent_size, latent_size)] +
            [block_fn_2(latent_size, latent_size) for _ in range(depth2 - 1)])
        self.conv2s_r = nn.ModuleList(
            [block_fn_2(latent_size, latent_size)] +
            [block_fn_2(latent_size, latent_size) for _ in range(depth2 - 1)])
        self.pred = nn.Linear(latent_size, 1)

    def forward(self, x, na, edge1, edge2, edge2_r, pos1, pos2, test = False):

        if not self.na:
            x = self.emb(x).squeeze()
        else:
            x = na

        for conv1 in self.conv1s:
            x = conv1(x, edge1)

        x = x[pos1[:, 0]] * x[pos1[:, 1]]
        for i in range(len(self.conv2s)):
            x = self.conv2s[i](x, edge2) + self.conv2s_r[i](x, edge2_r)
        x = x[pos2]
        mask = torch.cat(
            [torch.ones([1, x.shape[0] // 2], dtype=bool), torch.zeros([1, x.shape[0] // 2], dtype=bool)]).t().reshape(
            -1)
        x = x[mask] * x[~mask]
        x = self.pred(x)
        return x

class Model_HY(nn.Module):
    def __init__(self, max_x, use_node_attr, na_size, latent_size=32, depth1=2, depth2=2, dropout=0):
        super().__init__()
        block_fn = lambda insize, outsize: Seq([
            SAGEConv(insize, outsize),
            nn.ReLU(inplace=True)
        ])
        self.na = use_node_attr
        if not use_node_attr:
            self.emb = nn.Sequential(nn.Embedding(max_x + 1, latent_size),
                                     GraphNorm(latent_size),
                                     Dropout(p=dropout, inplace=True))
        input_size = latent_size if not use_node_attr else na_size
        self.conv1s = nn.ModuleList(
            [block_fn(input_size, latent_size)] +
            [block_fn(latent_size, latent_size) for _ in range(depth1 - 1)])
        self.conv2s = nn.ModuleList(
            [block_fn(latent_size, latent_size)] +
            [block_fn(latent_size, latent_size) for _ in range(depth2 - 1)])
        self.conv2s_r = nn.ModuleList(
            [block_fn(latent_size, latent_size)] +
            [block_fn(latent_size, latent_size) for _ in range(depth2 - 1)])
        self.pred = nn.Linear(latent_size, 1)

    def forward(self, x, na, edge1, edge2, edge2_r, pos1, pos2):
        if not self.na:
            x = self.emb(x).squeeze()
        else:
            x = na
        for conv1 in self.conv1s:
            x = conv1(x, edge1)
        x = x[pos1[:,0]] * x[pos1[:,1]]
        #x = x[pos1].reshape(pos1.shape[0], -1)

        for i in range(len(self.conv2s)):
            x = self.conv2s[i](x, edge2) + self.conv2s_r[i](x, edge2_r)

        x = x[pos2]
        mask = torch.cat([torch.ones([1, x.shape[0]//2], dtype=bool), torch.zeros([1, x.shape[0]//2], dtype=bool)]).t().reshape(-1)
        x = x[mask] * x[~mask]
        #x = torch.cat([x[mask0], x[mask1]],1)
        x = self.pred(x)
        return x