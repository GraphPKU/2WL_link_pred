from json import tool
import numpy as np
from torch import nn
from matmul import matmul
import torch
import torch.nn.functional as F
from torch.nn.modules.dropout import Dropout
from torch_geometric.nn import GCNConv, GraphNorm, SAGEConv
from utils import idx2mask
from copy import deepcopy


class Seq(nn.Module):
    def __init__(self, modlist):
        super().__init__()
        self.modlist = nn.ModuleList(modlist)

    def forward(self, *args, **kwargs):
        out = self.modlist[0](*args, **kwargs)
        for i in range(1, len(self.modlist)):
            out = self.modlist[i](out)
        return out

def localize(x, ei):
    #x = copy(xx)
    n = x.shape[0]
    m = x.shape[-1]
    pool = torch.arange(0, n * n)
    eim = torch.ones((n * n,), dtype=bool)
    eim[ei[0] * n + ei[1]] = 0
    pool = pool[eim].numpy().tolist()
    x = x.reshape(n**2, -1)
    tem = torch.zeros((1, m), dtype=torch.float, device=x.device)
    tem[0, 0] = 1
    x[pool, :] = tem.expand(eim.sum(), m)
    return x.reshape(n, n, -1)

class Net(nn.Module):
    def __init__(self,
                 max_x,
                 latent_size_1wl=20,
                 latent_size_2wl=20,
                 layer1=2,
                 layer2=1,
                 dp1=0.0,
                 dp2=0.0,
                 dp3=0.0,
                 mul_pool=True,
                 use_feat=False,
                 use_ea=False,
                 feat=None,
                 easize=None,
                 act = True):
        print(max_x)
        super(Net, self).__init__()
        self.mul_pool = mul_pool
        self.use_ea = use_ea
        print(use_ea)
        self.use_feat = use_feat
        self.layer1 = layer1
        self.layer2 = layer2
        input_node_size = latent_size_1wl
        if use_feat:
            input_node_size += feat.shape[1]
            self.feat = nn.parameter.Parameter(feat, requires_grad=False)

        self.embedding = nn.Sequential(nn.Embedding(max_x + 1, latent_size_1wl),
                                       nn.Dropout(p=dp1))
        relu_sage = lambda a, b, dp: Seq([
            GCNConv(a, b),
            GraphNorm(b),
            nn.Dropout(dp, inplace=True),
            nn.ReLU(inplace=True)
        ])
        relu_sage_end = lambda a, b, dp: Seq([
            GCNConv(a, b),
            GraphNorm(b),
            nn.Dropout(dp, inplace=True)
        ])
        if not act:
            self.nconvs = nn.ModuleList(
                [relu_sage_end(input_node_size, latent_size_1wl, dp2)] + [
                    relu_sage_end(latent_size_1wl, latent_size_1wl, 0)
                    for i in range(layer1 - 1)
                ])
        else:
            self.nconvs = nn.ModuleList(
                [relu_sage(input_node_size, latent_size_1wl, dp2)] + [
                    relu_sage(latent_size_1wl, latent_size_1wl, dp2)
                    for i in range(layer1 - 1)
                ])

        input_edge_size = latent_size_1wl
        if use_ea:
            input_edge_size += easize.shape[1]

        relu_lin = lambda a, b, dp: nn.Sequential(
            nn.Linear(a, b), nn.Dropout(p=dp, inplace=True),
            nn.ReLU(inplace=True))
        self.mlps_1 = nn.ModuleList(
            [relu_lin(input_edge_size + 1, latent_size_2wl, dp3)] + [
                relu_lin(latent_size_2wl, latent_size_2wl, dp3)
                for i in range(layer2 - 1)
            ])
        self.mlps_2 = nn.ModuleList(
            [relu_lin(input_edge_size + 1, latent_size_2wl, dp3)] + [
                relu_lin(latent_size_2wl, latent_size_2wl, dp3)
                for i in range(layer2 - 1)
            ])
        relu_norm_lin = lambda a, b, dp: nn.Sequential(
            nn.Linear(a, b), GraphNorm(b), nn.Dropout(p=dp, inplace=True),
            nn.ReLU(inplace=True))
        self.mlps_3 = nn.ModuleList(
            [relu_norm_lin(latent_size_2wl + input_edge_size + 1, latent_size_2wl, dp3)] +
            [
                relu_norm_lin(latent_size_2wl * 2, latent_size_2wl, dp3)
                for i in range(layer2 - 1)
            ])

        self.lin_dir = nn.Linear(latent_size_2wl, 1)

    def forward(self, x, ei, pos, ea=None):
        edge_index = ei
        #x = self.feat
        x = self.embedding(x)
        if self.use_feat:
            x = torch.cat((x, self.feat), dim=1)
        n = x.shape[0]
        for i in range(self.layer1):
            x = self.nconvs[i](x, edge_index)
        #pdb.set_trace()
        colx = x.unsqueeze(0).expand(n, -1, -1).reshape(n * n, -1)
        rowx = x.unsqueeze(1).expand(-1, n, -1).reshape(n * n, -1)
        x = rowx * colx
        if self.use_ea:
            tea = torch.zeros((n, n, ea.shape[1]), device=x.device)
            tea[ei[0] * n + ei[1]] = ea
            self.x = torch.cat((x, tea.reshape(n * n, -1)), dim=-1)

        eim = torch.zeros((n*n,), device = x.device)
        eim[ei[0] * n + ei[1]] = 1
        eim = eim.reshape(n, n)
        add_chan = torch.eye(n, device = x.device)

        x = x.reshape(n, n, -1)
        for i in range(1):
            add_chan = torch.mm(add_chan, eim)
            x = torch.cat((x, add_chan.reshape(n, n, -1)), dim=-1)
        #x = localize(x, ei)
        '''
        nl = torch.eye(n, device=x.device).unsqueeze(-1)
        x = torch.cat((x, nl), dim=-1)
        '''
        for i in range(self.layer2):
            #xx = deepcopy(x)
            x1 = self.mlps_1[i](x).permute(2, 0, 1)
            x2 = self.mlps_2[i](x).permute(2, 0, 1)
            x = torch.cat([x, (x1 @ x2).permute(1, 2, 0)], -1)
            x = self.mlps_3[i](x)
        x = (x * x.permute(1, 0, 2)).reshape(n * n, -1) if self.mul_pool else (x + x.permute(1, 0, 2)).reshape(n * n, -1)
        x = x[pos[:, 0] * n + pos[:, 1]]
        x = self.lin_dir(x)
        return x

class Net_ddi(nn.Module):
    def __init__(self,
                 max_node,
                 latent_size=20,
                 layer1=2,
                 layer2=1,
                 dp1=0.0,
                 dp2=0.0,
                 dp3=0.0,
                 use_feat=False,
                 use_ea=False,
                 feat=None,
                 easize=None):
        super(Net_ddi, self).__init__()
        self.use_ea = use_ea
        print(use_ea)
        self.use_feat = use_feat
        self.layer1 = layer1
        self.layer2 = layer2
        input_node_size = latent_size
        if use_feat:
            input_node_size += feat.shape[1]
            self.feat = nn.parameter.Parameter(feat, requires_grad=False)

        self.embedding = nn.Sequential(nn.Embedding(max_node, latent_size),
                                       nn.Dropout(p=dp1))
        relu_sage = lambda a, b, dp: Seq([
            GCNConv(a, b),
            GraphNorm(b),
            nn.Dropout(dp, inplace=True),
            nn.ReLU(inplace=True)
        ])
        self.nconvs = nn.ModuleList(
            [relu_sage(input_node_size, latent_size, dp2)] + [
                relu_sage(latent_size, latent_size, dp2)
                for i in range(layer1 - 1)
            ])

        input_edge_size = latent_size
        if use_ea:
            input_edge_size += easize.shape[1]

        relu_lin = lambda a, b, dp: nn.Sequential(
            nn.Linear(a, b), nn.Dropout(p=dp, inplace=True),
            nn.ReLU(inplace=True))
        self.mlps_1 = nn.ModuleList(
            [relu_lin(input_edge_size + 1, latent_size, dp3)] + [
                relu_lin(latent_size, latent_size, dp3)
                for i in range(layer2 - 1)
            ])
        self.mlps_2 = nn.ModuleList(
            [relu_lin(input_edge_size + 1, latent_size, dp3)] + [
                relu_lin(latent_size, latent_size, dp3)
                for i in range(layer2 - 1)
            ])
        relu_norm_lin = lambda a, b, dp: nn.Sequential(
            nn.Linear(a, b), GraphNorm(b), nn.Dropout(p=dp, inplace=True),
            nn.ReLU(inplace=True))
        self.mlps_3 = nn.ModuleList(
            [relu_norm_lin(latent_size + input_edge_size + 1, latent_size, dp3)] +
            [
                relu_norm_lin(latent_size * 2, latent_size, dp3)
                for i in range(layer2 - 1)
            ])

        self.lin_dir = nn.Linear(latent_size, 1)

    def forward(self, x, ei, pos, ea=None):
        edge_index = ei
        x = self.embedding(x)
        import pdb
        pdb.set_trace()
        if self.use_feat:
            x = torch.cat((x, self.feat), dim=1)
        n = x.shape[0]
        for i in range(self.layer1):
            x = self.nconvs[i](x, edge_index)
        colx = x.unsqueeze(0).expand(n, -1, -1).reshape(n * n, -1)
        rowx = x.unsqueeze(1).expand(-1, n, -1).reshape(n * n, -1)
        x = rowx * colx
        if self.use_ea:
            tea = torch.zeros((n, n, ea.shape[1]), device=x.device)
            tea[ei[0] * n + ei[1]] = ea
            self.x = torch.cat((x, tea.reshape(n * n, -1)), dim=-1)

        eim = torch.zeros((n*n,), device = x.device)
        eim[ei[0] * n + ei[1]] = 1
        eim = eim.reshape(n, n, 1)

        x = x.reshape(n, n, -1)
        x = torch.cat((x, eim), dim=-1)
        #x = localize(x, ei)
        '''
        nl = torch.eye(n, device=x.device).unsqueeze(-1)
        x = torch.cat((x, nl), dim=-1)
        '''
        for i in range(self.layer2):
            #xx = deepcopy(x)
            x1 = self.mlps_1[i](x).permute(2, 0, 1)
            x2 = self.mlps_2[i](x).permute(2, 0, 1)
            x = torch.cat([x, (x1 @ x2).permute(1, 2, 0)], -1)
            x = self.mlps_3[i](x)
        x = (x + x.permute(1, 0, 2)).reshape(n * n, -1)
        x = x[pos[:, 0] * n + pos[:, 1]]
        x = self.lin_dir(x)
        return x

class Net_local(nn.Module):
    def __init__(self,
                 max_x,
                 latent_size=20,
                 layer1=2,
                 layer2=1,
                 dp1=0.0,
                 dp2=0.0,
                 dp3=0.0,
                 use_feat=False,
                 use_ea=False,
                 feat=None,
                 easize=None):
        super(Net, self).__init__()
        self.use_ea = use_ea
        self.use_feat = use_feat
        self.layer1 = layer1
        self.layer2 = layer2
        input_node_size = latent_size
        if use_feat:
            input_node_size += feat.shape[1]
            self.feat = nn.parameter.Parameter(feat, requires_grad=False)

        self.embedding = nn.Sequential(nn.Embedding(max_x + 1, latent_size),
                                       nn.Dropout(p=dp1))
        relu_sage = lambda a, b, dp: Seq([
            GCNConv(a, b),
            GraphNorm(b),
            nn.Dropout(dp, inplace=True),
            nn.ReLU(inplace=True)
        ])
        self.nconvs = nn.ModuleList(
            [relu_sage(input_node_size, latent_size, dp2)] + [
                relu_sage(latent_size, latent_size, dp2)
                for i in range(layer1 - 1)
            ])

        input_edge_size = latent_size
        if use_ea:
            input_edge_size += easize.shape[1]

        relu_lin = lambda a, b, dp: nn.Sequential(
            nn.Linear(a, b), nn.Dropout(p=dp, inplace=True),
            nn.ReLU(inplace=True))
        self.mlps_1 = nn.ModuleList(
            [relu_lin(input_edge_size, latent_size, dp3)] + [
                relu_lin(latent_size, latent_size, dp3)
                for i in range(layer2 - 1)
            ])
        self.mlps_2 = nn.ModuleList(
            [relu_lin(input_edge_size, latent_size, dp3)] + [
                relu_lin(latent_size, latent_size, dp3)
                for i in range(layer2 - 1)
            ])
        relu_norm_lin = lambda a, b, dp: nn.Sequential(
            nn.Linear(a, b), GraphNorm(b), nn.Dropout(p=dp, inplace=True),
            nn.ReLU(inplace=True))
        self.mlps_3 = nn.ModuleList(
            [relu_norm_lin(latent_size + input_edge_size, latent_size, dp3)] +
            [
                relu_norm_lin(latent_size * 2, latent_size, dp3)
                for i in range(layer2 - 1)
            ])

        self.lin_dir = nn.Linear(latent_size, 1)

    def forward(self, x, ei, pos, D, ea=None):
        edge_index = ei
        x = self.embedding(x)
        if self.use_feat:
            x = torch.cat((x, self.feat), dim=1)
        n = x.shape[0]
        for i in range(self.layer1):
            x = self.nconvs[i](x, edge_index)
        colx = x.unsqueeze(0).expand(n, -1, -1).reshape(n * n, -1)
        rowx = x.unsqueeze(1).expand(-1, n, -1).reshape(n * n, -1)
        x = rowx * colx
        if self.use_ea:
            tea = torch.zeros((n, n, ea.shape[1]), device=x.device)
            tea[ei[0] * n + ei[1]] = ea
            self.x = torch.cat((x, tea.reshape(n * n, -1)), dim=-1)
        x = x.reshape(n, n, -1)
        for i in range(self.layer2):
            x1 = self.mlps_1[i](x)
            x2 = self.mlps_2[i](x)
            x = torch.cat([x, matmul(x1, x2, D)], -1)
            x = self.mlps_3[i](x)
        x = (x * x.permute(1, 0, 2)).reshape(n * n, -1)
        x = x[pos[:, 0] * n + pos[:, 1]]
        x = self.lin_dir(x)
        return x


class WLGNN(nn.Module):
    def __init__(self,
                 max_x,
                 max_z=1,
                 latent_size=32,
                 label_dim=16,
                 depth1=1,
                 depth2=1,
                 dropout=0.5,
                 pooling1="sum"):
        super().__init__()
        block_fn = lambda insize, outsize: Seq([
            GCNConv(insize, outsize),
            GraphNorm(outsize),
            Dropout(p=dropout, inplace=True),
            nn.ReLU(inplace=True)
        ])
        self.emb1 = nn.Sequential(nn.Embedding(max_x + 1, latent_size),
                                  GraphNorm(latent_size),
                                  Dropout(p=dropout, inplace=True))
        self.emb2 = nn.Sequential(nn.Embedding(max_z + 1, label_dim),
                                  GraphNorm(label_dim),
                                  Dropout(p=dropout, inplace=True))
        self.conv1s = nn.ModuleList(
            [block_fn(latent_size, latent_size) for _ in range(depth1)])
        self.conv2s = nn.ModuleList(
            [block_fn(latent_size + label_dim, latent_size)] +
            [block_fn(latent_size, latent_size) for _ in range(depth2 - 1)])
        self.pred = nn.Linear(latent_size, 1)
        pool_fns = {"sum": SumPool, "mul": MulPool, "cat": CatPool}
        self.pool1 = pool_fns[pooling1]()

    def forward(self, x, edge1, edge2, pos1, pos2):
        x = self.emb1(x).squeeze()
        for conv1 in self.conv1s:
            x = conv1(x, edge1)
        x = self.pool1(x[pos1[:, 0]], x[pos1[:, 1]])
        zemb = self.emb2(idx2mask(x.shape[0], pos2).to(torch.long)).squeeze()
        x = torch.cat((x, zemb), dim=-1)
        for conv2 in self.conv2s:
            x = conv2(x, edge2)
        x = x[pos2]
        x = self.pred(x)
        return x


class SumPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        return x1 + x2


class MulPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        return x1 * x2


class CatPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        return torch.cat([x1, x2], 1)