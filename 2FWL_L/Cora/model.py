from json import tool
import numpy as np
from sqlalchemy import Identity
from torch import Tensor, conv_tbc, nn
from matmul import matmul
import torch
import torch.nn.functional as F
from torch.nn.modules.dropout import Dropout
from torch_geometric.nn import GCNConv, GraphNorm, SAGEConv, APPNP
from utils import idx2mask
from scipy.sparse import coo_matrix, csr_matrix
from torch_sparse import spspmm
from copy import deepcopy
import time


class Seq(nn.Module):

    def __init__(self, modlist):
        super().__init__()
        self.modlist = nn.ModuleList(modlist)

    def forward(self, *args, **kwargs):
        out = self.modlist[0](*args, **kwargs)
        for i in range(1, len(self.modlist)):
            out = self.modlist[i](out)
        return out


def sparse_bmm(edge_index_0, a, edge_index_1, b, n, fast=False):
    m = a.shape[-1]
    if not fast:
        x = []
        for i in range(m):
            #sm1 = torch.sparse.FloatTensor(edge_index_0, a[:, i], torch.Size([n, n]))
            #sm2 = torch.sparse.FloatTensor(edge_index_1, b[:, i], torch.Size([n, n]))
            #sm = torch.sparse.mm(sm1, sm2).coalesce()
            #x.append(sm.values())
            #ei = sm.indices()
            idx, val = spspmm(edge_index_0, a[:, i], edge_index_1, b[:, i], n,
                              n, n, True)
            x.append(val)
            ei = idx
        x = torch.stack(x).t()
    else:
        m0 = edge_index_0.shape[1]
        m1 = edge_index_1.shape[1]
        a = a.t().reshape(-1)
        b = b.t().reshape(-1)
        edge_index_0 = torch.stack([edge_index_0 for _ in range(m)],
                                   1).reshape(2, -1)
        edge_index_1 = torch.stack([edge_index_1 for _ in range(m)],
                                   1).reshape(2, -1)
        tem0 = (torch.arange(m) * n).unsqueeze(1).expand(-1, 2 * m0).reshape(
            -1, 2).t().to(edge_index_0.device)
        tem1 = (torch.arange(m) * n).unsqueeze(1).expand(-1, 2 * m1).reshape(
            -1, 2).t().to(edge_index_1.device)
        edge_index_0 += tem0
        edge_index_1 += tem1
        #sm1 = torch.sparse.FloatTensor(edge_index_0, a, torch.Size([n * m, n * m]))
        #sm2 = torch.sparse.FloatTensor(edge_index_1, b, torch.Size([n * m, n * m]))
        #sm = torch.sparse.mm(sm1, sm2).coalesce()
        #ind, val = sm.indices(), sm.values()
        ind, val = spspmm(edge_index_0, a, edge_index_1, b, n * m, n * m,
                          n * m, True)
        return torch.sparse_coo_tensor(ind[:, :ind.shape[1] // m],
                                       val.reshape(m, -1).t(),
                                       (n, n, m)).coalesce()

    return torch.sparse_coo_tensor(ei, x, (n, n, m)).coalesce()


def sparse_cat(x, edge_index, v):
    edges = merge(x.indices(), edge_index, x.size(0))
    tem0 = torch.zeros(edges.shape[1], x.values().shape[-1], device=x.device)
    tem1 = torch.zeros(edges.shape[1], v.shape[-1], device=x.device)
    mask0 = edge_mask(edges, x.indices(), x.size(0))
    mask1 = edge_mask(edges, edge_index, x.size(0))
    tem0[mask0] = x.values()
    tem1[mask1] = v
    return torch.from_numpy(edges).to(torch.long).to(v.device), torch.cat(
        [tem0, tem1], 1)


def merge(edge0, edge1, n):
    m0 = csr_matrix((np.ones((edge0.shape[1]), dtype=int),
                     (edge0[0].cpu().numpy(), edge0[1].cpu().numpy())),
                    shape=(n, n)).tolil()
    #m1 = csr_matrix((np.ones((edge1.shape[1]), dtype=int), (edge1[0].cpu().numpy(), edge1[1].cpu().numpy())), shape = (n, n)).tolil()
    m0[edge1[0].cpu().numpy().tolist(), edge1[1].cpu().numpy().tolist()] = 1
    return np.stack((m0.nonzero()[0], m0.nonzero()[1]))


def edge_mask(edge0, edge1, n):
    mask = csr_matrix((np.ones(
        (edge0.shape[1]), dtype=int), (edge0[0], edge0[1])),
                      shape=(n, n)).tolil()
    mask[edge1[0].cpu().numpy().tolist(), edge1[1].cpu().numpy().tolist()] = 2
    mask = mask.tocsr()
    return torch.tensor(mask.data == 2, dtype=torch.bool, device=edge1.device)


def edge_list(edge0, edge1, n):
    pairs = merge(edge0, edge1, n)
    m = csr_matrix(
        (np.arange(1, pairs.shape[1] + 1, dtype=int), (pairs[0], pairs[1])),
        shape=(n, n)).tolil()
    return torch.tensor(m[edge1[0].tolist(), edge1[1].tolist()].data[0],
                        dtype=torch.long,
                        device=edge1.device) - 1


def add_zero(x, edge0, edge1):
    mask = edge_mask(edge0, edge1, x.shape[0])
    xx = torch.zeros((mask.shape[0], x.shape[1]),
                     dtype=torch.float,
                     device=x.device)
    xx[mask] = x
    return xx

class WLGNN(nn.Module):
    def __init__(self, kind:str, feat:Tensor, dim:int=64, dropout:float=0.3) -> None:
        super().__init__()
        if kind=="sage":
            conv_fn = SAGEConv
        elif kind=="gcn":
            conv_fn = GCNConv
        else:
            raise NotImplementedError
        self.mp1 = Seq([conv_fn(feat.shape[1], dim), nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True)])
        self.mp2 = Seq([conv_fn(dim, dim), nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True)])
        self.register_buffer("feat", feat)

    def forward(self, x, ei, pos, ea=None, tst=False):
        x = self.mp1(self.feat, ei)
        x = self.mp2(x, ei)
        xx = torch.sum(x[pos[:, 0]] * x[pos[:, 1]], dim=-1, keepdim=True)
        return xx

class Net_wxy(nn.Module):
    def __init__(self,
                 max_x,
                 dim1=20,
                 dim2=20,
                 layer1=2,
                 layer2=1,
                 dp0=0.0,
                 dp1=0.0,
                 dp3=0.0,
                 dp4=0.0,
                 fast_bsmm=False,
                 use_feat=False,
                 use_ea=False,
                 alpha=0.1,
                 ln1=False,
                 ln3=False,
                 ln4=False,
                 act1=False,
                 act3=False,
                 act4=False,
                 feat=None,
                 easize=None):
        super().__init__()
        self.use_ea = use_ea
        self.max_x = max_x
        self.use_feat = use_feat
        self.layer1 = layer1
        self.layer2 = layer2
        self.fast = fast_bsmm
        input_node_size = dim1
        if use_feat:
            input_node_size += feat.shape[1]
            self.feat = nn.parameter.Parameter(feat, requires_grad=False)
        use_affine = False
        relu_lin = lambda a, b, dp, lnx, actx: nn.Sequential(
            nn.Linear(a, b),
            nn.LayerNorm(b, elementwise_affine=use_affine) if lnx else nn.Identity(), 
            nn.Dropout(p=dp, inplace=True),
            nn.ReLU(inplace=True) if actx else nn.Identity())
        self.lin1 = nn.Sequential(
            nn.Dropout(dp0),
            relu_lin(self.feat.shape[1], dim1, dp1, ln1, act1)
        )
        self.nconvs = APPNP(layer1, alpha)
        self.mlps_1 = relu_lin(dim1 * 2, dim2, dp3, ln3, act3)
        self.mlps_2 = relu_lin(dim1 * 2, dim2, dp3, ln3, act3)
        self.mlps_3 = nn.ModuleList([
            relu_lin(dim2 + 1, dim2, dp4, ln4, act4) for _ in range(layer2)
        ])
        
    def forward(self, x, ei, pos, ea=None, tst=False):
        edge_index = ei

        x = self.feat
        n = x.shape[0]
        x = self.lin1(x)
        x = self.nconvs(x, edge_index)
        xx = x[pos[:, 0]] * x[pos[:, 1]]

        current_edges = edge_index

        val = torch.cat([x[edge_index[0]], x[edge_index[1]]], 1)  
        x = self.mlps_1(val)
        mul = self.mlps_2(val)

        for i in range(self.layer2):
            x = sparse_bmm(current_edges,
                           x,
                           edge_index,
                           mul,
                           n,
                           fast=self.fast)
            current_edges, value = sparse_cat(
                x, edge_index,
                torch.ones((edge_index.shape[1], 1), device=x.device))
            x = self.mlps_3[i](value)
        sm = torch.sparse.FloatTensor(
            torch.cat(
                [current_edges[1].unsqueeze(0), current_edges[0].unsqueeze(0)],
                0), x, torch.Size([n, n, x.shape[-1]])).coalesce().values()
        x = x * sm
        x = add_zero(x, pos.t().cpu().numpy(), current_edges)

        pred_list = edge_list(current_edges, pos.t(), n)
        x = x[pred_list]
        x = torch.cat([x, xx], 1)
        x = torch.sum(x, dim=-1, keepdim=True) #self.lin_dir(x)
        return x


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
                 fast_bsmm=False,
                 use_feat=False,
                 use_ea=False,
                 ln0=False,
                 ln1=False,
                 ln2=False,
                 ln3=False,
                 act1=False,
                 act2=False,
                 act3=False,
                 feat=None,
                 easize=None):
        super(Net, self).__init__()
        self.use_ea = use_ea
        self.max_x = max_x
        self.use_feat = use_feat
        self.layer1 = layer1
        self.layer2 = layer2
        self.fast = fast_bsmm
        input_node_size = latent_size_1wl
        if use_feat:
            input_node_size += feat.shape[1]
            self.feat = nn.parameter.Parameter(feat, requires_grad=False)
        use_affine = False
        self.embedding = nn.Sequential(
            nn.Embedding(max_x + 1, latent_size_1wl),
            nn.LayerNorm(latent_size_1wl, elementwise_affine=use_affine)
            if ln0 else nn.Identity(), nn.Dropout(p=dp1))
        relu_sage = lambda a, b, dp: Seq([
            GCNConv(a, b),
            nn.LayerNorm(b, elementwise_affine=use_affine)
            if ln1 else nn.Identity(),
            nn.Dropout(dp, inplace=True),
            nn.ReLU(inplace=True) if act1 else nn.Identity(),
        ])
        self.nconvs = nn.ModuleList(
            [relu_sage(feat.shape[1], latent_size_1wl, dp2)] + [
                relu_sage(latent_size_1wl, latent_size_1wl, dp2)
                for i in range(layer1 - 1)
            ])
        input_edge_size = latent_size_1wl
        if use_ea:
            input_edge_size += easize.shape[1]
        relu_lin = lambda a, b, dp: nn.Sequential(
            nn.Linear(a, b),
            nn.LayerNorm(b, elementwise_affine=use_affine)
            if ln2 else nn.Identity(), nn.Dropout(p=dp, inplace=True),
            nn.ReLU(inplace=True) if act2 else nn.Identity())
        self.mlps_1 = relu_lin(input_edge_size * 2, latent_size_2wl, dp3)
        self.mlps_2 = nn.ModuleList(
            [relu_lin(input_edge_size * 2, latent_size_2wl, dp3)] + [
                relu_lin(input_edge_size * 2, latent_size_2wl, dp3)
                for i in range(layer2 - 1)
            ])
        relu_norm_lin = lambda a, b, dp: nn.Sequential(
            nn.Linear(a, b),
            nn.LayerNorm(b, elementwise_affine=use_affine)
            if ln3 else nn.Identity(), nn.Dropout(p=dp, inplace=True),
            nn.ReLU(inplace=True) if act3 else nn.Identity())
        self.mlps_3 = nn.ModuleList([
            relu_norm_lin(latent_size_2wl + 1, latent_size_2wl, dp3)
            for _ in range(layer2)
        ])
        self.lin_dir = nn.Linear(latent_size_1wl + latent_size_2wl, 1)

    def forward(self, x, ei, pos, ea=None, tst=False):
        edge_index = ei

        x = self.feat
        n = x.shape[0]
        for i in range(self.layer1):
            x = self.nconvs[i](x, edge_index)
        xx = x[pos[:, 0]] * x[pos[:, 1]]

        val = torch.cat(
            [x[edge_index[0]], x[edge_index[1]]],
            1)  #colx = x.unsqueeze(0).expand(n, -1, -1).reshape(n * n, -1)

        if self.use_ea:
            val = torch.cat([val, ea], 0)

        x = val.clone()
        x = self.mlps_1(x)
        current_edges = edge_index

        for i in range(self.layer2):
            #xx = deepcopy(x)
            mul = self.mlps_2[i](val)
            x = sparse_bmm(current_edges,
                           x,
                           edge_index,
                           mul,
                           n,
                           fast=self.fast)
            current_edges, value = sparse_cat(
                x, edge_index,
                torch.ones((edge_index.shape[1], 1), device=x.device))
            x = self.mlps_3[i](value)
        sm = torch.sparse.FloatTensor(
            torch.cat(
                [current_edges[1].unsqueeze(0), current_edges[0].unsqueeze(0)],
                0), x, torch.Size([n, n, x.shape[-1]])).coalesce().values()
        x = x * sm
        x = add_zero(x, pos.t().cpu().numpy(), current_edges)

        pred_list = edge_list(current_edges, pos.t(), n)

        x = x[pred_list]
        x = torch.cat([x, xx], 1)

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
        self.mlps_3 = nn.ModuleList([
            relu_norm_lin(latent_size + input_edge_size + 1, latent_size, dp3)
        ] + [
            relu_norm_lin(latent_size * 2, latent_size, dp3)
            for i in range(layer2 - 1)
        ])

        self.lin_dir = nn.Linear(latent_size, 1)

    def forward(self, x, ei, pos, ea=None):
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

        eim = torch.zeros((n * n, ), device=x.device)
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
        x = (x * x.permute(1, 0, 2)).reshape(n * n, -1)
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