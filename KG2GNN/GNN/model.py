import torch
import torch.nn as nn
import torch.nn.functional as F
from GNN.layer import GraphConvolutionLayer
from torch.nn.modules.dropout import Dropout
from torch_geometric.nn import GCNConv, GraphNorm, SAGEConv
from scipy.sparse import coo_matrix,csr_matrix
from torch_sparse import spspmm
import numpy as np
from utils.tool import to_undirected_index

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
            idx, val = spspmm(edge_index_0, a[:, i], edge_index_1, b[:, i], n, n, n, True)
            x.append(val)
            ei = idx
        x = torch.stack(x).t()
    else:
        m0 = edge_index_0.shape[1]
        m1 = edge_index_1.shape[1]
        a = a.t().reshape(-1)
        b = b.t().reshape(-1)
        edge_index_0 = torch.stack([edge_index_0 for _ in range(m)], 1).reshape(2, -1)
        edge_index_1 = torch.stack([edge_index_1 for _ in range(m)], 1).reshape(2, -1)
        tem0 = (torch.arange(m) * n).unsqueeze(1).expand(-1, 2 * m0).reshape(-1, 2).t().to(edge_index_0.device)
        tem1 = (torch.arange(m) * n).unsqueeze(1).expand(-1, 2 * m1).reshape(-1, 2).t().to(edge_index_1.device)
        edge_index_0 += tem0
        edge_index_1 += tem1
        #sm1 = torch.sparse.FloatTensor(edge_index_0, a, torch.Size([n * m, n * m]))
        #sm2 = torch.sparse.FloatTensor(edge_index_1, b, torch.Size([n * m, n * m]))
        #sm = torch.sparse.mm(sm1, sm2).coalesce()
        #ind, val = sm.indices(), sm.values()
        ind, val = spspmm(edge_index_0, a, edge_index_1, b, n * m, n * m, n * m, True)
        return torch.sparse_coo_tensor(ind[:, :ind.shape[1]//m], val.reshape(m, -1).t(), (n, n, m)).coalesce()

    return torch.sparse_coo_tensor(ei, x, (n, n, m)).coalesce()

def sparse_cat(x, edge_index, v):
    edges = merge(x.indices(), edge_index, x.size(0))
    tem0 = torch.zeros(edges.shape[1], x.values().shape[-1], device=x.device)
    tem1 = torch.zeros(edges.shape[1], v.shape[-1], device=x.device)
    mask0 = edge_mask(edges, x.indices(), x.size(0))
    mask1 = edge_mask(edges, edge_index, x.size(0))
    tem0[mask0] = x.values()
    tem1[mask1] = v
    return torch.from_numpy(edges).to(torch.long).to(v.device), torch.cat([tem0, tem1], 1)

def merge(edge0, edge1, n):
    m0 = csr_matrix((np.ones((edge0.shape[1]), dtype=int), (edge0[0].cpu().numpy(), edge0[1].cpu().numpy())), shape = (n, n)).tolil()
    #m1 = csr_matrix((np.ones((edge1.shape[1]), dtype=int), (edge1[0].cpu().numpy(), edge1[1].cpu().numpy())), shape = (n, n)).tolil()
    m0[edge1[0].cpu().numpy().tolist(), edge1[1].cpu().numpy().tolist()] = 1
    return np.stack((m0.nonzero()[0], m0.nonzero()[1]))

def edge_mask(edge0, edge1, n):
    mask = csr_matrix((np.ones((edge0.shape[1]), dtype=int), (edge0[0], edge0[1])), shape = (n, n)).tolil()
    mask[edge1[0].cpu().numpy().tolist(), edge1[1].cpu().numpy().tolist()] = 2
    mask = mask.tocsr()
    return torch.tensor(mask.data==2, dtype=torch.bool, device=edge1.device)

def edge_list(edge0, edge1, n):
    pairs = merge(edge0, edge1, n)
    m = csr_matrix((np.arange(0, pairs.shape[1], dtype=int), (pairs[0], pairs[1])), shape=(n, n)).tolil()
    return torch.tensor(m[edge1[0].tolist(), edge1[1].tolist()].data[0],
                        dtype = torch.long, device=edge1.device)

def add_zero(x, edge0, edge1):
    mask = edge_mask(edge0, edge1, x.shape[0])
    xx = torch.zeros((mask.shape[0], x.shape[1]), dtype=torch.float, device = x.device)
    xx[mask] = x
    return xx

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
    def __init__(self, n_feat, latent_size_1=32, latent_size_2=24, depth2=2, nclass=100, ln1=True, ln2=True, ln3=True, dp1=0.5, dp2=0.5, dp3=0.5, act1=True, act2=True, act3=True, fast = False, **kwargs):
        super().__init__()
        block_fn = lambda dim1, dim2, ln, act, dp: Seq([
            GCNConv(dim1, dim2),
            nn.LayerNorm(dim2, elementwise_affine=False) if ln else nn.Identity(),
            Dropout(p=dp, inplace=True),
            nn.ReLU(inplace=True) if act else nn.Identity()
        ])
        relu_lin = lambda a, b, dp, ln, act: nn.Sequential(
            nn.Linear(a, b), 
            nn.LayerNorm(b, elementwise_affine=False) if ln else nn.Identity(),
            nn.Dropout(p=dp, inplace=True),
            nn.ReLU(inplace=True) if act else nn.Identity())

        self.conv2s = nn.ModuleList(
            [block_fn(n_feat, latent_size_1, ln1, act1, dp1)] +
            [block_fn(latent_size_1, latent_size_1, ln1, act1, dp1) for _ in range(depth2 - 1)])

        self.conv2s_r = nn.ModuleList(
            [block_fn(n_feat, latent_size_1, ln1, act1, dp1)] +
            [block_fn(latent_size_1, latent_size_1, ln1, act1, dp1) for _ in range(depth2 - 1)])

        self.mlps_1 = relu_lin(n_feat, latent_size_2, dp2, ln2, act2)
        self.mlps_2 = relu_lin(n_feat, latent_size_2, dp2, ln2, act2)
        self.mlps_3 = relu_lin(latent_size_2 + 1, latent_size_2, dp3, ln3, act3)
        self.mlps_4 = nn.Linear(latent_size_1 + latent_size_2, nclass)

        self.fast = fast

    def forward(self, x, edge2, edge2_r, ei, pred_links, num_node, pos_mask):
        val = x
        for i in range(len(self.conv2s)):
            x = self.conv2s[i](x, edge2) + self.conv2s_r[i](x, edge2_r)

        x_1 = self.mlps_1(val[pos_mask])
        x_2 = self.mlps_2(val[pos_mask])
        x_3 = sparse_bmm(ei, x_1, ei, x_2, num_node, fast=self.fast)
        current_edges, value = sparse_cat(x_3, ei, torch.ones((ei.shape[1], 1), device=x.device))
        value = self.mlps_3(value)
        value = add_zero(value, pred_links.cpu().numpy(), current_edges)
        pred_list = edge_list(current_edges, pred_links, num_node)
        value = value[pred_list]

        x = torch.cat([x, value], dim = -1)
        x = x[:x.shape[0]//2] + x[x.shape[0]//2:]
        x = torch.sigmoid(self.mlps_4(x))
        return x

class WLGNN_hy(nn.Module):
    def __init__(self, n_feat, latent_size_1=32, latent_size_2=24, depth2=2, nclass=100, dp1=0.5, dp2=0.5, fast = False):
        super().__init__()
        block_fn = lambda insize, outsize: Seq([
            GCNConv(insize, outsize),
            GraphNorm(outsize),
            Dropout(p=dp1, inplace=True),
            nn.ReLU(inplace=True)
        ])
        block_fn_end = lambda insize, outsize: Seq([
            GCNConv(insize, outsize),
            GraphNorm(outsize),
            Dropout(p=dp1, inplace=True),
            #nn.ReLU(inplace=True)
        ])
        relu_lin = lambda a, b, dp: nn.Sequential(
            nn.Linear(a, b), nn.Dropout(p=dp, inplace=True),
            nn.ReLU(inplace=True))

        self.conv2s = nn.ModuleList(
            [block_fn(n_feat, latent_size_1)] +
            [block_fn_end(latent_size_1, latent_size_1) for _ in range(depth2 - 1)])

        self.conv2s_r = nn.ModuleList(
            [block_fn(n_feat, latent_size_1)] +
            [block_fn_end(latent_size_1, latent_size_1) for _ in range(depth2 - 1)])

        self.mlps_1 = relu_lin(n_feat, latent_size_2, dp2)
        self.mlps_2 = relu_lin(n_feat, latent_size_2, dp2)
        self.mlps_3 = relu_lin(latent_size_2 + 1, latent_size_2, dp2)
        self.mlps_4 = nn.Linear(latent_size_1 + latent_size_2, nclass)

        self.fast = fast

    def forward(self, x, edge2, edge2_r, ei, pred_links, num_node, pos_mask):
        n = x.shape[0]
        val = x.clone()
        for i in range(len(self.conv2s)):
            x = self.conv2s[i](x, edge2) + self.conv2s_r[i](x, edge2_r)

        x_1 = self.mlps_1(val[pos_mask])
        x_2 = self.mlps_2(val[pos_mask])
        x_3 = sparse_bmm(ei, x_1, ei, x_2, num_node, fast=self.fast)
        current_edges, value = sparse_cat(x_3, ei, torch.ones((ei.shape[1], 1), device=x.device))
        value = self.mlps_3(value)
        value = add_zero(value, pred_links.cpu().numpy(), current_edges)
        pred_list = edge_list(current_edges, pred_links, num_node)
        value = value[pred_list]

        if (x.shape[0]!=value.shape[0]):
            import pdb
            pdb.set_trace()

        x = torch.cat([x, value], dim = -1)
        x = x[:x.shape[0]//2] + x[x.shape[0]//2:]
        x = torch.sigmoid(self.mlps_4(x))
        return x


class GCN(nn.Module):
    
    """
    a GCN model with multiple layers
    """
    
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
            
        #in our case, nfeat = nclass
        
        self.gc1 = GraphConvolutionLayer(nfeat, nhid)
        self.gc2 = GraphConvolutionLayer(nhid, nclass)
        #self.gc2 = GraphConvolutionLayer(nhid, nhid2)
        #self.gc3 = GraphConvolutionLayer(nhid2, nclass)
        
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        
        """
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(x, adj)
        """
        #output with applying a sigmoid function
        x = torch.sigmoid(x)
        output = torch.reshape(x,(adj.shape[0], -1))
        
        return output

class WLGNN_d(nn.Module):
    def __init__(self, n_feat, dim=32, depth2=2, nclass=100, ln1=True, act1=True, dp1=0.3, directed = True):
        super().__init__()
        block_fn = lambda dim1, dim2, ln, act, dp: Seq([
            GCNConv(dim1, dim2),
            nn.LayerNorm(dim2, elementwise_affine=False) if ln else nn.Identity(),
            Dropout(p=dp, inplace=True),
            nn.ReLU(inplace=True) if act else nn.Identity()
        ])
        
        self.conv2s = nn.ModuleList(
            [block_fn(n_feat, dim, ln1, act1, dp1)] +
            [block_fn(dim, dim, ln1, act1, dp1) for _ in range(depth2 - 1)])
        
        if directed:
            self.conv2s_r = nn.ModuleList(
                [block_fn(n_feat, dim, ln1, act1, dp1)] +
                [block_fn(dim, dim, ln1, act1, dp1) for _ in range(depth2 - 1)])

        self.mlp = nn.Sequential(nn.Linear(dim, nclass), nn.Sigmoid())
        self.directed = directed

    def forward(self, x, edge2, edge2_r, ei2):
        for i in range(len(self.conv2s)):
            x = self.conv2s[i](x, edge2) + self.conv2s_r[i](x, edge2_r) if self.directed \
                else self.conv2s[i](x, ei2)
        if self.directed:
            x = x[:x.shape[0]//2] + x[x.shape[0]//2:]
        x = self.mlp(x)
        return x


class WLGNN_d_hy(nn.Module):
    def __init__(self, n_feat, latent_size=32, depth2=2, nclass=100, dropout=0.5, directed = True):
        super().__init__()
        block_fn = lambda insize, outsize: Seq([
            GCNConv(insize, outsize),
            GraphNorm(outsize),
            nn.LayerNorm(outsize, elementwise_affine=False),
            Dropout(p=dropout, inplace=True),
            nn.ReLU(inplace=True)
        ])
        block_fn_end = lambda insize, outsize: Seq([
            GCNConv(insize, outsize),
            GraphNorm(outsize),
            #Dropout(p=dropout, inplace=True),
            #nn.ReLU(inplace=True)
        ])

        self.conv2s = nn.ModuleList(
            [block_fn(n_feat, latent_size)] +
            [block_fn_end(latent_size, nclass) for _ in range(depth2 - 1)])

        if directed:
            self.conv2s_r = nn.ModuleList(
                [block_fn(n_feat, latent_size)] +
                [block_fn_end(latent_size, nclass) for _ in range(depth2 - 1)])

        self.directed = directed

    def forward(self, x, edge2, edge2_r, ei2, m=0):
        n = x.shape[0]//2 - m
        for i in range(len(self.conv2s)):
            x = self.conv2s[i](x, edge2) + self.conv2s_r[i](x, edge2_r) if self.directed \
                else self.conv2s[i](x, ei2)
        if self.directed:
            x = torch.cat([x[:m] + x[m + n:2 * m + n], x[m:m + n]], 0)
        x = torch.sigmoid(x)
        return x