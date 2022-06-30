import torch
from torch import Tensor
from torch_scatter import scatter_add
from scipy.sparse import csr_matrix
from torch_sparse import spspmm
import numpy as np

@torch.jit.script
def degree(ei: Tensor, num_node: int):
    return scatter_add(torch.ones_like(ei[1]), ei[1], dim_size=num_node)

@torch.jit.script
def set_mul(a: Tensor, b: Tensor):
    a = a.reshape(-1, 1)
    b = b.reshape(1, -1)
    a = a.expand(-1, b.shape[1])
    b = b.expand(a.shape[0], -1)
    return torch.cat((a.reshape(-1, 1), b.reshape(-1, 1)), dim=-1)


@torch.jit.script
def check_in_set(target, set):
    # target (n,), set(m,)
    a = target.reshape(-1, 1)
    b = set.reshape(1, -1)
    out = []
    cutshape = 1024 * 1024 * 1024 // b.shape[1]
    out = torch.cat([
        torch.sum((a[i:i + cutshape] == b), dim=-1)
        for i in range(0, a.shape[0], cutshape)
    ])
    return out


@torch.jit.script
def get_ei2(n_node: int, pos_edge, pred_edge):

    edge = torch.cat((pos_edge, pred_edge), dim=-1)  #pos.transpose(0, 1)
    idx = torch.arange(edge.shape[1], device=edge.device)
    idx_pos = torch.arange(pos_edge.shape[1], device=edge.device)
    edge2 = [
        set_mul(idx_pos[pos_edge[1] == i], idx[edge[0] == i])
        for i in range(n_node)
    ]
    return torch.cat(edge2, dim=0).t()

def sparse_bmm(edge_index_0, a, edge_index_1, b, n, fast=False):
    m = a.shape[-1]
    if not fast:
        x = []
        for i in range(m):
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
    mask = csr_matrix((np.ones((edge0.shape[1]), dtype=int), (edge0[0], edge0[1])),shape=(n, n)).tolil()
    mask[edge1[0].cpu().numpy().tolist(), edge1[1].cpu().numpy().tolist()] = 2
    mask = mask.tocsr()
    import pdb
    if (mask.data==2).sum() != edge1.shape[1]:
        pdb.set_trace()
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

@torch.jit.script
def blockei2(ei2, blocked_idx):
    return ei2[:, torch.logical_not(check_in_set(ei2[0], blocked_idx))]


@torch.jit.script
def idx2mask(num: int, idx):
    mask = torch.zeros((num), device=idx.device, dtype=torch.bool)
    mask[idx] = True
    return mask


@torch.jit.script
def mask2idx(mask):
    idx = torch.arange(mask.shape[0], device=mask.device)
    return idx[mask]


#@torch.jit.script
def sample_block(sample_idx, size, ei, ei2=None):
    ea = torch.ones((ei.shape[-1],), dtype=torch.float, device=ei.device)
    ea_new = ea[torch.logical_not(idx2mask(ei.shape[1], sample_idx))]
    ei_new = ei[:, torch.logical_not(idx2mask(ei.shape[1], sample_idx))]
    ei2_new = blockei2(ei2, sample_idx) if ei2 is not None else None
    adj = torch.sparse_coo_tensor(ei_new, ea_new, (size, size))
    x_new = torch.sparse.sum(adj, dim=1).to_dense().to(torch.int64).reshape(-1)
    return ei_new, x_new, ei2_new

def reverse(edge_index):
    tem0 = 1 - (edge_index[0] > edge_index[0] // 2 * 2).to(torch.long) * 2
    tem1 = 1 - (edge_index[1] > edge_index[1] // 2 * 2).to(torch.long) * 2
    edge = torch.cat([(edge_index[0] + tem0).unsqueeze(0), edge_index[1].unsqueeze(0)])
    edge_r = torch.cat([edge_index[0].unsqueeze(0), (edge_index[1] + tem1).unsqueeze(0)])
    #return edge_index
    return edge, edge_r

import math

from torch_geometric.utils import to_undirected
from torch_geometric.deprecation import deprecated

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

def all_neg_edges(edge_index, n):
    row = torch.arange(n).expand(n,n).reshape(n*n, 1).cuda()
    col = torch.arange(n).t().expand(n,n).reshape(n*n, 1).cuda()
    edge = torch.cat([row, col], -1)
    edge = edge[edge[:, 0] >= edge[:, 1], :]
    for i in range(edge_index.shape[1]):
        p, q = edge_index[:, i]
        edge = edge[(edge[:, 0] != p) | (edge[:, 1] != q), :]
    edge = edge.cpu()
    return edge.t()

def random_split_edges(data, val_ratio: float = 0.05,
                           test_ratio: float = 0.1):
    r"""Splits the edges of a :class:`torch_geometric.data.Data` object
    into positive and negative train/val/test edges.
    As such, it will replace the :obj:`edge_index` attribute with
    :obj:`train_pos_edge_index`, :obj:`train_pos_neg_adj_mask`,
    :obj:`val_pos_edge_index`, :obj:`val_neg_edge_index` and
    :obj:`test_pos_edge_index` attributes.
    If :obj:`data` has edge features named :obj:`edge_attr`, then
    :obj:`train_pos_edge_attr`, :obj:`val_pos_edge_attr` and
    :obj:`test_pos_edge_attr` will be added as well.

    .. warning::

        :meth:`~torch_geometric.utils.train_test_split_edges` is deprecated and
        will be removed in a future release.
        Use :class:`torch_geometric.transforms.RandomLinkSplit` instead.

    Args:
        data (Data): The data object.
        val_ratio (float, optional): The ratio of positive validation edges.
            (default: :obj:`0.05`)
        test_ratio (float, optional): The ratio of positive test edges.
            (default: :obj:`0.1`)

    :rtype: :class:`torch_geometric.data.Data`
    """

    assert 'batch' not in data  # No batch-mode.

    num_nodes = data.num_nodes
    row, col = data.edge_index
    edge_attr = data.edge_attr
    data.edge_index = data.edge_attr = None

    # Return upper triangular portion.
    mask = row < col
    row, col = row[mask], col[mask]

    if edge_attr is not None:
        edge_attr = edge_attr[mask]

    n_v = int(math.floor(val_ratio * row.size(0)))
    n_t = int(math.floor(test_ratio * row.size(0)))

    # Positive edges.
    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]
    if edge_attr is not None:
        edge_attr = edge_attr[perm]

    r, c = row[:n_v], col[:n_v]
    data.val_pos_edge_index = torch.stack([r, c], dim=0)
    if edge_attr is not None:
        data.val_pos_edge_attr = edge_attr[:n_v]

    r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
    data.test_pos_edge_index = torch.stack([r, c], dim=0)
    if edge_attr is not None:
        data.test_pos_edge_attr = edge_attr[n_v:n_v + n_t]

    r, c = row[n_v + n_t:], col[n_v + n_t:]
    data.train_pos_edge_index = torch.stack([r, c], dim=0)
    '''
    if edge_attr is not None:
        out = to_undirected(data.train_pos_edge_index, edge_attr[n_v + n_t:])
        data.train_pos_edge_index, data.train_pos_edge_attr = out
    else:
        data.train_pos_edge_index = to_undirected(data.train_pos_edge_index)
    '''

    # Negative edges.
    neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
    neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
    neg_adj_mask[row, col] = 0

    neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
    perm = torch.randperm(neg_row.size(0))[:n_v + n_t]
    neg_row, neg_col = neg_row[perm], neg_col[perm]

    neg_adj_mask[neg_row, neg_col] = 0
    data.train_neg_adj_mask = neg_adj_mask

    row, col = neg_row[:n_v], neg_col[:n_v]
    data.val_neg_edge_index = torch.stack([row, col], dim=0)

    row, col = neg_row[n_v:n_v + n_t], neg_col[n_v:n_v + n_t]
    data.test_neg_edge_index = torch.stack([row, col], dim=0)

    return data
