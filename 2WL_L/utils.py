import torch
from torch import Tensor
from torch_scatter import scatter_add
from torch_geometric.nn.conv import edge_conv
from torch_geometric.utils.convert import to_cugraph
from torch_sparse.tensor import to

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
def sample_block(sample_idx, ea, size, ei, ei2):
    ea_new = ea[torch.logical_not(idx2mask(ei.shape[1], sample_idx))]
    ei_new = ei[:, torch.logical_not(idx2mask(ei.shape[1], sample_idx))]
    pos_pos_new = ei[:, sample_idx].t()
    ei2_new = blockei2(ei2, sample_idx)
    #print(ei_new, ea_new, (size, size))
    adj = torch.sparse_coo_tensor(ei_new, ea_new, (size, size))
    x_new = torch.sparse.sum(adj, dim=1).to_dense().to(torch.int64).reshape(size, 1)
    return ei_new, x_new, pos_pos_new, ei2_new

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
