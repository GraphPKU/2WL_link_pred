import torch
from torch import Tensor
from torch_geometric.utils import to_undirected, remove_self_loops
from torch_scatter import scatter_add
import math


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
def check_in_set(target: Tensor, set: Tensor):
    # target (n,), set(m,)
    a = target.reshape(-1, 1)
    b = set.reshape(1, -1)
    out = []
    cutshape = (1 << 28) // b.shape[1]
    out = torch.cat([
        torch.sum((a[i:i + cutshape] == b), dim=-1, dtype=torch.bool)
        for i in range(0, a.shape[0], cutshape)
    ])
    return out


@torch.jit.script
def edgegraph(pos1: Tensor):
    n_node1 = pos1.shape[0]
    idx1 = torch.arange(n_node1, device=pos1.device)
    cutshape = (1 << 24) // pos1.shape[0]
    edge2 = []
    pos_b = pos1.t().unsqueeze(-2)[[0, 1, 0, 1]]  # (4, 1, -1)
    for i in range(0, n_node1, cutshape):
        idx_a = idx1[i:i + cutshape]
        pos_a = pos1[i:i + cutshape].t().unsqueeze(-1)[[0, 0, 1, 1]
                                                       ]  # (4, -1, 1)
        tpos = torch.sum(pos_a == pos_b, dim=0, dtype=torch.bool).flatten()
        edge2.append(set_mul(idx_a, idx1)[tpos])
    edge2 = torch.cat(edge2, dim=0).t()
    return remove_self_loops(edge2)[0]


@torch.jit.script
def partial_edgegraph(pos1: Tensor, pos2: Tensor):
    n_node1 = pos1.shape[0]
    n_node2 = pos2.shape[0]
    idx1 = torch.arange(n_node1, device=pos1.device)
    idx2 = n_node1 + torch.arange(n_node2, device=pos1.device)
    cutshape = (1 << 24) // pos2.shape[0]
    edge2 = []
    pos_b = pos2.t().unsqueeze(-2)[[0, 1, 0, 1]]
    for i in range(0, n_node1, cutshape):
        idx_a = idx1[i:i + cutshape]
        pos_a = pos1[i:i + cutshape].t().unsqueeze(-1)[[0, 0, 1, 1]]
        tpos = torch.sum(pos_a == pos_b, dim=0, dtype=torch.bool).flatten()
        edge2.append(set_mul(idx_a, idx2)[tpos])
    edge2 = torch.cat(edge2, dim=0).t()
    return edge2


def enlarge_edgegraph(ei2: Tensor, pos1: Tensor, pos2: Tensor):
    pei2 = to_undirected(partial_edgegraph(pos1, pos2))
    ei22 = to_undirected(pos1.shape[0] + edgegraph(pos2))
    return torch.cat((ei2, pei2, ei22), dim=1)


@torch.jit.script
def idx2mask(num: int, idx: Tensor):
    mask = torch.zeros((num), device=idx.device, dtype=torch.bool)
    mask[idx] = True
    return mask


@torch.jit.script
def mask2idx(mask: Tensor):
    idx = torch.arange(mask.shape[0], device=mask.device)
    return idx[mask]




@torch.jit.script
def setmul4compute_D(a, b, c):
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


@torch.jit.script
def compute_D(ei, num_node: int, self_loop: bool=True):
    vec_i, vec_k = ei[0], ei[1]
    vec_j = torch.arange(num_node, device=ei.device)
    tD = setmul4compute_D(vec_i, vec_k, vec_j)
    D = torch.cat((tD[[0, 2, 1]], tD[[2, 0, 1]]), dim=-1)
    if self_loop:
        vec_i = torch.arange(num_node, device=ei.device)
        vec_j = vec_i
        vec_k = vec_i
        tD = setmul4compute_D(vec_i, vec_i, vec_j)
        D = torch.cat((D, tD[[0, 2, 1]], tD[[2, 0, 1]]), dim=-1)
    D = torch.unique(D, dim=-1)
    return D

def random_split_edges(data, val_ratio: float = 0.05,
                           test_ratio: float = 0.1):

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
    if edge_attr is not None:
        data.train_pos_edge_attr = edge_attr[n_v + n_t:]

    return data