from torch import Tensor
import torch
from torch_scatter import scatter_add
@torch.jit.script
def matmul(A: Tensor, B: Tensor, D: Tensor):
    '''
    A: (I,K)
    B: (J,K)
    D: (3,L), L is the number of non-zero elements in D. Three rows of D contain i, j, k respectively
    return : idx, val form a coo sparse matrix
    '''
    L = A.shape[-1]
    I = A.shape[0]
    J = B.shape[0]
    K = A.shape[1]
    idx_A = D[0]*K+D[2]
    val_A = A.reshape(-1,A.shape[2])[idx_A]
    idx_B = D[1]*K+D[2]
    val_B = B.reshape(-1,A.shape[2])[idx_B]
    val = val_A*val_B
    matidx = D[0]*J + D[1]
    return scatter_add(val, matidx, dim=0, dim_size=I*J).reshape(I, J, L)

'''
A = torch.rand((4,5,6))
B = torch.rand((3,5,6))
D = torch.tensor([[1,3,3],[2,1,1],[3,4,3]])
S = matmul(A, B, D)
print(torch.sum(S, dim=-1))
'''