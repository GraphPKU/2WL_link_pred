from torch import nn
import torch
import torch.nn.functional as F
from torch.nn.modules.dropout import Dropout
from torch_geometric.nn import GCNConv, GraphNorm, SAGEConv, APPNP, GINConv
from utils import reverse, sparse_bmm, sparse_cat, add_zero, edge_list
import time

class WLNet(torch.nn.Module):
    def __init__(self,
                 max_x,
                 use_feat=False,
                 feat=None,
                 hidden_dim_1=20,
                 hidden_dim_2=20,
                 layer1=2,
                 layer2=1,
                 layer3=1,
                 dp0_0 = 0.0,
                 dp0_1 = 0.0,
                 dp1=0.0,
                 dp2=0.0,
                 dp3=0.0,
                 ln0=True,
                 ln1=True,
                 ln2=True,
                 ln3=True,
                 ln4=True,
                 act0=False,
                 act1=False,
                 act2=False,
                 act3=True,
                 act4=True,
                 ):
        super(WLNet, self).__init__()

        self.use_feat = use_feat
        self.feat = feat
        use_affine = False

        relu_lin = lambda a, b, dp, lnx, actx: Seq([
            nn.Linear(a, b),
            nn.LayerNorm(b, elementwise_affine=use_affine) if lnx else nn.Identity(),
            nn.Dropout(p=dp, inplace=True),
            nn.ReLU(inplace=True) if actx else nn.Identity()])
        if feat is not None:
            self.lin1 = nn.Sequential(
                nn.Dropout(dp0_0),
                relu_lin(feat.shape[1], hidden_dim_1, dp0_1, ln0, act0)
            )

        Convs = lambda a, b, dp, lnx, actx: Seq([
            SAGEConv(a, b),
            nn.LayerNorm(b, elementwise_affine=use_affine) if lnx else nn.Identity(),
            nn.Dropout(p=dp, inplace=True),
            nn.ReLU(inplace=True) if actx else nn.Identity()])

        self.embedding = torch.nn.Sequential(torch.nn.Embedding(max_x + 1, hidden_dim_1),
                                             torch.nn.Dropout(p=dp1))
        #self.embedding = nn.Embedding(max_x + 1, latent_size_1)

        self.nconvs = nn.ModuleList([Convs(hidden_dim_1, hidden_dim_1, dp2, ln1, act1)] +
                                    [Convs(hidden_dim_1, hidden_dim_1, dp2, ln2, act2) for _ in range(layer1 - 1)]
                                    )

        input_edge_size = hidden_dim_1

        self.h_1 = Seq([relu_lin(input_edge_size + 1, hidden_dim_2, dp3, ln3, act3)] +
                       [relu_lin(hidden_dim_2, hidden_dim_2, dp3, ln3, act3) for _ in range(layer2 - 1)])

        self.g_1 = Seq([relu_lin(hidden_dim_2 * 2 + input_edge_size + 1, hidden_dim_2, dp3, ln4, act4)] +
                       [relu_lin(hidden_dim_2, hidden_dim_2, dp3, ln4, act4) for _ in range(layer3 - 1)])

        self.lin_dir = torch.nn.Linear(hidden_dim_2, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x, ei, pos, ei2=None, test=False):
        edge_index = ei
        n = x.shape[0]
        if self.use_feat:
            x = self.feat
            x = self.lin1(x)
        else:
            x = self.embedding(x)
        # x = F.relu(self.nlin1(x))

        for conv in self.nconvs:
            x = conv(x, edge_index)
        colx = x.unsqueeze(0).expand(n, -1, -1).reshape(n * n, -1)
        rowx = x.unsqueeze(1).expand(-1, n, -1).reshape(n * n, -1)
        x = rowx * colx
        x = x.reshape(n, n, -1)
        eim = torch.zeros((n * n,), device=x.device)
        eim[edge_index[0] * n + edge_index[1]] = 1
        eim = eim.reshape(n, n, 1)
        x = torch.cat((x, eim), dim=-1)
        x = mataggr(x, self.h_1, self.g_1)
        x = (x * x.permute(1, 0, 2)).reshape(n * n, -1)
        x = x[pos[:, 0] * n + pos[:, 1]]
        x = self.lin_dir(x)
        return x

class LocalWLNet(nn.Module):
    def __init__(self,
                 max_x,
                 use_node_feat,
                 node_feat,
                 channels_1wl=256,
                 channels_2wl=32,
                 depth1=1,
                 depth2=1,
                 dp_1wl=0.5,
                 dp_2wl=0.5,
                 act0 = True,
                 act1 = True,
                 use_affine = False,
                 ):
        super().__init__()

        

        relu_lin = lambda a, b, dp, lnx, actx: nn.Sequential(
            nn.Linear(a, b),
            nn.LayerNorm(b, elementwise_affine=use_affine) if lnx else nn.Identity(),
            nn.Dropout(p=dp, inplace=True),
            nn.ReLU(inplace=True) if actx else nn.Identity())

        relu_conv = lambda insize, outsize, dp, act, **kwargs: Seq([
            GCNConv(insize, outsize, **kwargs),
            nn.LayerNorm(outsize, elementwise_affine=use_affine),#GraphNorm(outsize),
            Dropout(p=dp, inplace=True),
            nn.ReLU(inplace=True) if act else nn.Identity()
        ])

        self.max_x = max_x
        self.use_node_feat = use_node_feat
        self.node_feat = node_feat
        if use_node_feat:
            self.lin1 = nn.Identity()
            '''
            nn.Sequential(
                nn.Dropout(dp_lin0),
                relu_lin(node_feat.shape[-1], channels_1wl, dp_lin1, True, False)
            )
            '''
        else:
            self.emb = None 
            '''
            nn.Sequential(nn.Embedding(max_x + 1, channels_1wl),
                                     GraphNorm(channels_1wl),
                                     Dropout(p=dp_emb, inplace=True))
            '''

        self.conv1s = nn.ModuleList(
            [relu_conv(node_feat.shape[-1], channels_1wl, dp_1wl, True)]+[relu_conv(channels_1wl, channels_1wl, dp_1wl, True) for _ in range(depth1 - 2)] +
            [relu_conv(channels_1wl, channels_1wl, dp_1wl, act0)])

        self.lin2 = nn.Linear(channels_1wl, channels_2wl)
        self.conv2s = nn.ModuleList(
            [relu_conv(channels_2wl, channels_2wl, dp_2wl, act1, decomposed_layers=4) for _ in range(depth2)])
        self.conv2s_r = nn.ModuleList(
            [relu_conv(channels_2wl, channels_2wl, dp_2wl, act1, decomposed_layers=4) for _ in range(depth2)])
        self.pred = nn.Sequential(nn.Linear(channels_2wl,channels_2wl), nn.ReLU(inplace=True), nn.Linear(channels_2wl, 1))
        self.ln = nn.LayerNorm(channels_2wl)

    def forward(self, x, edge1, pos, idx = None, ei2 = None, test = False):
        x = self.lin1(self.node_feat) if self.use_node_feat else self.emb(x).squeeze()
        for i in range(len(self.conv1s)):
            x = x + self.conv1s[i](x, edge1)
        x = self.lin2(x)
        x = self.ln(x)
        x = x[pos[:, 0]] * x[pos[:, 1]]
        edge2, edge2_r = reverse(ei2)
        for i in range(len(self.conv2s)):
            x = x + self.conv2s[i](x, edge2) + self.conv2s_r[i](x, edge2_r)
        x = x[idx]
        N = x.shape[0]
        x = x.reshape(N//2, 2, -1).mean(dim=1)
        x = self.pred(x)
        return x



class FWLNet(nn.Module):
    def __init__(self,
                 max_x,
                 use_feat=False,
                 feat=None,
                 hidden_dim_1=20,
                 hidden_dim_2=20,
                 layer1=2,
                 layer2=1,
                 dp1=0.0,
                 dp2=0.0,
                 dp3=0.0,
                 mul_pool=True,
                 use_ea=False,
                 easize=None,
                 act = True):
        super(FWLNet, self).__init__()
        self.mul_pool = mul_pool
        self.use_ea = use_ea
        self.use_feat = use_feat
        self.layer1 = layer1
        self.layer2 = layer2
        input_node_size = hidden_dim_1
        if use_feat:
            input_node_size += feat.shape[1]
            self.feat = nn.parameter.Parameter(feat, requires_grad=False)

        self.embedding = nn.Sequential(nn.Embedding(max_x + 1, hidden_dim_1),
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
                [relu_sage_end(input_node_size, hidden_dim_1, dp2)] + [
                    relu_sage_end(hidden_dim_1, hidden_dim_1, 0)
                    for i in range(layer1 - 1)
                ])
        else:
            self.nconvs = nn.ModuleList(
                [relu_sage(input_node_size, hidden_dim_1, dp2)] + [
                    relu_sage(hidden_dim_1, hidden_dim_1, dp2)
                    for i in range(layer1 - 1)
                ])

        input_edge_size = hidden_dim_1
        if use_ea:
            input_edge_size += easize.shape[1]

        relu_lin = lambda a, b, dp: nn.Sequential(
            nn.Linear(a, b), nn.Dropout(p=dp, inplace=True),
            nn.ReLU(inplace=True))
        self.mlps_1 = nn.ModuleList(
            [relu_lin(input_edge_size + 1, hidden_dim_2, dp3)] + [
                relu_lin(hidden_dim_2, hidden_dim_2, dp3)
                for i in range(layer2 - 1)
            ])
        self.mlps_2 = nn.ModuleList(
            [relu_lin(input_edge_size + 1, hidden_dim_2, dp3)] + [
                relu_lin(hidden_dim_2, hidden_dim_2, dp3)
                for i in range(layer2 - 1)
            ])
        relu_norm_lin = lambda a, b, dp: nn.Sequential(
            nn.Linear(a, b), GraphNorm(b), nn.Dropout(p=dp, inplace=True),
            nn.ReLU(inplace=True))
        self.mlps_3 = nn.ModuleList(
            [relu_norm_lin(hidden_dim_2 + input_edge_size + 1, hidden_dim_2, dp3)] +
            [
                relu_norm_lin(hidden_dim_2 * 2, hidden_dim_2, dp3)
                for i in range(layer2 - 1)
            ])

        self.lin_dir = nn.Linear(hidden_dim_2, 1)

    def forward(self, x, ei, pos, ei2=None, test=False):
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
            print("ERROR")
            exit(0)

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

class LocalFWLNet(nn.Module):
    def __init__(self,
                 max_x,
                 use_feat=False,
                 feat=None,
                 use_degree=True,
                 use_appnp=False,
                 reduce_feat=False,
                 sum_pooling=False,
                 hidden_dim_1wl=20,
                 hidden_dim_2wl=20,
                 layer1=2,
                 layer2=1,
                 layer3=1,
                 dp_emb=0.0,
                 dp_lin0=0.0,
                 dp_lin1=0.0,
                 dp_1wl=0.0,
                 dp_2wl0=0.0,
                 dp_2wl1=0.0,
                 alpha=0.1,
                 ln_lin=False,
                 ln_1wl=False,
                 ln_2wl0=False,
                 ln_2wl1=False,
                 gn_1wl=True,
                 gn_2wl1=True,
                 act_lin=False,
                 act_1wl=True,
                 act_2wl0=True,
                 act_2wl1=True,
                 fast_bsmm = False,
                 use_ea=False,
                 easize=None):
        super(LocalFWLNet, self).__init__()
        assert use_feat or use_degree
        self.use_ea = use_ea
        self.max_x = max_x
        self.use_feat = use_feat
        self.use_degree = use_degree
        self.use_appnp = use_appnp
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.fast = fast_bsmm
        self.sum_pooling = sum_pooling
        relu_sage = lambda a, b, dp, lnx, gnx, actx: Seq([
            GCNConv(a, b),
            nn.LayerNorm(b, elementwise_affine=use_affine) if lnx else nn.Identity(),
            GraphNorm(b) if gnx else nn.Identity(),
            nn.Dropout(dp, inplace=True),
            nn.ReLU(inplace=True) if actx else nn.Identity()
        ])
        relu_lin = lambda a, b, dp, lnx, gnx, actx: nn.Sequential(
            nn.Linear(a, b),
            nn.LayerNorm(b, elementwise_affine=use_affine) if lnx else nn.Identity(),
            GraphNorm(b) if gnx else nn.Identity(),
            nn.Dropout(p=dp, inplace=True),
            nn.ReLU(inplace=True) if actx else nn.Identity())
        use_affine = False
        input_node_size = hidden_dim_1wl if use_degree else 0
        if use_feat:
            input_node_size += feat.shape[1]
            self.feat = nn.parameter.Parameter(feat, requires_grad=False)
        self.embedding = nn.Sequential(
            nn.Embedding(max_x + 1, hidden_dim_1wl),
            nn.Dropout(p=dp_emb))
        if not use_appnp:
            self.nconvs = nn.ModuleList(
                [relu_sage(input_node_size, hidden_dim_1wl, dp_1wl, ln_1wl, gn_1wl, act_1wl)] + [
                    relu_sage(hidden_dim_1wl, hidden_dim_1wl, dp_1wl, ln_1wl, gn_1wl, act_1wl)
                    for _ in range(layer1 - 1)
                ])
        else:
            self.nconvs = APPNP(layer1, alpha)
        if reduce_feat:
            assert self.use_feat
            self.lin1 = nn.Sequential(
                nn.Dropout(dp_lin0),
                relu_lin(input_node_size, hidden_dim_1wl, dp_lin1, ln_lin, False, act_lin)
            )
        else:
            self.lin1 = nn.Identity()
        input_edge_size = hidden_dim_1wl

        self.mlps_1 = relu_lin(input_edge_size * 2, hidden_dim_2wl, dp_2wl0, ln_2wl0, False, act_2wl0)
        self.mlps_2 = nn.ModuleList(
            [relu_lin(input_edge_size * 2, hidden_dim_2wl, dp_2wl0, ln_2wl0, False, act_2wl0)] + [
             relu_lin(input_edge_size * 2, hidden_dim_2wl, dp_2wl0, ln_2wl0, False, act_2wl0)
             for _ in range(layer2 - 1)
            ])
        self.mlps_3 = nn.ModuleList(
            [relu_lin(hidden_dim_2wl + 1, hidden_dim_2wl, dp_2wl1, ln_2wl1, gn_2wl1, act_2wl1)
             for _ in range(layer3)
            ])
        if not sum_pooling:
            self.lin_dir = nn.Linear(hidden_dim_1wl + hidden_dim_2wl, 1)

    def forward(self, x, ei, pos, ei2=None, test=False):
        t0 = time.time()
        edge_index = ei
        #pos = pos1[pos2][:, 0].reshape(-1, 2)
        n = x.shape[0]

        x = self.embedding(x)
        t1 = time.time()
        if self.use_feat:
            x = torch.cat((x, self.feat), dim=1)
            if not self.use_degree:
                x = self.feat
        x = self.lin1(x)
        if not self.use_appnp:
            for i in range(self.layer1):
                x = self.nconvs[i](x, edge_index)
        else:
            x = self.nconvs(x, edge_index)
        t2 = time.time()
        xx = x[pos[:, 0]] * x[pos[:, 1]]

        val = torch.cat([x[edge_index[0]], x[edge_index[1]]], 1)#colx = x.unsqueeze(0).expand(n, -1, -1).reshape(n * n, -1)

        x = val.clone()
        t3 = time.time()
        x = self.mlps_1(x)
        current_edges = edge_index

        for i in range(self.layer3):
            if i < self.layer2:
                mul = self.mlps_2[i](val)
            x = sparse_bmm(current_edges, x, edge_index, mul, n, fast=self.fast)
            current_edges, value = sparse_cat(x, edge_index, torch.ones((edge_index.shape[1], 1), device=x.device))
            x = self.mlps_3[i](value)
        t4 = time.time()
        import pdb
        #pdb.set_trace()
        sm = torch.sparse.FloatTensor(torch.cat([current_edges[1].unsqueeze(0), current_edges[0].unsqueeze(0)], 0), x,
                                      torch.Size([n, n, x.shape[-1]])).coalesce().values()
        t5 = time.time()
        #pdb.set_trace()
        x = x * sm
        t6 = time.time()
        #pdb.set_trace()
        x = add_zero(x, pos.t().cpu().numpy(), current_edges)
        t7 = time.time()
        #pdb.set_trace()

        pred_list = edge_list(current_edges, pos.t(), n)
        t8 = time.time()
        #pdb.set_trace()
        x = x[pred_list]
        x = torch.cat([x, xx], 1)
        x = self.lin_dir(x) if not self.sum_pooling else torch.sum(x, dim=-1, keepdim=True)
        #t6 = time.time()
        #print(t1 - t0)
        #print(t2 - t1)
        #print(t3 - t2)
        #print(t4 - t3)
        #print(t5 - t4)
        #print(t6 - t5)
        #print(t7 - t6)
        #print(t8 - t7)
        return x


from torch_geometric.utils import degree, to_undirected
from torch_sparse import SparseTensor
from torch import Tensor



class WXYFWLNet(nn.Module):
    def __init__(self,
                 max_x=2000,
                 feat=None,
                 hidden_dim_1=4,
                 hidden_dim_2=4,
                 layer1=2,
                 layer2=2,
                 dp1=0.0,
                 dp2=0.0,
                 dp3=0.0,
                 cat="no"):
        super().__init__()
        assert cat in ["mul", "add", "no"]
        self.cat = cat
        self.layer1 = layer1
        self.layer2 = layer2
        input_node_size = hidden_dim_1
        if feat is not None:
            input_node_size += feat.shape[1]
            self.register_buffer("feat", feat)
        else:
            self.feat = None
        self.embedding = nn.Sequential(nn.Embedding(max_x + 1, hidden_dim_1),
                                       nn.Dropout(p=dp1))
        relu_sage = lambda a, b, dp: Seq([
            GCNConv(a, b),
            nn.LeakyReLU(inplace=True),
            nn.Linear(b, b),
            nn.LeakyReLU(inplace=True)
        ])
        self.nconvs = nn.ModuleList(
                [relu_sage(input_node_size, hidden_dim_1, dp2)] + [
                    relu_sage(hidden_dim_1, hidden_dim_1, dp2)
                    for i in range(layer1 - 1)
                ])
        if hidden_dim_1 != hidden_dim_2:
            self.lin1 = nn.Sequential(nn.Linear(hidden_dim_1, hidden_dim_2), nn.LayerNorm(hidden_dim_2), nn.ReLU(inplace=True))
        else:
            self.lin1 = nn.Identity()
        relu_lin = lambda a, b, dp: nn.Sequential(
            nn.Linear(a, b), 
            nn.LeakyReLU(inplace=True))
        self.mlps_1 = nn.ModuleList([
                relu_lin(hidden_dim_2, hidden_dim_2, dp3)
                for i in range(layer2)
            ])
        self.mlps_2 = nn.ModuleList([
                relu_lin(hidden_dim_2, hidden_dim_2, dp3)
                for i in range(layer2)
            ])
        self.lin_dir = nn.Linear(hidden_dim_2, 1)
        self.eiemb = nn.Embedding(3, hidden_dim_2)

    def onewl(self, x, ei):
        x = degree(x)
        x = self.embedding(x)
        if self.feat is not None:
            x = torch.cat((x, self.feat), dim=1)
        for i in range(self.layer1):
            x = x + self.nconvs[i](x, ei)
        return x

    def adjemb(self, ei, nnode: int):
        eim = torch.zeros((nnode, nnode), dtype=torch.long, device = ei.device)
        eim[ei[0], ei[1]] = 1
        return self.eiemb(eim)
        
    def forward(self, x, ei, tar_edge):
        norm = x.shape[0] ** (-0.5)
        x = self.onewl(x, ei)
        x = self.lin1(x)
        x = x.unsqueeze(0) * x.unsqueeze(1) 
        x = x * self.adjemb(ei, x.shape[0])
        for i in range(self.layer2):
            if self.cat == "mul":
                x = x * norm * (self.mlps_1[i](x).permute(2, 0, 1) @ self.mlps_2[i](x).permute(2, 0, 1)).permute(1, 2, 0)
            elif self.cat == "add":
                x = x + norm * (self.mlps_1[i](x).permute(2, 0, 1) @ self.mlps_2[i](x).permute(2, 0, 1)).permute(1, 2, 0)
            elif self.cat == "no":
                x = norm * (self.mlps_1[i](x).permute(2, 0, 1) @ self.mlps_2[i](x).permute(2, 0, 1)).permute(1, 2, 0)
        x = x[tar_edge[0], tar_edge[1]] + x[tar_edge[1], tar_edge[0]]
        x = self.lin_dir(x)
        return x


class Net_cora(nn.Module):

    def __init__(self,
                 max_x,
                 use_feat=False,
                 feat=None,
                 hidden_dim_1wl=20,
                 hidden_dim_2wl=20,
                 layer1=2,
                 layer2=1,
                 dp1=0.0,
                 dp2=0.0,
                 dp3=0.0,
                 fast_bsmm=False,
                 use_ea=False,
                 ln0=False,
                 ln1=False,
                 ln2=False,
                 ln3=False,
                 act1=False,
                 act2=False,
                 act3=False,
                 easize=None):
        super(Net_cora, self).__init__()
        self.use_ea = use_ea
        self.max_x = max_x
        self.use_feat = use_feat
        self.layer1 = layer1
        self.layer2 = layer2
        self.fast = fast_bsmm
        input_node_size = hidden_dim_1wl
        if use_feat:
            input_node_size += feat.shape[1]
            self.feat = nn.parameter.Parameter(feat, requires_grad=False)
        use_affine = False
        self.embedding = nn.Sequential(
            nn.Embedding(max_x + 1, hidden_dim_1wl),
            nn.LayerNorm(hidden_dim_1wl, elementwise_affine=use_affine)
            if ln0 else nn.Identity(), nn.Dropout(p=dp1))
        relu_sage = lambda a, b, dp: Seq([
            GCNConv(a, b),
            nn.LayerNorm(b, elementwise_affine=use_affine)
            if ln1 else nn.Identity(),
            nn.Dropout(dp, inplace=True),
            nn.ReLU(inplace=True) if act1 else nn.Identity(),
        ])
        self.nconvs = nn.ModuleList(
            [relu_sage(feat.shape[1], hidden_dim_1wl, dp2)] + [
                relu_sage(hidden_dim_1wl, hidden_dim_1wl, dp2)
                for i in range(layer1 - 1)
            ])
        input_edge_size = hidden_dim_1wl
        if use_ea:
            input_edge_size += easize.shape[1]
        relu_lin = lambda a, b, dp: nn.Sequential(
            nn.Linear(a, b),
            nn.LayerNorm(b, elementwise_affine=use_affine)
            if ln2 else nn.Identity(), nn.Dropout(p=dp, inplace=True),
            nn.ReLU(inplace=True) if act2 else nn.Identity())
        self.mlps_1 = relu_lin(input_edge_size * 2, hidden_dim_2wl, dp3)
        self.mlps_2 = nn.ModuleList(
            [relu_lin(input_edge_size * 2, hidden_dim_2wl, dp3)] + [
                relu_lin(input_edge_size * 2, hidden_dim_2wl, dp3)
                for i in range(layer2 - 1)
            ])
        relu_norm_lin = lambda a, b, dp: nn.Sequential(
            nn.Linear(a, b),
            nn.LayerNorm(b, elementwise_affine=use_affine)
            if ln3 else nn.Identity(), nn.Dropout(p=dp, inplace=True),
            nn.ReLU(inplace=True) if act3 else nn.Identity())
        self.mlps_3 = nn.ModuleList([
            relu_norm_lin(hidden_dim_2wl + 1, hidden_dim_2wl, dp3)
            for _ in range(layer2)
        ])
        self.lin_dir = nn.Linear(hidden_dim_1wl + hidden_dim_2wl, 1)

    def forward(self, x, ei, pos1, pos2, ei2=None, test=False):
        edge_index = ei
        pos = pos1[pos2][:, 0].reshape(-1, 2)

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

class Net(nn.Module):
    def __init__(self,
                 max_x,
                 use_feat=False,
                 feat=None,
                 hidden_dim_1wl=20,
                 hidden_dim_2wl=20,
                 layer1=2,
                 layer2=1,
                 dp_emb=0.0,
                 dp_1wl=0.0,
                 dp_2wl=0.0,
                 fast_bsmm = False,
                 use_ea=False,
                 easize=None):
        super(Net, self).__init__()
        self.nowl = layer1 == 0
        self.use_ea = use_ea
        self.max_x = max_x
        self.use_feat = use_feat
        self.layer1 = layer1
        self.layer2 = layer2
        self.fast = fast_bsmm
        input_node_size = hidden_dim_1wl
        if use_feat:
            input_node_size += feat.shape[1]
            self.feat = nn.parameter.Parameter(feat, requires_grad=False)

        self.embedding = nn.Sequential(nn.Embedding(max_x + 1, hidden_dim_1wl),
                                       nn.Dropout(p=dp_emb))
        relu_sage = lambda a, b, dp: Seq([
            GCNConv(a, b),
            GraphNorm(b),
            nn.Dropout(dp, inplace=True),
            nn.ReLU(inplace=True)
        ])
        self.nconvs = nn.ModuleList(
            [relu_sage(input_node_size, hidden_dim_1wl, dp_1wl)] + [
                relu_sage(hidden_dim_1wl, hidden_dim_1wl, dp_1wl)
                for i in range(layer1 - 1)
            ])

        input_edge_size = hidden_dim_1wl
        if use_ea:
            input_edge_size += easize.shape[1]

        relu_lin = lambda a, b, dp: nn.Sequential(
            nn.Linear(a, b), nn.Dropout(p=dp, inplace=True),
            nn.ReLU(inplace=True))
        self.mlps_1 = relu_lin(input_edge_size * 2, hidden_dim_2wl, dp_2wl)
        self.mlps_2 = nn.ModuleList(
            [relu_lin(input_edge_size * 2, hidden_dim_2wl, dp_2wl)] + [
                relu_lin(input_edge_size * 2, hidden_dim_2wl, dp_2wl)
                for i in range(layer2 - 1)
            ])
        relu_norm_lin = lambda a, b, dp: nn.Sequential(
            nn.Linear(a, b), GraphNorm(b), nn.Dropout(p=dp, inplace=True),
            nn.ReLU(inplace=True))
        self.mlps_3 = nn.ModuleList(
            [
                relu_norm_lin(hidden_dim_2wl + 1, hidden_dim_2wl, dp_2wl)
                for _ in range(layer2)
            ])

        self.lin_dir = nn.Linear(hidden_dim_1wl + hidden_dim_2wl, 1)

    def forward(self, x, ei, pos1, pos2, ei2=None, test=False):
        #import pdb
        #pdb.set_trace()
        t0 = time.time()
        edge_index = ei
        pos = pos1[pos2][:, 0].reshape(-1, 2)

        x = self.embedding(x)

        if self.use_feat:
            x = torch.cat((x, self.feat), dim=1)
        n = x.shape[0]
        for i in range(self.layer1):
            x = self.nconvs[i](x, edge_index)
        xx = x[pos[:, 0]] * x[pos[:, 1]]

        val = torch.cat([x[edge_index[0]], x[edge_index[1]]], 1)#colx = x.unsqueeze(0).expand(n, -1, -1).reshape(n * n, -1)

        if self.use_ea:
            val = torch.cat([val, ea], 0)

        x = val.clone()
        x = self.mlps_1(x)
        current_edges = edge_index

        for i in range(self.layer2):
            mul = self.mlps_2[i](val)
            x = sparse_bmm(current_edges, x, edge_index, mul, n, fast=self.fast)
            current_edges, value = sparse_cat(x, edge_index, torch.ones((edge_index.shape[1], 1), device=x.device))
            x = self.mlps_3[i](value)
        sm = torch.sparse.FloatTensor(torch.cat([current_edges[1].unsqueeze(0), current_edges[0].unsqueeze(0)], 0), x,
                                      torch.Size([n, n, x.shape[-1]])).coalesce().values()

        x = x * sm
        x = add_zero(x, pos.t().cpu().numpy(), current_edges)

        pred_list = edge_list(current_edges, pos.t(), n)
        x = x[pred_list]
        x = torch.cat([x, xx], 1)

        x = self.lin_dir(x)
        return x

class Seq(nn.Module):
    def __init__(self, modlist):
        super().__init__()
        self.modlist = nn.ModuleList(modlist)

    def forward(self, *args, **kwargs):
        out = self.modlist[0](*args, **kwargs)
        for i in range(1, len(self.modlist)):
            out = self.modlist[i](out)
        return out

def mataggr(A, h, g):
    '''
    A (n, n, d). n is number of node, d is latent dimension
    h, g are mlp
    '''
    B = h(A)
    #C = f(A)
    n, d = A.shape[0], A.shape[1]
    vec_p = (torch.sum(B, dim=1, keepdim=True)).expand(-1, n, -1)
    vec_q = (torch.sum(B, dim=0, keepdim=True)).expand(n, -1, -1)
    D = torch.cat([A, vec_p, vec_q], -1)
    return g(D)
