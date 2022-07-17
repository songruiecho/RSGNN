import torch
import torch as th
import math

import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_sparse import spmm   # product between dense matrix and sparse matrix
import torch.nn.functional as F
import config

class GraphConvolution(Module):
    """
    Simple pygGCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(th.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(th.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, infeatn, adj):
        support = th.spmm(infeatn, self.weight)
        output = th.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(Module):
    def __init__(self, nhid, drop=0.2, nclass=2):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(768, nhid)
        self.gc2 = GraphConvolution(768, nhid)
        self.gc = GraphConvolution(nhid, nclass)
        self.mlp = th.nn.Linear(768, nclass)
        self.gate = th.nn.Linear(nclass * 2, nclass)
        self.dropout = drop

    def forward(self, x, adj, _):
        # x_mlp = th.relu(self.mlp(x))
        x = th.dropout(x, self.dropout, train=self.training)
        x1 = self.gc1(x, adj)
        x = th.dropout(x1, self.dropout, train=self.training)
        x = self.gc(x, adj)

        return x

class RSGCN(Module):
    def __init__(self, nhid, L=2, drop=0.1, fusion='att', nclass=2):
        super(RSGCN, self).__init__()
        self.fusion = fusion
        self.conv1r1 = GraphConvolution(768, nhid)
        self.conv1r2 = GraphConvolution(768, nhid)
        self.mlp1 = nn.Linear(768, nhid)
        self.af1 = AttentionFusion(nhid)
        self.conv2r1 = GraphConvolution(nhid, nclass)
        self.conv2r2 = GraphConvolution(nhid, nclass)
        self.mlp2 = nn.Linear(nhid, nclass)
        self.af2 = AttentionFusion(nclass)
        self.dropout = drop

    def forward(self, x, adj, s_adj):
        x = th.dropout(x, self.dropout, train=self.training)
        x_r1 = self.conv1r1(x, adj)
        x_r2 = self.conv1r2(x, s_adj)
        x_1 = torch.tanh(self.mlp1(x))
        # x1 = x_r2 + x_r1 + x_1
        x1 = self.af1(x_r1, x_r2, x_1)
        # x1 = torch.max(torch.stack([x_r1, x_r2, x_1], dim=-1), dim=-1)[0]
        self.x1 = x1
        x1 = th.dropout(x1, self.dropout, train=self.training)
        x_r1 = self.conv2r1(x1, adj)
        x_r2 = self.conv2r2(x1, s_adj)
        x_1 = torch.tanh(self.mlp2(x1))
        x2 = x_r2 + x_r1 + x_1
        self.x2 = x2
        # x2 = torch.max(torch.stack([x_r1, x_r2, x_1], dim=-1), dim=-1)[0]
        return x2

class AttentionFusion(Module):
    def __init__(self, nhid):
        super(AttentionFusion, self).__init__()
        self.w = th.nn.Parameter(th.FloatTensor(3*nhid, 3)).cuda()
        stdv = 1. / math.sqrt(3*nhid)
        self.w.data.uniform_(-stdv, stdv)

    def forward(self, x1, x2, x3):
        a = th.mm(torch.cat([x1, x2, x3], dim=-1), self.w)
        a = torch.abs(a)
        a = th.softmax(a, dim=-1).unsqueeze(1)
        a = a.repeat(1, x1.shape[1], 1)
        x = th.stack([x1, x2, x3], dim=-1)
        x = th.sum(x*a, dim=-1).squeeze()
        return x
