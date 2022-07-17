import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch_sparse import spmm   # product between dense matrix and sparse matrix
import math
'''
GRAPH ATTENTION NETWORKS, ICLR2018
'''


class SparseGATLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, input_dim, out_dim, dropout=0, alpha=0.1, concat=True):
        super(SparseGATLayer, self).__init__()
        self.in_features = input_dim
        self.out_features = out_dim
        self.alpha = alpha
        self.concat = concat
        self.dropout = dropout
        self.W = nn.Parameter(torch.zeros(size=(input_dim, out_dim))).cuda()  # FxF'
        self.attn = nn.Parameter(torch.zeros(size=(1, 2 * out_dim))).cuda()  # 2F'
        nn.init.xavier_normal_(self.W, gain=1.414)
        nn.init.xavier_normal_(self.attn, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, x, adj):
        '''
        :param x:   dense tensor. size: nodes*feature_dim
        :param adj:    parse tensor. size: nodes*nodes
        :return:  hidden features
        '''
        N = x.size()[0]
        edge = adj._indices()
        if x.is_sparse:
            h = torch.sparse.mm(x, self.W)
        else:
            h = torch.mm(x, self.W)
        # Self-attention (because including self edges) on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()  # edge_h: 2*D x E
        values = self.attn.mm(edge_h).squeeze()   # Features are projected using attention parameters
        edge_e_a = self.leakyrelu(values)  # edge_e_a: E   attetion score for each edge
        # softmax exp(each-max) first
        edge_e = torch.exp(edge_e_a - torch.max(edge_e_a))
        # Simulate row sums by multiplying them with the identity matrix
        e_rowsum = spmm(edge, edge_e, m=N, n=N, matrix=torch.ones(size=(N, 1)).cuda())  # e_rowsum: N x 1
        h_prime = spmm(edge, edge_e, n=N,m=N, matrix=h)
        h_prime = h_prime.div(e_rowsum + torch.Tensor([9e-15]).cuda())  # h_prime: N x out
        # softmax over
        if self.concat:
            # if this layer is not last layer
            return self.leakyrelu(h_prime)
        else:
            # if this layer is last layer
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, outputs, heads=2, nclass=2, dropout=0.1, alpha=0.01):
        """Sparse version of GAT, without multi-head"""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.hidden = outputs
        # multi-head GAT
        # self.linear = torch.nn.Linear(768, outputs)
        self.gc1 = [SparseGATLayer(768, outputs//heads)
                           for _ in range(heads)]
        self.gc2 = [SparseGATLayer(768, outputs // heads)
                    for _ in range(heads)]
        self.attention_out = SparseGATLayer(outputs, nclass)
        for i, attention in enumerate(self.gc1):
            self.add_module('attention_{}'.format(i), attention)
        for i, attention in enumerate(self.gc2):
            self.add_module('attention_{}'.format(i), attention)
        self.mlp = torch.nn.Linear(768, nclass)
        self.gate = torch.nn.Linear(nclass * 2, nclass)

    def forward(self, x, adj, s_adj):
        # x_mlp = torch.relu(self.mlp(x))
        # adj = adj + s_adj
        x1 = torch.cat([att(x, adj) for att in self.gc1], dim=1)  # 主要的GCN
        # x2 = torch.cat([att(x, s_adj) for att in self.gc2], dim=1)
        # x = torch.cat([x1, x2], dim=-1)
        # x = torch.relu(x)
        # x = torch.dropout(x1, self.dropout, train=self.training)
        x = self.attention_out(x1, adj)
        # z = self.gate(torch.cat([x, x_mlp], dim=-1))
        return x