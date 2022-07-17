# 加载数据集的方法
# 加载的数据内容包括：图；标签集合；mask
import numpy as np
import scipy.sparse as sp
import torch
import os
import sys
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import config
import pickle
from itertools import combinations
import networkx as nx
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from scipy.spatial import distance

class Dataloader(object):
    def __init__(self, fold):
        self.labels = []
        self.fold = fold
        self.g = nx.Graph()
        self.tfidf_g = nx.Graph()
        self.get_graph()
        self.get_tfidf_graph()
        self.get_features()

    def get_graph(self):
        '''
        :return:  获取基于社区的graph
        '''
        with open('../datas/{}/clean_data.txt'.format(config.dataset), 'r') as rf:
            self.datas = [each.strip().split(':') for each in rf.readlines()]
        with open('../datas/{}/fold/fold{}test.txt'.format(config.dataset, self.fold)) as rf:
            self.test = [each.strip().split(':') for each in rf.readlines()]  # 【idx of sentence, label】
        with open('../datas/{}/fold/fold{}train.txt'.format(config.dataset, self.fold)) as rf:
            self.train = [each.strip().split(':') for each in rf.readlines()]
        for i in range(len(self.datas)):
            self.g.add_node(i)
            self.tfidf_g.add_node(i)
        with open('../datas/{}/community.pkl'.format(config.dataset), 'rb') as rf:
            communities = pickle.load(rf)
            for person in communities.keys():
                community = communities[person]
                if len(community) < 2:
                    continue
                cs = combinations(community, 2)  # complete graph for the community
                for edge in cs:
                    self.g.add_edge(edge[0], edge[1], weight=1)
        self.g = self.preprocess_adj(nx.to_numpy_matrix(self.g))

    def get_features(self):
        ''' load features from pre-trained models
        :return:
        '''
        if 'features_fold{}.npy'.format(self.fold) not in os.listdir('../datas/{}/'.format(config.dataset)):
            bert = BertModel.from_pretrained('../datas/{}/fold-{}'.format(config.dataset, self.fold)).cuda()
            self.tokenizer = BertTokenizer.from_pretrained(config.bert_path)
            features = []
            with open('../datas/{}/clean_data.txt'.format(config.dataset), 'r') as rf:
                datas = [each.strip().split(':') for each in rf.readlines()]
            for i in tqdm(range(len(datas))):
                data = datas[i]
                self.labels.append(int(data[0]))  # int labels
                inputs = self.tokenizer(data[1])
                ids = torch.from_numpy(np.array([inputs['input_ids']])).cuda()
                mask = torch.from_numpy(np.array([inputs['attention_mask']])).cuda()
                out = bert(input_ids=ids, attention_mask=mask)[1]
                out = out.cpu().detach().numpy()[0]
                features.append(out)
            self.features = np.array(features)
            np.save('../datas/{}/features_fold{}.npy'.format(config.dataset, self.fold), self.features)
        else:
            print('load pre-trained features')
            self.features = np.load('../datas/{}/features_fold{}.npy'.format(config.dataset, self.fold))

    def get_data(self):
        '''
        :return:
        '''
        self.train_index = []
        self.test_index = []
        self.labels = [0]*len(self.datas)
        for each in self.test:
            self.test_index.append(int(each[0]))
            self.labels[int(each[0])] = int(each[1])
        for each in self.train:
            self.train_index.append(int(each[0]))
            self.labels[int(each[0])] = int(each[1])
        self.labels = torch.from_numpy(np.array(self.labels)).long()
        self.features = torch.from_numpy(np.array(self.features)).float()
        return self.features, self.g, self.tfidf_g, self.labels, self.train_index, self.test_index

    def preprocess_adj(self, adj, is_sparse=config.is_sparse):
        """Preprocessing of adjacency matrix for simple pygGCN model and conversion to
        tuple representation."""
        # if config.model_name != 'RGCN':
        #     adj_normalized = self.normalize_adj(adj + sp.eye(adj.shape[0]))
        # else:
        #     adj_normalized = self.normalize_adj(adj)
        adj_normalized = self.normalize_adj(adj + sp.eye(adj.shape[0]))
        # adj_normalized = adj = sp.coo_matrix(adj + sp.eye(adj.shape[0]))
        if is_sparse:
            adj_normalized = self.sparse_mx_to_torch_sparse_tensor(adj_normalized)
            return adj_normalized
        else:
            return torch.from_numpy(adj_normalized.A).float()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def normalize_adj(self, adj):
        """Symmetrically normalize adjacency matrix."""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    def get_tfidf_vec(self):
        """ get tf-idf cos sim
        :param content_lst:
        :return:
        """
        text_tfidf = Pipeline([
            ("vect", CountVectorizer(min_df=1,
                                     max_df=1.0,
                                     token_pattern=r"\S+",
                                     )),
            ("tfidf", TfidfTransformer(norm=None,
                                       use_idf=True,
                                       smooth_idf=False,
                                       sublinear_tf=False
                                       ))
        ])
        contents = [each[-1] for each in self.datas]
        tfidf_vec = text_tfidf.fit_transform(contents)
        tfidf_vec = torch.from_numpy(tfidf_vec.toarray())
        normal_vec = torch.norm(tfidf_vec, dim=-1).unsqueeze(-1)  # l2 norm
        normal_vec = torch.mm(normal_vec, normal_vec.t())
        cos_sim = torch.mm(tfidf_vec, tfidf_vec.t())   # calculate cos sim
        cos_sim = cos_sim/normal_vec
        cos_sim = cos_sim.detach().numpy()
        index = np.where(cos_sim>config.sim)

        self.tfidf_edges = []
        for i,j in tqdm(zip(index[0], index[1])):
            if i >= j:
                continue
            sim = cos_sim[i,j]
            self.tfidf_edges.append([i,j,sim])   # save cos sim
        print(len(self.tfidf_edges))
        self.tfidf_edges = np.array(self.tfidf_edges)
        np.save('../datas/{}/tfidf_edges_{}.npy'.format(config.dataset, str(config.sim)), self.tfidf_edges)

    def get_tfidf_graph(self):
        if 'tfidf_edges_{}.npy'.format(str(config.sim)) not in os.listdir('../datas/{}/'.format(config.dataset)):
            self.get_tfidf_vec()
        else:
            edges = np.load('../datas/{}/tfidf_edges_{}.npy'.format(config.dataset, str(config.sim)))
            for edge in edges:
                self.tfidf_g.add_edge(int(edge[0]), int(edge[1]), weight=edge[2])  # 社区内部的边的权重为1
        self.tfidf_g = self.preprocess_adj(nx.to_numpy_matrix(self.tfidf_g))

if __name__ == '__main__':
    for fff in [9,8,7,6,5]:
        loader = Dataloader(fold=fff)
    # loader.get_data()

