import os
import torch
import requests
import logging
import shutil
from typing import Optional
import numpy as np
import scipy.sparse as sp
import networkx as nx
import scipy.io
from dgl import DGLGraph
import dgl
logger = logging.getLogger()

class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.dataset_name = self.config.dataset.name
        self.cache_dir = self.config.dataset.cache_dir
        self.self_loop = self.config.dataset.self_loop

    def get(self,):
        if self.cache_dir is None:
            self.cache_dir = os.path.join(os.environ["APP_HOME"], "datasets")
        self.file_path = os.path.join(self.cache_dir, self.dataset_name+'.pt')
        self.zip_path = os.path.join(self.cache_dir, self.dataset_name+'.pt.zip')
        if os.path.exists(self.file_path):
            data = torch.load(self.file_path)
        else:
            url = "https://github.com/pygod-team/data/raw/main/" + self.dataset_name + ".pt.zip"
            if not os.path.exists(self.cache_dir):
                os.mkdir(self.cache_dir)
            res = requests.get(url=url, stream=True)
            if res.status_code != 200:
                logger.error(">>>src.data.data_preprocess.py|>>>DataPreprocess.get()|>>>Failed downloading url %s" % url)
                raise RuntimeError("Failed downloading url %s" % url)
            with open(self.zip_path, "wb") as f:
                for chunk in res.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            shutil.unpack_archive(self.zip_path, self.cache_dir)
            data = torch.load(self.file_path)
        logger.info(">>>src.data.data_preprocess.py|>>>DataPreprocess.get()|>>>Get dataset {} successfully!".format(self.dataset_name))
        return data

    def sparse_to_tuple(self, sparse_mx):
        if not sp.isspmatrix_coo(sparse_mx):
            sparse_mx = sparse_mx.tocoo()
        coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
        values = sparse_mx.data
        shape = sparse_mx.shape
        return coords, values, shape

    def laplacian_positional_encoding(self, graph, pos_enc_dim):
        """
            Graph positional encoding v/ Laplacian eigenvectors
        """
        adj_matrix = graph.adjacency_matrix()
        indices = adj_matrix.indices()
        values = adj_matrix.val
        shape = adj_matrix.shape
        indices_cpu = [idx.cpu().numpy() if hasattr(idx, 'cpu') else idx for idx in indices]
        values_cpu = values.cpu().numpy() if hasattr(values, 'cpu') else values
        A = sp.coo_matrix((values_cpu, (indices_cpu[0], indices_cpu[1])), shape=shape).tocsr().astype(float)
        N = sp.diags(dgl.backend.asnumpy(graph.in_degrees()).clip(1) ** -0.5, dtype=float)
        L = sp.eye(graph.number_of_nodes()) - N * A * N
        EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR', tol=1e-2)
        EigVec = EigVec[:, EigVal.argsort()]
        lap_pos_enc = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float()
        return lap_pos_enc

    def re_features(self, adj, features, K):
        nodes_features = torch.empty(features.shape[0], 1, K+1, features.shape[1])
        for i in range(features.shape[0]):
            nodes_features[i, 0, 0, :] = features[i]
        x = features + torch.zeros_like(features)
        for i in range(K):
            x = torch.matmul(adj, x)
            for index in range(features.shape[0]):
                nodes_features[index, 0, i + 1, :] = x[index]
        nodes_features = nodes_features.squeeze()
        return nodes_features

    def preprocess(self, data):
        if self.dataset_name in ['books', 'disney', 'enron', 'reddit', 'weibo']:
            features = data.x.numpy()
            edge_index = data.edge_index.numpy()
            adj_data = np.ones_like(edge_index[0])
            adj = sp.csr_matrix((adj_data, (edge_index[0], edge_index[1])), shape=[features.shape[0], features.shape[0]])
            adj = sp.lil_matrix(adj)
            labels = data.y.numpy()
            labels = labels.astype(int)
        else:
            pass
        ### 预处理过程
        adj_ = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj_ = adj_ + sp.eye(adj_.shape[0])
        adj_array = sp.csr_array(adj_)
        graph = nx.from_scipy_sparse_array(adj_array, create_using=nx.DiGraph())
        adj_label = adj + sp.eye(adj.shape[0])
        adj_label = self.sparse_to_tuple(adj_label)
        if self.dataset_name in ['cora', 'citeseer', 'pubmed', 'BlogCatalog', 'Flickr', 'ACM']:
            features_ = self.sparse_to_tuple(features.tocoo())
            num_features = features_[2][1]
            features_nonzero = features_[1].shape[0]
            features = np.asarray(features.todense())
        else:
            num_features = features.shape[1]
            features_nonzero = num_features
        if self.self_loop:
            graph.remove_edges_from(nx.selfloop_edges(graph))
            graph.add_edges_from(zip(graph.nodes(), graph.nodes()))
        data.graph = DGLGraph(graph) # 图
        data.features_nonzero = features_nonzero # 非零特征数量
        data.adj_label = adj_label # 邻接标签
        data.num_labels = np.max(labels) # 标签数量
        data.adj_new = torch.FloatTensor(adj_.todense()) # 邻接矩阵 
        if self.config.model.name == 'gel':
            lpe = self.laplacian_positional_encoding(data.graph, pos_enc_dim=self.config.model.pe_dim)
            x_concat = torch.cat((data.x, lpe), dim=1)
            x_processed = self.re_features(data.adj_new, x_concat, K=self.config.model.hops)
            data.x_concat = x_concat
            data.x_processed = x_processed
        else:
            pass
        return data