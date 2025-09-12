import torch
from torch_geometric.nn import Node2Vec
from torch_geometric.utils import to_scipy_sparse_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import SpectralEmbedding

import torch.nn as nn

class MatrixFactorizationEmbedding(nn.Module):
    """
    Simple matrix factorization for node embeddings using SVD.
    """
    def __init__(self, edge_index, cfg):
        super().__init__()
        # Build adjacency matrix
        adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)
        svd = TruncatedSVD(n_components=cfg.get('embedding_dim', 64))
        self.embeddings = torch.tensor(svd.fit_transform(adj), dtype=torch.float)

    def forward(self):
        return self.embeddings


class SpectralNodeEmbedding(nn.Module):
    """
    Spectral embedding for nodes using sklearn's SpectralEmbedding.
    """
    def __init__(self, edge_index, cfg):
        super().__init__()
        cfg = cfg['spectral']
        adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)
        embedding = SpectralEmbedding(n_components=cfg.get('embedding_dim', 64), affinity='precomputed')
        adj_dense = adj.toarray()
        self.embeddings = torch.tensor(embedding.fit_transform(adj_dense), dtype=torch.float)

    def forward(self):
        return self.embeddings


class Node2VecEmbedding(nn.Module):
    """
    Node2Vec embedding using torch_geometric's implementation.
    """
    def __init__(self, edge_index, cfg):
        super().__init__()
        self.model = Node2Vec(
            edge_index, 
            embedding_dim=cfg.get('embedding_dim', 64), 
            walk_length=cfg.get('walk_length', 20),
            context_size=cfg.get('context_size', 10), 
            walks_per_node=cfg.get('walks_per_node', 10),
            num_nodes=cfg.get('num_nodes', 0), 
            sparse=True
        )

    def forward(self):
        return self.model.embedding.weight