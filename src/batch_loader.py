import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader
from torch_geometric.utils import negative_sampling

import logging

logger = logging.getLogger(__name__)

class BatchLoader:
    "Batch data loader"

    def __init__(self, cfg):
        self.cfg = cfg

    def load(self, data, shuffle):
        method = self.cfg.get('batch_method', 'binary_link_neighbors')
        if method == "binary_link_neighbors":
            return self.BinaryLinkNeighbors(data, shuffle)
        elif method == "node_neighbors":
            return self.NodeNeighLoaders(data, shuffle)
        else:
            raise ValueError(f"Unknown sampling method: {method}")


    def BinaryLinkNeighbors(self, data, shuffle):
        neg_sampling_recurrence = self.cfg.get('negative_sampling_recurrence', "epoch")

        if neg_sampling_recurrence == "fixed":
            num_edges = data['user', 'interacts', 'item'].edge_label_index.size(1)
            amount = int(num_edges * self.cfg['negative_sampling_ratio'])
            neg_sampling = NegativeSampling(mode='binary', amount=amount)
        else:
            neg_sampling = None
        loader = LinkNeighborLoader(data=data,
            num_neighbors= self.cfg['num_neighbors'],
            edge_label_index=(("user", "interacts", "item"), data['user', 'interacts', 'item'].edge_label_index),
            edge_label=data['user', 'interacts', 'item'].edge_label,
            batch_size=self.cfg['batch_size'],
            neg_sampling=neg_sampling,
            shuffle=shuffle,
        )
        return loader
    
    def NodeNeighLoaders(self, data, shuffle):
        loader = NeighborLoader(data=data,
            num_neighbors= self.cfg['num_neighbors'],
            batch_size=self.cfg['batch_size'],
            shuffle=shuffle,
            input_nodes=('user', data['user'].node_id)
        )
        return loader