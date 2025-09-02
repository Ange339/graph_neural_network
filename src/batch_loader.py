import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader


class BatchLoader:
    "Batch data loader"

    def __init__(self, cfg):
        self.cfg = cfg

    def load(self, data, shuffle):
        method = self.cfg.get('batch_method', 'binary_link_neighbors')
        if method == "binary_link_neighbors":
            return self.BinaryLinkNeighbors(data, shuffle)
        else:
            raise ValueError(f"Unknown sampling method: {method}")


    def BinaryLinkNeighbors(self, data, shuffle):
        loader = LinkNeighborLoader(data=data,
            num_neighbors=self.cfg['num_neighbors'],
            edge_label_index=(("user", "interacts", "item"), data['user', 'interacts', 'item'].edge_label_index),
            edge_label=data['user', 'interacts', 'item'].edge_label,
            batch_size=self.cfg['batch_size'],
            #neg_sampling=NegativeSampling(mode='binary', amount=cfg['num_negative_sampling']), I will make my version
            shuffle=shuffle,
        )
        return loader
