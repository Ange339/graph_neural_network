import torch
from torch_geometric.utils import negative_sampling
import logging

logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NegativeSampler:
    def __init__(self, cfg):
        self.cfg = cfg

    def sample(self, data):
        pos_edge_index = data['user', 'interacts', 'item'].edge_label_index
        users = data['user', 'interacts', 'item'].edge_label_index[0]
        items = data['user', 'interacts', 'item'].edge_label_index[1]
        num_users = data['user'].num_nodes
        num_items = data['item'].num_nodes
        num_neg_samples, negative_sampling_ratio = self.obtain_num_neg_samples(num_users, num_items, pos_edge_index.size(1))

        sampling_func = self.get_sampling_func()

        mask = torch.ones(num_neg_samples, dtype=torch.bool, device=device)
        neg_edge_index = sampling_func(data, mask)
        mask = self.collision_check(pos_edge_index, neg_edge_index)
        i = 0
        while mask.any() and i < 3:  # Limit to 3 attempts to avoid infinite loop
            neg_edge_index = sampling_func(data, mask, neg_edge_index[0], neg_edge_index[1])
            mask = self.collision_check(pos_edge_index, neg_edge_index)
            i += 1
        # logging.debug(f"Positive edges: {pos_edge_index[:,:10]} ...")  # Log first 10 positive edges
        # logging.debug(f"Negative edges: {neg_edge_index[:,:10]} ...")  # Log first 10 negative edges
        # uniq_cols, counts = torch.unique(neg_edge_index, dim=1, return_counts=True)
        # logging.debug(uniq_cols)
        # logging.debug(f"N. unique edges: {uniq_cols.size(1)/neg_edge_index.size(1):%}")
        # logging.debug(f"Negative sampling completed in {i+1} attempts. Number of false negatives: {mask.sum().item()}") if i > 0 else None
        neg_edge_label = torch.zeros(neg_edge_index.size(1), dtype=torch.float32, device=device)
        return neg_edge_index, neg_edge_label


    def get_sampling_func(self):
        sampling_strategy = self.cfg.get('negative_sampling_method', 'batch_random')
        if sampling_strategy == 'batch_random':
            return self.batch_random_sample
        elif sampling_strategy == 'pairwise_random':
            return self.pairwise_random_sample
        else:
            raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")


    def eval_sample(self, data):
        """Return directly val/test data with negative samples included."""
        neg_edge_index, neg_edge_label = self.sample(data)
        pos_edge_index, pos_edge_label = data['user', 'interacts', 'item'].edge_label_index, data['user', 'interacts', 'item'].edge_label
        edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        edge_label = torch.cat([pos_edge_label, neg_edge_label], dim=0)
        data['user', 'interacts', 'item'].edge_label_index = edge_label_index
        data['user', 'interacts', 'item'].edge_label = edge_label
        return data


    def batch_random_sample(self, data, mask, negative_users = None, negative_items = None):
        "Custom negative sampling: generate user-item pairs not in positive set."
        num_users = data['user'].num_nodes
        num_items = data['item'].num_nodes
        num_neg_samples = mask.sum().item()
        if negative_users is None or negative_items is None:
            negative_users = torch.randint(0, num_users, (num_neg_samples,), device=device)
            negative_items = torch.randint(0, num_items, (num_neg_samples,), device=device)
        else:
            negative_users[mask] = torch.randint(0, num_users, (num_neg_samples,), device=device)
            negative_items[mask] = torch.randint(0, num_items, (num_neg_samples,), device=device)
        neg_edge_index = torch.stack([negative_users, negative_items], dim=0)
        return neg_edge_index

    def pairwise_random_sample(self, data, mask, negative_users = None, negative_items = None):
        "Pairwise negative sampling: for each user, sample negative items."
        pos_edge_index = data['user', 'interacts', 'item'].edge_label_index
        users = pos_edge_index[0]
        num_items = data['item'].num_nodes
        
        num_neg_samples = mask.sum().item()
        if negative_items is None:
            negative_users = users.repeat_interleave(num_neg_samples // users.size(0))
            negative_items = torch.randint(0, num_items, (num_neg_samples,), device=device)
        else:
            negative_items[mask] = torch.randint(0, num_items, (num_neg_samples,), device=device)
        neg_edge_index = torch.stack([negative_users, negative_items], dim=0)
        return neg_edge_index


    def obtain_num_neg_samples(self, num_users, num_items, num_pos_edges):
        negative_sampling_ratio = self.cfg['negative_sampling_ratio']
        max_edges = num_users * num_items
        num_neg_samples = int(negative_sampling_ratio * num_pos_edges)
        if num_neg_samples > max_edges:
            negative_sampling_ratio = max_edges // num_pos_edges
            num_neg_samples = negative_sampling_ratio * num_pos_edges  # Otherwise sampling error
        return num_neg_samples, negative_sampling_ratio


    def collision_check(self, pos_edge_index, neg_edge_index):
        """
        For the collision, we will appy quotient-remainder theorem to hash the edges.
        Theorem: For any integer a and any positive integer b, there exist unique integers q and r such that
        a = bq + r and 0 <= r < b.

        """
        edge_max = max(pos_edge_index.max().item(), neg_edge_index.max().item()) + 1
        hashd_pos = (pos_edge_index[0, :]  + pos_edge_index[1:, :] * edge_max)[0]
        hashd_neg = (neg_edge_index[0, :]  + neg_edge_index[1:, :] * edge_max)[0]
        mask = torch.isin(hashd_neg, hashd_pos)
        return mask


    # def batch_random_sample(self, batch_data):
    #     "Custom negative sampling: generate user-item pairs not in positive set."
    #     pos_edge_index = batch_data['user', 'interacts', 'item'].edge_label_index
    #     users = batch_data['user', 'interacts', 'item'].edge_label_index[0]
    #     items = batch_data['user', 'interacts', 'item'].edge_label_index[1]
    #     num_users = batch_data['user'].num_nodes
    #     num_items = batch_data['item'].num_nodes

    #     num_neg_samples, negative_sampling_ratio = self.obtain_num_neg_samples(num_users, num_items, pos_edge_index.size(1))

    #     # Randomly sample user-item pairs. Given the sparsity of interactions, collisions with positive samples are rare.
    #     negative_users = torch.randint(0, num_users, (num_neg_samples,), device=device)
    #     negative_items = torch.randint(0, num_items, (num_neg_samples,), device=device)
    #     neg_edge_index = torch.stack([negative_users, negative_items], dim=0)
    #     mask = (neg_edge_index == pos_edge_index).all(dim=0)

    #     i = 0
    #     while mask.any() and i < 3:  # Limit to 3 attempts to avoid infinite loop, this shouldn't be a problem in practice
    #         neg_edge_index[:, mask] = torch.stack([
    #             torch.randint(0, num_users, (mask.sum().item(),), device=device),
    #             torch.randint(0, num_items, (mask.sum().item(),), device=device)
    #         ], dim=0)
    #         # Check for collisions with positive samples
    #         mask = (neg_edge_index == pos_edge_index).all(dim=0)
    #         i += 1

    #     neg_edge_label = torch.zeros(neg_edge_index.size(1), dtype=torch.float32, device=device)
    #     return neg_edge_index, neg_edge_label



    # def pairwise_random_sample(self, data):
    #     pos_edge_index = data['user', 'interacts', 'item'].edge_index
    #     users, pos_items = pos_edge_index
    #     num_items = pos_items.max().item()  # assumes items are 0..N-1

    #     # Sample negatives for each user
    #     negative_sampling_ratio = self.cfg['negative_sampling_ratio']
    #     num_neg_samples = negative_sampling_ratio * len(pos_items)

        
    #     pos_edge_index_replicated = pos_edge_index.repeat((1, negative_sampling_ratio))

    #     logger.info(pos_edge_index_replicated.size())

    #     neg_items = torch.randint(0, num_items, (num_neg_samples,), device=device)
    #     neg_edge_index = torch.stack([users.repeat((1, negative_sampling_ratio)), neg_items], dim=0)
    #     mask = (neg_edge_index == pos_edge_index).all(dim=0)
    #     i = 0
    #     while mask.any() and i < 3:  # Limit to 3 attempts to avoid infinite loop, this shouldn't be a problem in practice
    #         neg_edge_index[1, mask] = torch.randint(
    #             low=0, high=num_items, size=(mask.sum().item(),), device=device
    #         )
    #         mask = (neg_edge_index == pos_edge_index).all(dim=0)
    #         i += 1

    #     neg_edge_label = torch.zeros(neg_edge_index.size(1), dtype=torch.float32, device=device)

    #     return neg_edge_index, neg_edge_label




   
# def graph_random_sample(data, num_neg_samples, negative_sampling_ratio):
#     "Randomly sample negative examples from the graph dataset"
#     if not num_neg_samples:
#         num_neg_samples = int(len(data['user', 'interacts', 'item'].edge_label_index[0]) * negative_sampling_ratio)
#     edge_label_index = data['user', 'interacts', 'item'].edge_index
#     negative_samples = negative_sampling(edge_label_index, num_neg_samples=num_neg_samples)
#     return negative_samples, torch.zeros(num_neg_samples, dtype=torch.float32)
