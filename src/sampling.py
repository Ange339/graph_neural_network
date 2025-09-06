import torch
from torch_geometric.utils import negative_sampling



class NegativeSampler:
    def __init__(self, cfg, device='cpu'):
        self.cfg = cfg
        self.device = device


    def sample(self, data):
        sampling_strategy = self.cfg.get('negative_sampling_strategy', 'batch_random')
        if sampling_strategy == 'batch_random':
            return self.batch_random_sample(data)
        elif sampling_strategy == 'pairwise_random':
            return self.pairwise_random_sample(data)
        else:
            raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")


    def batch_random_sample(self, batch_data):
        "Randomly sample negative examples from the batch dataset"
        num_neg_samples = int(len(batch_data['user', 'interacts', 'item'].edge_label_index[0]) * self.cfg['negative_sampling_ratio'])
        #logger.info(f"Number of negative samples: {num_neg_samples}")
        edge_label_index = batch_data['user', 'interacts', 'item'].edge_label_index
        negative_samples = negative_sampling(edge_label_index, num_neg_samples=num_neg_samples).to(self.device)
        negative_samples_label = torch.zeros(negative_samples.size(1), dtype=torch.float32, device=self.device)
        return negative_samples, negative_samples_label


    def pairwise_random_sample(self, data):
        users, pos_items = data['user', 'interacts', 'item'].edge_index
        num_items = pos_items.max().item() + 1  # assumes items are 0..N-1

        # Sample negatives for each user
        num_neg_samples = self.cfg['negative_sampling_ratio'] * num_items

        if num_neg_samples > num_items:
            num_neg_samples = num_items  # Otherwise sampling error


        mask = torch.ones(num_neg_samples, dtype=torch.bool, device=self.device)
        i = 0
        while mask.any() and i < 3:  # Limit to 3 attempts to avoid infinite loop, this shouldn't be a problem in practice
            neg_items[mask] = torch.randint(
                low=0, high=num_items, size=(mask.sum().item(),), device=self.device
            )
            mask = neg_items == pos_items
            i += 1

        # Construct negative edge_index
        neg_edge_index = torch.stack([users, neg_items], dim=0)
        neg_edge_label = torch.zeros(neg_edge_index.size(1), dtype=torch.float32, device=self.device)

        return neg_edge_index, neg_edge_label


# def graph_random_sample(data, num_neg_samples, negative_sampling_ratio):
#     "Randomly sample negative examples from the graph dataset"
#     if not num_neg_samples:
#         num_neg_samples = int(len(data['user', 'interacts', 'item'].edge_label_index[0]) * negative_sampling_ratio)
#     edge_label_index = data['user', 'interacts', 'item'].edge_index
#     negative_samples = negative_sampling(edge_label_index, num_neg_samples=num_neg_samples)
#     return negative_samples, torch.zeros(num_neg_samples, dtype=torch.float32)
