import numpy as np
import polars as pl
import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
from torchmetrics.classification import AveragePrecision, AUROC
from torch_geometric.utils import negative_sampling

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Split
def random_link_split(cfg):
    """
    Split the graph data into train/val/test sets.
    """
    if "seed" in cfg:
        torch.manual_seed(cfg["seed"])

    train_val_test_split = T.RandomLinkSplit(
        num_val=cfg.get("val_size", 0.1),
        num_test=cfg.get("test_size", 0.1),
        disjoint_train_ratio=cfg.get("disjoint_train_ratio", 0.0),
        neg_sampling_ratio=1.0, # This is only for the validation/testing purpose
        add_negative_train_samples=False, # We will add negative samples with a custom sampler for the training data
        edge_types=("user", "interacts", "item"),
        rev_edge_types=("item", "rev_interacts", "user"),
    )

    return train_val_test_split


def validate_edge_indices(data):
    """
    Validate the edge indices in the graph data.
    """
    all_index = set()
    edge_index = data["user", "interacts", "item"].edge_index
    for u, v in zip(edge_index[0], edge_index[1]):
        all_index.add((u.item(), v.item()))
    
    all_label_index = set()
    edge_label_index = data["user", "interacts", "item"].edge_label_index
    for u, v in zip(edge_label_index[0], edge_label_index[1]):
        all_label_index.add((u.item(), v.item()))

    assert len(all_index) == len(edge_index[0])
    assert len(all_label_index) == len(edge_label_index[0])

    intersection = all_index.intersection(all_label_index)
    

    if len(intersection) > 0:
        logger.warning(f"Warning: Overlapping edges found between edge_index and edge_label_index")
        logger.info(f"Overlapping edges size: {len(intersection)}")



## Loss
def binary_recon_loss(preds, edge_label):
    """
    Compute the binary reconstruction loss.
    It encourages the model to assign high scores to positive edges and low scores to negative edges.
    Formula: -log(pos_preds) - log(1 - neg_preds)
    """
    EPS = 1e-4

    pos_mask = edge_label == 1
    pos_preds = preds[pos_mask]
    neg_preds = preds[~pos_mask]

    pos_loss = -torch.log(pos_preds + EPS).mean()
    neg_loss = -torch.log(1 - neg_preds + EPS).mean()
    return pos_loss + neg_loss


# def binary_recon_loss(preds, edge_label):
#     # logging.debug(f"Preds: {preds}")
#     # logging.debug(f"Edge Label: {edge_label}")

#     loss_fn = torch.nn.BCEWithLogitsLoss()
#     return loss_fn(preds, edge_label)

def kl_loss(mu, logvar):
    """
    Compute the Kullback-Leibler divergence loss.
    It measures the difference between the learned latent distribution and the prior distribution.
    Formula: D_KL(q(z|x) || p(z)) = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    """
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return kl_div



## Negative Sampling
def batch_random_sample(batch_data, negative_sampling_ratio):
    "Randomly sample negative examples from the batch dataset"
    num_neg_samples = int(len(batch_data['user', 'interacts', 'item'].edge_label_index[0]) * negative_sampling_ratio)
    #logger.info(f"Number of negative samples: {num_neg_samples}")
    edge_label_index = batch_data['user', 'interacts', 'item'].edge_label_index
    negative_samples = negative_sampling(edge_label_index, num_neg_samples=num_neg_samples)
    return negative_samples, torch.zeros(negative_samples.size(1), dtype=torch.float32)


# def graph_random_sample(data, num_neg_samples, negative_sampling_ratio):
#     "Randomly sample negative examples from the graph dataset"
#     if not num_neg_samples:
#         num_neg_samples = int(len(data['user', 'interacts', 'item'].edge_label_index[0]) * negative_sampling_ratio)
#     edge_label_index = data['user', 'interacts', 'item'].edge_index
#     negative_samples = negative_sampling(edge_label_index, num_neg_samples=num_neg_samples)
#     return negative_samples, torch.zeros(num_neg_samples, dtype=torch.float32)


## Metrics
def compute_auc(y_scores, y_true):
    y_true = y_true.long()
    y_scores = torch.sigmoid(y_scores)  # Apply sigmoid to logits
    score = AUROC(task="binary")(y_scores, y_true).item()
    return score

def compute_average_precision(y_scores, y_true):
    y_true = y_true.long()
    y_scores = torch.sigmoid(y_scores)  # Apply sigmoid to logits
    score = AveragePrecision(task="binary")(y_scores, y_true).item()
    return score


## Evaluation
@torch.no_grad()
def batch_evaluate(loader, model, sampling_strategy = "batch_random"):
    total_loss = 0
    total_loss_recon = 0
    total_loss_kl = 0
    all_preds = []
    all_labels = []

    model.eval()
    for batch in loader:
        batch = batch.to(device)
        z, mu, logvar = model(batch)
        positive_index = batch["user", "interacts", "item"].edge_label_index
        positive_labels = torch.ones(positive_index.size(1), dtype=torch.float32)
        
        if sampling_strategy == "batch_random":
            negative_index, negative_labels = batch_random_sample(batch)

        pos_preds = model.decode(z, positive_index)
        neg_preds = model.decode(z, negative_index)

        preds = torch.cat([pos_preds, neg_preds])
        edge_labels = torch.cat([positive_labels, negative_labels])
        all_labels.append(edge_labels)
        all_preds.append(preds)

        loss_recon = binary_recon_loss(preds, edge_labels)
        loss_kl = kl_loss(mu, logvar)

        loss = loss_recon + loss_kl

        total_loss += loss.item()
        total_loss_recon += loss_recon.item()
        total_loss_kl += loss_kl.item()

    y_scores = torch.cat(all_preds)
    y_true = torch.cat(all_labels)
    auc = compute_auc(y_scores, y_true)
    avg_precision = compute_average_precision(y_scores, y_true)
    return { "loss": total_loss / len(loader), 
             "loss_recon": total_loss_recon / len(loader), 
             "loss_kl": total_loss_kl / len(loader), 
             "auc": auc, 
             "average_precision": avg_precision }

@torch.no_grad()
def evaluate(data, model):
    model.eval()
    z, mu, logvar = model(data)

    edge_label = data["user", "interacts", "item"].edge_label
    edge_label_index = data["user", "interacts", "item"].edge_label_index
    pos_mask = edge_label == 1
    positive_index = edge_label_index[:,pos_mask]
    negative_index = edge_label_index[:,~pos_mask]

    pos_preds = model.decode(z, positive_index)
    neg_preds = model.decode(z, negative_index)

    preds = torch.cat([pos_preds, neg_preds])

    loss_recon = binary_recon_loss(preds, edge_label)
    loss_kl = kl_loss(mu, logvar)
    loss = loss_recon + loss_kl

    auc = compute_auc(preds, edge_label)
    avg_precision = compute_average_precision(preds, edge_label)
    return { "loss": loss.item(),
            "loss_recon": loss_recon.item(),
            "loss_kl": loss_kl.item(),
            "auc": auc,
            "average_precision": avg_precision }



