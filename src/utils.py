import numpy as np
import polars as pl
import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
from torchmetrics.classification import AveragePrecision, AUROC
from torch_geometric.utils import negative_sampling

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

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
        neg_sampling_ratio=cfg['negative_sampling_ratio'], # This is only for the validation/testing purpose
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
def evaluate(data, model, loss_functions, kl_beta, cfg):
    model.eval()
    z_dict, mu_dict, logvar_dict = model(data)

    # Edge extraction
    edge_label = data["user", "interacts", "item"].edge_label
    edge_label_index = data["user", "interacts", "item"].edge_label_index
    pos_mask = edge_label == 1
    positive_index = edge_label_index[:,pos_mask]
    negative_index = edge_label_index[:,~pos_mask]

    # Decode
    pos_preds = model.decode(z_dict['user'], z_dict['item'], positive_index)
    neg_preds = model.decode(z_dict['user'], z_dict['item'], negative_index)
    preds = torch.cat([pos_preds, neg_preds])

    # Scores
    loss_recon = loss_functions.reconstruction_loss(preds, edge_label).item()

    if cfg.get("variational", False):
        mu, logvar = torch.cat([mu_dict['user'], mu_dict['item']]), torch.cat([logvar_dict['user'], logvar_dict['item']])
        loss_kl = loss_functions.kl_loss(mu, logvar, beta=kl_beta).item()
    else:
        loss_kl = 0

    loss = loss_recon + loss_kl

    auc = compute_auc(preds, edge_label)
    avg_precision = compute_average_precision(preds, edge_label)

    result = { "loss": loss,
                "loss_recon": loss_recon,
                "loss_kl": loss_kl,
                "auc": auc,
                "average_precision": avg_precision }

    return result



# @torch.no_grad()
# def batch_evaluate(loader, model, sampling_strategy = "batch_random"):
#     total_loss = 0
#     total_loss_recon = 0
#     total_loss_kl = 0
#     all_preds = []
#     all_labels = []

#     model.eval()
#     for batch in loader:
#         batch = batch.to(device)
#         z, mu, logvar = model(batch)
#         positive_index = batch["user", "interacts", "item"].edge_label_index
#         positive_labels = torch.ones(positive_index.size(1), dtype=torch.float32)
        
#         if sampling_strategy == "batch_random":
#             negative_index, negative_labels = batch_random_sample(batch)

#         pos_preds = model.decode(z, positive_index)
#         neg_preds = model.decode(z, negative_index)

#         preds = torch.cat([pos_preds, neg_preds])
#         edge_labels = torch.cat([positive_labels, negative_labels])
#         all_labels.append(edge_labels)
#         all_preds.append(preds)

#         loss_recon = binary_recon_loss(preds, edge_labels)
#         loss_kl = kl_loss(mu, logvar)

#         loss = loss_recon + loss_kl

#         total_loss += loss.item()
#         total_loss_recon += loss_recon.item()
#         total_loss_kl += loss_kl.item()

#     y_scores = torch.cat(all_preds)
#     y_true = torch.cat(all_labels)
#     auc = compute_auc(y_scores, y_true)
#     avg_precision = compute_average_precision(y_scores, y_true)
#     return { "loss": total_loss / len(loader), 
#              "loss_recon": total_loss_recon / len(loader), 
#              "loss_kl": total_loss_kl / len(loader), 
#              "auc": auc, 
#              "average_precision": avg_precision }


## Visualization
def get_embeddings(model):
    user_embeddings = model['user_embedding.weight'].detach().cpu().numpy()
    item_embeddings = model['item_embedding.weight'].detach().cpu().numpy()
    return user_embeddings, item_embeddings

def tsne_visualization(embeddings, labels, title="t-SNE Visualization", filename=None):
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=embeddings_2d[:,0], y=embeddings_2d[:,1], hue=labels, palette="viridis", legend="full", alpha=0.7)
    plt.title(title)
    plt.savefig(filename)