import os
import sys
import logging
import pathlib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import yaml
import pprint
import time
from collections import Counter, defaultdict
from tqdm import trange

import numpy as np
import pandas as pd
import polars as pl
import torch
from torch import optim

import matplotlib.pyplot as plt

from model import *
from src.data_loader import GraphLoader
from src.batch_loader import BatchLoader
from src.sampling import NegativeSampler
from src.loss import LossFunction
from src.registry import MODEL_REGISTRY
from src.utils import *

# Logging
DIR = pathlib.Path(__file__).parent.resolve()
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# Parser
parser = argparse.ArgumentParser(description="Load configuration file")
parser.add_argument("--config", type=str, default=f"{DIR}/config.yaml", help="Path to YAML config file")
args, unknown = parser.parse_known_args()

with open(args.config, "r") as f:
    cfg = yaml.safe_load(f)

logger.info("Configuration:")
logger.info(pprint.pformat(cfg))

################################################################################################
############################               DATA LOAD             ###############################
################################################################################################
OUT_DIR = DIR / cfg.get("out_dir", "out_dir")
os.makedirs(OUT_DIR, exist_ok=True)

# Prepare the data
graph_loader = GraphLoader(cfg)
graph_data, data = graph_loader.load()

# Split the data into train/val/test sets
train_val_test_split = random_link_split(cfg)
train_data, val_data, test_data = train_val_test_split(graph_data)

logger.debug(f"Train: {train_data}")
logger.debug(f"Val: {val_data}")
logger.debug(f"Test: {test_data}")

# batch loader
batch_loader = BatchLoader(cfg)
train_loader = batch_loader.load(train_data, shuffle=True)

logging.info(f"Number of training batches: {len(train_loader)}")

# val_loader = batch_loader.load(val_data, shuffle=False)
# test_loader = batch_loader.load(test_data, shuffle=False)


################################################################################################
############################                MODEL                ###############################
################################################################################################


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

val_data = val_data.to(device)
test_data = test_data.to(device)

encoder = MODEL_REGISTRY.get(cfg['encoder'], SageEncoder)
decoder = MODEL_REGISTRY.get(cfg['decoder'], InnerProductDecoder)

model = GAE(
    data=train_data,
    encoder=encoder,
    decoder=decoder,
    cfg=cfg
).to(device)


logging.info("Model summary:")
logging.info(pprint.pformat(model))


################################################################################################
############################                TRAIN                ###############################
################################################################################################

# History
history = {
    "train" : defaultdict(list),
    "val" : defaultdict(list),
}

best_model_cfg = {
    "best_model": None,
    "loss": float('inf'),
    "epoch": 0,
    "auc": 0,
    "average_precision": 0
}

## Optimizer
optimizer = optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg['opt_decay_step'], gamma=cfg['opt_decay_rate'])

## Sampling settings
negative_sampler = NegativeSampler(cfg)
val_data = negative_sampler.eval_sample(val_data)
test_data = negative_sampler.eval_sample(test_data)

## Loss functions
loss_function = LossFunction(cfg)
variational = cfg.get("variational", True)
kl_beta = cfg.get("kl_beta", 1.0)
kl_warmup_epoch = cfg.get("kl_warmup_epoch", 0)


########### TRAINING LOOP ###########
for epoch in trange(cfg['epochs'], desc="Training", unit="Epochs"):
    total_loss = 0
    total_loss_recon = 0
    total_loss_kl = 0
    
    all_preds = []
    all_labels = []

    ## Training
    model.train()
    num_positive = 0
    num_negatives = 0

    ## Beta warmup
    if kl_warmup_epoch > 0 and epoch < kl_warmup_epoch:
        kl_beta = (epoch / kl_warmup_epoch) * cfg.get("kl_beta", 1.0)

    for batch in train_loader:
        # Batch encoding
        batch = batch.to(device)
        z_dict, mu_dict, logvar_dict = model(batch)

        # Edge sampling
        ## Positive
        positive_index = batch["user", "interacts", "item"].edge_label_index
        positive_labels = torch.ones(positive_index.size(1), dtype=torch.float32, device=device)
        num_positive += positive_index.size(1)

        ## Negative sampling
        negative_index, negative_labels = negative_sampler.sample(batch)
        num_negatives += negative_index.size(1)

        # Model decoding
        pos_preds = model.decode(z_dict['user'], z_dict['item'], positive_index)
        neg_preds = model.decode(z_dict['user'], z_dict['item'], negative_index)
        preds = torch.cat([pos_preds, neg_preds])
        edge_labels = torch.cat([positive_labels, negative_labels])

        ## Compute the loss
        loss_recon = loss_function.reconstruction_loss(preds, edge_labels)

        if variational:
            mu, logvar = torch.cat([mu_dict['user'], mu_dict['item']]), torch.cat([logvar_dict['user'], logvar_dict['item']])
            loss_kl = loss_function.kl_loss(mu, logvar, beta=kl_beta)
        else:
            loss_kl = torch.tensor(0.0)

        loss = loss_recon + loss_kl

        # Loss backward
        optimizer.zero_grad() # Zero gradients
        loss.backward()
        optimizer.step()

        ## Store the information
        all_preds.append(preds)
        all_labels.append(edge_labels)
        total_loss += loss.item()
        total_loss_recon += loss_recon.item()
        total_loss_kl += loss_kl.item()

    scheduler.step() # Update learning rate

    # Evaluation
    if epoch % cfg['eval_interval'] == 0:
        ## Train data
        total_loss = total_loss / len(train_loader)
        total_loss_recon = total_loss_recon / len(train_loader)
        total_loss_kl = total_loss_kl / len(train_loader)

        y_scores, y_true = torch.cat(all_preds), torch.cat(all_labels)
        auc_score = compute_auc(y_scores, y_true)
        avg_precision = compute_average_precision(y_scores, y_true)
        precision_at_k = compute_precision_at_k(y_scores, y_true, k=cfg.get("retrieval_k", 10))
        recall_at_k = compute_recall_at_k(y_scores, y_true, k=cfg.get("retrieval_k", 10))
        history["train"]["loss"].append(total_loss)
        history["train"]["loss_recon"].append(total_loss_recon)
        history["train"]["loss_kl"].append(total_loss_kl)
        history["train"]["auc"].append(auc_score)
        history["train"]["average_precision"].append(avg_precision)
        history["train"]["precision_at_k"].append(precision_at_k)
        history["train"]["recall_at_k"].append(recall_at_k)

        # Validation data
        val_metrics = evaluate(val_data, model, loss_function, kl_beta=kl_beta, cfg=cfg)
        history["val"]["loss"].append(val_metrics["loss"])
        history["val"]["loss_recon"].append(val_metrics["loss_recon"])
        history["val"]["loss_kl"].append(val_metrics["loss_kl"])
        history["val"]["auc"].append(val_metrics["auc"])
        history["val"]["average_precision"].append(val_metrics["average_precision"])
        history["val"]["precision_at_k"].append(val_metrics["precision_at_k"])
        history["val"]["recall_at_k"].append(val_metrics["recall_at_k"])

        logger.info(f"Epoch: {epoch}")
        logger.info(f"Train. Total Loss: {total_loss:.2f}, Rec.: {total_loss_recon:.2f}, KL: {total_loss_kl:.2f}, AUC: {auc_score:.2%}, AP: {avg_precision:.2%}, P@{cfg.get('retrieval_k',10)}: {precision_at_k:.2%}, R@{cfg.get('retrieval_k',10)}: {recall_at_k:.2%}")
        logger.info(f"Val. Total Loss: {val_metrics['loss']:.2f}, Rec.: {val_metrics['loss_recon']:.2f}, KL: {val_metrics['loss_kl']:.2f}, AUC: {val_metrics['auc']:.2%}, AP: {val_metrics['average_precision']:.2%}, P@{cfg.get('retrieval_k',10)}: {val_metrics['precision_at_k']:.2%}, R@{cfg.get('retrieval_k',10)}: {val_metrics['recall_at_k']:.2%}")

        logger.debug(f"Num positive: {num_positive}; Num negatives: {num_negatives}; Negative Ratio: {num_negatives / (num_positive) + 1e-6:.2f}")

    else:
        logger.info(f"Epoch: {epoch}, Loss: {total_loss/len(train_loader):.2f}")

    if val_metrics["auc"] > best_model_cfg["auc"]:
        best_model_cfg["best_model"] = model.state_dict()
        best_model_cfg["loss"] = val_metrics["loss"]
        best_model_cfg["epoch"] = epoch
        best_model_cfg["auc"] = val_metrics["auc"]
        best_model_cfg["average_precision"] = val_metrics["average_precision"]
        best_model_cfg["precision_at_k"] = val_metrics["precision_at_k"]
        best_model_cfg["recall_at_k"] = val_metrics["recall_at_k"]

# Save the best model
if cfg["save_model"]:
    torch.save(best_model_cfg["best_model"], OUT_DIR / "best_model.pth")


## Evaluation
# Loss Curves
fig, axs = plt.subplots(3, 3, figsize=(18, 12))

# Loss
axs[0][0].plot(history["train"]["loss"], label="Train")
axs[0][0].plot(history["val"]["loss"], label="Val")
axs[0][0].set_title("Loss")
axs[0][0].set_xlabel("Epoch")
axs[0][0].set_ylabel("Loss")

# Rec Loss
axs[0][1].plot(history["train"]["loss_recon"])
axs[0][1].plot(history["val"]["loss_recon"])
axs[0][1].set_title("Reconstruction Loss")
axs[0][1].set_xlabel("Epoch")
axs[0][1].set_ylabel("Loss")

# KL Divergence Loss
axs[0][2].plot(history["train"]["loss_kl"])
axs[0][2].plot(history["val"]["loss_kl"])
axs[0][2].set_title("KL Loss")
axs[0][2].set_xlabel("Epoch")
axs[0][2].set_ylabel("Loss")

# AUC
axs[1][0].plot(history["train"]["auc"])
axs[1][0].plot(history["val"]["auc"])
axs[1][0].set_title("AUC")
axs[1][0].set_xlabel("Epoch")
axs[1][0].set_ylabel("AUC")

# Average Precision
axs[1][1].plot(history["train"]["average_precision"])
axs[1][1].plot(history["val"]["average_precision"])
axs[1][1].set_title("Average Precision")
axs[1][1].set_xlabel("Epoch")
axs[1][1].set_ylabel("Average Precision")


# Precision@K
axs[1][2].plot(history["train"]["precision_at_k"])
axs[1][2].plot(history["val"]["precision_at_k"])
axs[1][2].set_title(f"Precision@{cfg.get('retrieval_k',10)}")
axs[1][2].set_xlabel("Epoch")
axs[1][2].set_ylabel(f"Precision@{cfg.get('retrieval_k',10)}")

# Recall@K
axs[2][0].plot(history["train"]["recall_at_k"])
axs[2][0].plot(history["val"]["recall_at_k"])
axs[2][0].set_title(f"Recall@{cfg.get('retrieval_k',10)}")
axs[2][0].set_xlabel("Epoch")
axs[2][0].set_ylabel(f"Recall@{cfg.get('retrieval_k',10)}")

# Set a single legend for the whole figure
lines_labels = [axs[0][0].get_lines()[0], axs[0][0].get_lines()[1]]
labels = ["Train", "Val"]
fig.legend(lines_labels, labels, loc="upper right", fontsize="large")
fig.suptitle("Training and Validation Metrics", fontsize=18)
plt.tight_layout()
plt.savefig(OUT_DIR / "training_curves.png")

logger.info(f"Best model found at epoch {best_model_cfg['epoch']}, Val Loss: {best_model_cfg['loss']:.2f}, AUC: {best_model_cfg['auc']:.2%}, AP: {best_model_cfg['average_precision']:.2%}, P@{cfg.get('retrieval_k',10)}: {best_model_cfg['precision_at_k']:.2%}, R@{cfg.get('retrieval_k',10)}: {best_model_cfg['recall_at_k']:.2%}")

if cfg["tsne_visualization"]:
    # Load the best model for visualization
    model.load_state_dict(best_model_cfg["best_model"])
    user_embeddings, item_embeddings = get_embeddings(model)
    tsne_transform(user_embeddings, title="t-SNE User Embeddings", filename=OUT_DIR / "tsne_user_embeddings.png")
    tsne_transform(item_embeddings, title="t-SNE Item Embeddings", filename=OUT_DIR / "tsne_item_embeddings.png")

logger.info(f"Done")