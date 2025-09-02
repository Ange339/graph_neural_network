import os
import sys
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
import matplotlib.pyplot as plt

from model import VGAE, VGAEncoder, InnerProductDecoder

from src.data_loader import GraphLoader
from src.batch_loader import BatchLoader
from src.registry import TRAIN_REGISTRY
from src.utils import *

import pathlib
DIR = pathlib.Path(__file__).parent.resolve()
PLOT_DIR = DIR / "plots"

os.makedirs(PLOT_DIR, exist_ok=True)

import logging

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    filename='train.log',
    filemode='w',
    level=logging.DEBUG,
    datefmt='%m-%d %H:%M:%S',
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logging.info("Running Urban Planning")
logger = logging.getLogger(__name__)

# ch = logging.StreamHandler()
# ch.setLevel(logging.INFO) # or any other level
# logger.addHandler(ch)

# Parser
parser = argparse.ArgumentParser(description="Load parquet datasets")
parser.add_argument("--config", type=str, default=f"{DIR}/config.yaml", help="Path to YAML config file")
args, unknown = parser.parse_known_args()

with open(args.config, "r") as f:
    cfg = yaml.safe_load(f)

logger.info("Configuration:")
logger.info(pprint.pformat(cfg))

# Prepare the data
graph_loader = GraphLoader(cfg)
graph_data = graph_loader.load()

# Split the data into train/val/test sets
train_val_test_split = random_link_split(cfg)
train_data, val_data, test_data = train_val_test_split(graph_data)

# batch loader
batch_loader = BatchLoader(cfg)
train_loader = batch_loader.load(train_data, shuffle=True)

# val_loader = batch_loader.load(val_data, shuffle=False)
# test_loader = batch_loader.load(test_data, shuffle=False)


################################################################################################
############################                MODEL                ###############################
################################################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

val_data = val_data.to(device)
test_data = test_data.to(device)

model = VGAE(
    data=train_data,
    encoder=TRAIN_REGISTRY.get(cfg['encoder'], VGAEncoder),
    decoder=TRAIN_REGISTRY.get(cfg['decoder'], InnerProductDecoder),
    cfg=cfg
).to(device)

model.summary()
optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])

################################################################################################
############################                TRAIN                ###############################
################################################################################################

history = {
    "train" : defaultdict(list),
    "val" : defaultdict(list),
}

best_model_cfg = {
    "best_model": None,
    "best_loss": float('inf'),
    "epoch": 0,
    "auc": 0,
    "average_precision": 0
}

sampling_strategy = cfg.get('sampling_strategy', 'batch_random')

for epoch in trange(cfg['epochs'], desc="Training", unit="Epochs"):
    total_loss = 0
    total_loss_recon = 0
    total_loss_kl = 0
    
    all_preds = []
    all_labels = []

    ## Training
    model.train()
    for batch in train_loader:
        optimizer.zero_grad() # Zero gradients
        batch = batch.to(device)

        z, mu, logvar = model(batch)

        positive_index = batch["user", "interacts", "item"].edge_label_index
        positive_labels = torch.ones(positive_index.size(1), dtype=torch.float32).to(device)

        if sampling_strategy == "batch_random":
            negative_index, negative_labels = batch_random_sample(batch, cfg['negative_sampling_ratio'])
            negative_index = negative_index.to(device)
            negative_labels = negative_labels.to(device)

        pos_preds = model.decode(z, positive_index)
        neg_preds = model.decode(z, negative_index)

        preds = torch.cat([pos_preds, neg_preds])
        edge_labels = torch.cat([positive_labels, negative_labels])

        ## Store the prediction
        all_preds.append(preds)
        all_labels.append(edge_labels)
        
        ## Compute the loss
        loss_recon = binary_recon_loss(preds, edge_labels)
        loss_kl = kl_loss(mu, logvar)
        loss = loss_recon + loss_kl

        # Store the loss
        total_loss += loss.item()
        total_loss_kl += loss_kl.item()
        
        # Loss backward
        loss.backward()
        optimizer.step()

    logger.info(f"Epoch: {epoch}, Loss: {total_loss/len(train_loader):.2f}")

    # Evaluation
    if epoch % cfg['eval_interval'] == 0:
        ## Train data
        y_scores, y_true = torch.cat(all_preds), torch.cat(all_labels)
        auc_score = compute_auc(y_scores, y_true)
        avg_precision = compute_average_precision(y_scores, y_true)
        history["train"]["loss"].append(total_loss / len(train_loader))
        history["train"]["loss_recon"].append(total_loss_recon / len(train_loader))
        history["train"]["loss_kl"].append(total_loss_kl / len(train_loader))
        history["train"]["auc"].append(auc_score)
        history["train"]["average_precision"].append(avg_precision)

        ## Validation data
        val_metrics = evaluate(val_data, model)
        history["val"]["loss"].append(val_metrics["loss"])
        history["val"]["loss_recon"].append(val_metrics["loss_recon"])
        history["val"]["loss_kl"].append(val_metrics["loss_kl"])
        history["val"]["auc"].append(val_metrics["auc"])
        history["val"]["average_precision"].append(val_metrics["average_precision"])

        logger.info(f"\nTrain. Total Loss: {total_loss:.2f}, Rec.: {total_loss_recon:.2f}, KL: {total_loss_kl:.2f}, AUC: {auc_score:.2%}, AP: {avg_precision:.2%}")
        logger.info(f"Val. Total Loss: {val_metrics['loss']:.2f}, Rec.: {val_metrics['loss_recon']:.2f}, KL: {val_metrics['loss_kl']:.2f}, AUC: {val_metrics['auc']:.2%}, AP: {val_metrics['average_precision']:.2%}")

    if val_metrics["loss"] < best_model_cfg["best_loss"]:
        best_model_cfg["best_loss"] = val_metrics["loss"]
        best_model_cfg["best_model"] = model.state_dict()
        best_model_cfg["epoch"] = epoch
        best_model_cfg["auc"] = val_metrics["auc"]
        best_model_cfg["average_precision"] = val_metrics["average_precision"]

if cfg["save_model"]:
    torch.save(best_model_cfg["best_model"], "best_model.pth")



## Evaluation

# Loss Curves
fig, axs = plt.subplots(2, 3, figsize=(18, 8))

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

# Set a single legend for the whole figure
lines_labels = [axs[0][0].get_lines()[0], axs[0][0].get_lines()[1]]
labels = ["Train", "Val"]
fig.legend(lines_labels, labels, loc="upper right", fontsize="large")
fig.suptitle("Training and Validation Metrics", fontsize=18)
plt.tight_layout()
plt.savefig(PLOT_DIR / "training_curves.png")


