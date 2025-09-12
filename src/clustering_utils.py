import polars as pl
import pandas as pd
import os

import torch
import numpy as np
from sklearn import cluster
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.manifold import TSNE

import seaborn as sns
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




def load(cfg):
    DIR = cfg['dir']
    descriptions = pl.read_parquet(os.path.join(DIR, cfg['descriptions_filename']), columns=["book_id", "book_name", "description", "filtered"])
    reviews = pl.read_parquet(os.path.join(DIR, cfg['reviews_filename']), columns=["user_id", "book_id", "book_review", "filtered"])
    id_map = pl.read_csv(os.path.join(DIR, "id_map.csv"))
    user_id_map = id_map.filter(pl.col("type") == "user").drop('type')
    item_id_map = id_map.filter(pl.col("type") == "item").drop('type')
    model = torch.load(cfg.get('model_path', 'best_model.pth'), map_location=device)
    user_z = model['user_embedding.weight'].detach().numpy()
    item_z = model['item_embedding.weight'].detach().numpy()
    data = {
        "descriptions": descriptions,
        "reviews": reviews,
        "user_z": user_z,
        "item_z": item_z
    }
    return data, user_id_map, item_id_map





def get_reviews_text(reviews: pl.DataFrame, id_map: pl.DataFrame, node_type: str):
    """
    type: 'user' or 'item'
    """
    reviews = reviews.join(id_map, left_on=f"{node_type}_id", right_on="id", how="inner")
    reviews = reviews.sort("new_id").group_by("user_id").head(5) #Random 5 reviews per user

    reviews = reviews.group_by("new_id").agg(
        pl.col("book_review").str.join("\n\n").alias("all_reviews")
    )
    reviews = reviews.sort("new_id")
    reviews = reviews['all_reviews'].to_numpy()
    return reviews


def get_descriptions_text(descriptions: pl.DataFrame, id_map: pl.DataFrame, node_type: str):
    """
    type: 'user' or 'item'
    """
    if node_type == 'user':
        raise ValueError("No descriptions for users")
    descriptions = descriptions.join(id_map, left_on=f"{node_type}_id", right_on="id", how="inner")

    descriptions = descriptions.fill_null("")
    descriptions = descriptions.with_columns(
        (pl.col('book_name') + ".\n\n" + pl.col('description')).alias('description')
    )

    descriptions = descriptions.sort("new_id")
    descriptions = descriptions['description'].to_numpy()
    return descriptions


def get_all_text(reviews: pl.DataFrame, descriptions: pl.DataFrame, id_map: pl.DataFrame, node_type: str):
    if node_type == 'user':
        all_text = get_reviews_text(reviews, id_map, node_type)
    elif node_type == 'item':
        reviews_text = get_reviews_text(reviews, id_map, node_type)
        descriptions_text = get_descriptions_text(descriptions, id_map, node_type)
        all_text = np.array([f"{desc}\n\n{rev}" for desc, rev in zip(descriptions_text, reviews_text)])
    else:
        raise ValueError("node_type must be 'user' or 'item'")
    return all_text


def compute_silhouette_score(embeddings, y_preds, y_true):
    score = silhouette_score(embeddings, y_preds)
    return score


def compute_nmi(embeddings, y_preds, y_true):
    nmi = normalized_mutual_info_score(y_true, y_preds)
    return nmi

def compute_ari(embeddings, y_preds, y_true):
    ari = adjusted_rand_score(y_true, y_preds)
    return ari


def compute_purity(embeddings, y_preds, y_true):
    "https://stackoverflow.com/questions/34047540/python-clustering-purity-metric"
    contingency_matrix = cluster.contingency_matrix(y_true, y_preds)
    purity = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
    return purity


def cluster_evaluation(embeddings, y_preds, y_true):
    nmi = compute_nmi(embeddings, y_preds, y_true)
    ari = compute_ari(embeddings, y_preds, y_true)
    purity = compute_purity(embeddings, y_preds, y_true)
    silhouette = compute_silhouette_score(embeddings, y_preds, y_true)

    return {
        "nmi": nmi,
        "ari": ari,
        "purity": purity,
        "silhouette": silhouette
    }


def tsne_visualization(embeddings, labels, title="t-SNE Visualization", filename=None):
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=embeddings_2d[:,0], y=embeddings_2d[:,1], hue=labels, palette="viridis", legend="full", alpha=0.7)
    plt.title(title)
    plt.savefig(filename)