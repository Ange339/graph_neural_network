from abc import ABC, abstractmethod
from typing import List, Literal
import torch
import polars as pl
from torch_scatter import scatter_mean, scatter_sum

import torch.nn as nn

import logging

logger = logging.getLogger(__name__)


class FeatureExtractor(ABC):
    """Abstract base class for all feature extractors."""

    @abstractmethod
    def build(self, data, node_type: str) -> torch.Tensor:
        """Each extractor must implement this to return a feature tensor."""
        pass


class TopologicalFeatures(FeatureExtractor):
    def __init__(self, features: List[str], degree_log_transform: bool = False):
        self.features = features
        self.degree_log_transform = degree_log_transform

    def build(self, data, node_type: Literal["users", "books"]) -> torch.Tensor:
        df = data[node_type]  # "users" or "books"

        if "degree" in self.features and self.degree_log_transform:
            df = df.with_columns((pl.col("degree").log()).alias("degree"))

        feats = df.select(self.features).to_numpy()
        return torch.tensor(feats, dtype=torch.float32)


class TextualDescriptionFeatures(FeatureExtractor):
    def __init__(self):
        pass

    def build(self, data, node_type: Literal["users", "books"]) -> torch.Tensor:
        if node_type == "users":
            logger.warning(f"Users do not have descriptions")
            return None
        else:
            ids = data['books_des_id']
            embeddings = data['embeddings_descriptions']
            perm = torch.argsort(ids)
            embeddings = embeddings[perm]
            return embeddings

class TextualReviewFeatures(FeatureExtractor):
    def __init__(self, aggr_fn: Literal["mean", "sum"] = 'mean'):
        VALID_AGGR = {"mean", "sum"}
        if aggr_fn not in VALID_AGGR:
            raise ValueError(f"aggr_fn must be 'mean' or 'sum', got '{aggr_fn}'")
        self.aggr_fn = aggr_fn

    def build(self, data, node_type: Literal["users", "books"]) -> torch.Tensor:

        ids = data[f'{node_type}_review_id']  # 'users_review_id' or 'books_review_id'
        embeddings_reviews = data['embeddings_reviews']

        if self.aggr_fn == "mean":
            embeddings_reviews_aggr = scatter_mean(embeddings_reviews, ids, dim=0)
        elif self.aggr_fn == "sum":
            embeddings_reviews_aggr = scatter_sum(embeddings_reviews, ids, dim=0)

        return embeddings_reviews_aggr



