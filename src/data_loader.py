import os
from typing import Literal
import numpy as np
import polars as pl
import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

from src.feature_extractor import *
from .registry import EMB_FEATURE_REGISTRY

import logging


logger = logging.getLogger(__name__)


class GraphLoader:
    def __init__(self, cfg):
        self.cfg = cfg

    def load(self):
        dataset = Dataset(self.cfg)
        text_embedding_processor = TextEmbeddingProcessor(self.cfg)
        graph_builder = GraphBuilder(self.cfg)

        data = dataset.load()
        data = dataset.preprocess(data)
        data = text_embedding_processor.to_torch(data)
        graph_data = graph_builder.build(data)
        return graph_data, data


class Dataset:
    def __init__(self, cfg):
        self.cfg = cfg

    def load(self):
        "Load the files by reading the filenames in config"
        DIR = self.cfg['dir']
        users = pl.read_parquet(os.path.join(DIR, self.cfg['users_filename']))
        books = pl.read_parquet(os.path.join(DIR, self.cfg['books_filename']))
        interactions = pl.read_parquet(os.path.join(DIR, self.cfg['interactions_filename']))
        descriptions = pl.read_parquet(os.path.join(DIR, self.cfg['descriptions_filename']), columns=["book_id", "filtered"])
        reviews = pl.read_parquet(os.path.join(DIR, self.cfg['reviews_filename']), columns=["user_id", "book_id", "filtered"])
        embeddings_descriptions = pl.read_parquet(os.path.join(DIR, self.cfg['embeddings_descriptions_filename']))
        embeddings_reviews = pl.read_parquet(os.path.join(DIR, self.cfg['embeddings_reviews_filename']))
        data =  {
            "users": users,
            "books": books,
            "interactions": interactions,
            "descriptions": descriptions,
            "reviews": reviews,
            "embeddings_descriptions": embeddings_descriptions,
            "embeddings_reviews": embeddings_reviews
        }
        return data

    def preprocess(self, data):
        "Preprocess the loaded data by eliminating observations based on the method"

        users = data["users"]
        books = data["books"]
        interactions = data["interactions"]
        descriptions = data["descriptions"]
        reviews = data["reviews"]
        embeddings_descriptions = data["embeddings_descriptions"]
        embeddings_reviews = data["embeddings_reviews"]


        # Drop hashed ID columns since we're going to redefine it
        users = users.drop('user_id_hashed', strict=False)
        books = books.drop('book_id_hashed', strict=False)
        interactions = interactions.drop(['user_id_hashed', 'book_id_hashed'], strict=False)

        edge_type = self.cfg.get("edge_type", "interactions")
        logger.info(f"Edge type: {edge_type}")
        if edge_type == "reads":
            interactions = interactions.filter(pl.col('is_read') == True)

        logger.info(f"Initial size. Users: {len(users)}, Books: {len(books)}, Total: {len(users) + len(books)}, Edges: {len(interactions)}, Density: {len(interactions) / (len(users) * len(books)):4%}")

        # Filter by degree.
        review_coreness_k = self.cfg.get("review_coreness_k", 3)
        coreness_k = self.cfg.get("coreness_k", 3)

        users = users.filter(pl.col('review_coreness') >= review_coreness_k, pl.col('coreness') >= coreness_k)
        users = users.join(interactions.select(["user_id"]).unique(), on="user_id", how="inner") # Keep only users with interactions

        books = books.filter(pl.col('review_coreness') >= review_coreness_k, pl.col('coreness') >= coreness_k)
        books = books.join(interactions.select(["book_id"]).unique(), on="book_id", how="inner") # Keep only books with interactions

        ## Reindexing and save the mapping
        users = users.with_columns((pl.col('user_id').rank()-1).cast(pl.Int64).alias('user_id_hashed'))
        books = books.with_columns((pl.col('book_id').rank()-1).cast(pl.Int64).alias('book_id_hashed'))

        user_id_map = users.select(['user_id', 'user_id_hashed']).rename({'user_id':'id', 'user_id_hashed':'new_id'}).with_columns(pl.lit('user').alias('type'))
        book_id_map = books.select(['book_id', 'book_id_hashed']).rename({'book_id':'id', 'book_id_hashed':'new_id'}).with_columns(pl.lit('book').alias('type'))
        id_map = pl.concat([user_id_map, book_id_map], how='vertical_relaxed')
        id_map.write_csv(os.path.join(self.cfg['dir'], self.cfg.get('id_map_filename', 'id_map.csv')))

        ## Filter and add the new_index information for the rest of data
        interactions = interactions.join(users.select(["user_id", "user_id_hashed"]), on="user_id", how="inner")
        interactions = interactions.join(books.select(["book_id", "book_id_hashed"]), on="book_id", how="inner") # Drop rows with reviews that are not english

        logger.info(f"Filtered size. Users: {len(users)}, Books: {len(books)}, Total: {len(users) + len(books)}, Edges: {len(interactions)}, Density: {len(interactions) / (len(users) * len(books)):4%}")

        user_degree = interactions.group_by('user_id').agg(pl.len().alias('degree')).sort('degree')
        book_degree = interactions.group_by('book_id').agg(pl.len().alias('degree')).sort('degree')
        logger.info(f"User degree. Min: {user_degree[0, 'degree']}, Max: {user_degree[-1, 'degree']}, Median: {user_degree[user_degree.height // 2, 'degree']}, Mean: {user_degree['degree'].mean():.2f}")
        logger.info(f"Top 5 users by degree:\n{user_degree.sort('degree', descending=True).head(5)}")
        logger.info(f"Book degree. Min: {book_degree[0, 'degree']}, Max: {book_degree[-1, 'degree']}, Median: {book_degree[book_degree.height // 2, 'degree']}, Mean: {book_degree['degree'].mean():.2f}")
        logger.info(f"Top 5 books by degree:\n{book_degree.sort('degree', descending=True).head(5)}")
        embeddings_descriptions = embeddings_descriptions.join(books.select(["book_id", "book_id_hashed"]), on="book_id", how="inner")
        embeddings_reviews = embeddings_reviews.join(interactions.select(["book_id", "user_id", "user_id_hashed", "book_id_hashed"]), on=["book_id", "user_id"], how="inner")

        ## Filter also the embeddings
        logger.info(f"Initial textual embeddings size. Descriptions: {len(embeddings_descriptions)}, Reviews: {len(embeddings_reviews)}, Total: {len(embeddings_descriptions) + len(embeddings_reviews)}")
        descriptions = data['descriptions'].filter(pl.col("filtered") == 1).drop("filtered")
        embeddings_descriptions = embeddings_descriptions.join(descriptions[["book_id"]], on="book_id", how="inner")

        reviews = data['reviews'].filter(pl.col("filtered") == 1).drop("filtered")
        embeddings_reviews = embeddings_reviews.join(reviews[["user_id", "book_id"]], on=["user_id", "book_id"], how="inner")
        logger.info(f"Filtered textual embeddings size. Descriptions: {len(embeddings_descriptions)}, Reviews: {len(embeddings_reviews)}, Total: {len(embeddings_descriptions) + len(embeddings_reviews)}")

        ## Sorting
        users = users.sort('user_id_hashed')
        books = books.sort('book_id_hashed')
        interactions = interactions.sort('user_id_hashed', 'book_id_hashed')
        embeddings_descriptions = embeddings_descriptions.sort('book_id_hashed')
        embeddings_reviews = embeddings_reviews.sort('user_id_hashed', 'book_id_hashed')

        data = {
            "users": users,
            "books": books,
            "interactions": interactions,
            "embeddings_descriptions": embeddings_descriptions,
            "embeddings_reviews": embeddings_reviews
        }

        return data



class TextEmbeddingProcessor:
    def __init__(self, cfg):
        self.cfg = cfg # It won't be used

    def to_torch(self, data):
        if  isinstance(data['embeddings_reviews'], pl.DataFrame):
            return self.embeddings_parquet2torch(data)
        else:
            pass

    def embeddings_parquet2torch(self, data):
        """
        Convert embeddings from parquet to torch tensors.
        Return also the index list for each embeddings 
        """

        # Reviews
        reviews = data['embeddings_reviews']

        ## Extract the embeddings
        emb_rev_columns = [col for col in reviews.columns if col.startswith('column')]
        embeddings_reviews = reviews.select(emb_rev_columns)
        embeddings_reviews = embeddings_reviews.to_numpy()
        embeddings_reviews = torch.tensor(embeddings_reviews, dtype=torch.float32)

        ## Extract the ids
        users_review_id = reviews['user_id_hashed'].to_numpy()
        users_review_id = torch.tensor(users_review_id, dtype=torch.long)
        books_review_id = reviews['book_id_hashed'].to_numpy()
        books_review_id = torch.tensor(books_review_id, dtype=torch.long)

        # Descriptions
        descriptions = data['embeddings_descriptions']

        ## Extract the embeddings
        emb_des_columns = [col for col in descriptions.columns if col.startswith('column')]
        embeddings_descriptions = descriptions.select(emb_des_columns)
        embeddings_descriptions = embeddings_descriptions.to_numpy()
        embeddings_descriptions = torch.tensor(embeddings_descriptions, dtype=torch.float32)

        books_des_id = descriptions['book_id_hashed'].to_numpy()
        books_des_id = torch.tensor(books_des_id, dtype=torch.long)

        assert embeddings_reviews.shape[0] == reviews.shape[0] and embeddings_reviews.shape[1] == len(emb_rev_columns), "Mismatch in embeddings_reviews shape"
        assert embeddings_descriptions.shape[0] == descriptions.shape[0] and embeddings_descriptions.shape[1] == len(emb_des_columns), "Mismatch in embeddings_descriptions shape"

        # Overwrite to free the memory
        data['embeddings_reviews'] = embeddings_reviews
        data['embeddings_descriptions'] = embeddings_descriptions
        data['users_review_id'] = users_review_id
        data['books_review_id'] = books_review_id
        data['books_des_id'] = books_des_id

        return data



class GraphBuilder():
    def __init__(self, cfg):
        self.cfg = cfg

    def build_edges(self, data):
        "Build graph edges from the data."
        edges = data['interactions'].select(['user_id_hashed', 'book_id_hashed']).rows()
        src, dst = [], []
        for e in edges:
            s, t = e
            src.append(s)
            dst.append(t)
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        return edge_index

    def build_nodes_features(self, data):
        """Return user_x, book_x feature tensors."""
        user_feats, book_feats = [], []

        for fconf in self.cfg.get("user_features", []):
            extractor_cls = EMB_FEATURE_REGISTRY[fconf["name"]]
            extractor = extractor_cls(**{k: v for k, v in fconf.items() if k != "name"})
            feat = extractor.build(data, "users")
            logger.info(f"User Feature {fconf['name']} shape: {feat.shape if feat is not None else None}")
            if feat is not None:
                user_feats.append(feat)

        for fconf in self.cfg.get("book_features", []):
            extractor_cls = EMB_FEATURE_REGISTRY[fconf["name"]]
            extractor = extractor_cls(**{k: v for k, v in fconf.items() if k != "name"})
            feat = extractor.build(data, "books")
            logger.info(f"Item Feature {fconf['name']} shape: {feat.shape if feat is not None else None}")
            if feat is not None:
                book_feats.append(feat)

        user_x = torch.cat(user_feats, dim=1) if user_feats else None
        book_x = torch.cat(book_feats, dim=1) if book_feats else None

        return user_x, book_x

    def build(self, data):
        edge_index = self.build_edges(data)
        user_x, book_x = self.build_nodes_features(data)
        
        graph_data = HeteroData()
        graph_data['user'].node_id = torch.arange(len(data['users']))
        graph_data['item'].node_id = torch.arange(len(data['books'])) 

        # Encode the genre information. They will not be used for training
        graph_data['user'].genre = torch.tensor(data['users']['genre'].to_list())
        graph_data['item'].genre = torch.tensor(data['books']['genre'].to_list())

        if user_x is not None:
            graph_data['user'].x = user_x
        if book_x is not None:
            graph_data['item'].x = book_x # From now we will call it item
        graph_data[('user', 'interacts', 'item')].edge_index = edge_index
        graph_data = T.ToUndirected()(graph_data) # Convert to undirected
        return graph_data



