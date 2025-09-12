from bertopic import BERTopic
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired
from hdbscan import HDBSCAN


## BERTopic Clustering
class BERTopicClustering:
    def __init__(self, cfg):
        self.cfg = cfg
    
    def model(self):
        dim_reduction_model = self.dim_reduction_model()
        clustering_model = self.clustering_model()
        vectorizer_model = self.vectorizer_model()
        ctfidf_model = self.ctfidf_model()
        representation_model = self.representation_model()

        topic_model = BERTopic(
            umap_model=dim_reduction_model,
            hdbscan_model=clustering_model,
            vectorizer_model=vectorizer_model,
            ctfidf_model=ctfidf_model,
            representation_model=representation_model,
            calculate_probabilities=False,
            low_memory=True,
            verbose=True
        )
        
        return topic_model
        

    def dim_reduction_model(self):
        cfg = self.cfg['dim_reduction']
        if cfg['name'] == 'umap':
            dim_reduction_model = UMAP(
                n_neighbors=cfg.get('n_neighbors', 15),
                n_components=cfg.get('n_components', 5), 
                min_dist=cfg.get('min_dist', 0.0),
                metric=cfg.get('metric', 'cosine'),
                random_state=42
                )
        else:
            raise NotImplementedError
        return dim_reduction_model

    def clustering_model(self):
        cfg = self.cfg['clustering']
        if cfg['name'] == 'hdbscan':
            clustering_model = HDBSCAN(
                min_cluster_size=cfg.get('min_cluster_size', 15),
                metric=cfg.get('metric', 'euclidean'),
                cluster_selection_method=cfg.get('cluster_selection_method', 'eom')
            )
        else:
            raise NotImplementedError
        return clustering_model
    
    def vectorizvectorizer_model(self):
        cfg = self.cfg['vectorizer']
        if cfg['name'] == 'count':
            vectorizer_model = CountVectorizer(
                ngram_range=cfg.get('ngram_range', (1, 2)),
                stop_words=cfg.get('stop_words', 'english'),
                min_df=cfg.get('min_df', 5)
            )
        else:
            raise NotImplementedError
        return vectorizer_model
    
    def ctfidf_model(self):
        cfg = self.cfg['ctfidf']
        if cfg['name'] == 'ctfidf':
            ctfidf_model = ClassTfidfTransformer(
                reduce_frequent_words=cfg.get('reduce_frequent_words', True),
                low_memory=cfg.get('low_memory', False)
            )
        else:
            raise NotImplementedError
        return ctfidf_model
    
    def representation_model(self):
        cfg = self.cfg['representation']
        if cfg['name'] == 'keybert':
            representation_model = KeyBERTInspired(
                top_n_words=cfg.get('top_n_words', 10),
                diversity=cfg.get('diversity', 0.5)
            )
        else:
            raise NotImplementedError
        return representation_model
