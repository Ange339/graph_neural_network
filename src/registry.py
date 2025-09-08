from torch_geometric.nn import GCNConv, SAGEConv, VGAE 
from src.feature_extractor import *
from model import *


EMB_FEATURE_REGISTRY = {
    "topo": TopologicalFeatures,
    "textual_desc": TextualDescriptionFeatures,
    "textual_reviews": TextualReviewFeatures,
}


MODEL_REGISTRY = {
    "sage_encoder": SageEncoder,
    "gat_encoder": GATEncoder,
    "inner_product_decoder": InnerProductDecoder,
}