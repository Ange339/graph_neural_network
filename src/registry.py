from torch_geometric.nn import GCNConv, SAGEConv, VGAE 

from src.feature_extractor import *
from model import *


EMB_FEATURE_REGISTRY = {
    "topo": TopologicalFeatures,
    "random": RandomFeatures,
    "textual_desc": TextualDescriptionFeatures,
    "textual_reviews": TextualReviewFeatures,
}


TRAIN_REGISTRY = {
    "vgae_encoder": VGAEncoder,
    "inner_product_decoder": InnerProductDecoder,
}