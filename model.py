import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, VGAE, BatchNorm

from torchsummary import summary

import logging

logger = logging.getLogger(__name__)


GNN_LAYER_REGISTRY = {
    "gcn_conv": GCNConv,
    "sage_conv": SAGEConv,
    "vgae": VGAE,
}

class VGAE(nn.Module):
    def __init__(self,data,encoder,decoder,cfg):
        """
        gnn_layer_cls: class of the GNN layer (e.g., GCNConv, SAGEConv)
        encoder_cls: class of the encoder (should accept gnn_layer_cls, in_channels, hidden_channels, latent_dim)
        decoder_cls: class of the decoder (should accept no arguments)
        in_channels: input feature dimension
        hidden_channels: hidden layer dimension
        latent_dim: latent space dimension
        encoder_kwargs: additional kwargs for encoder
        """
        super().__init__()
        
        user_in_channels=data["user"].num_features
        item_in_channels=data["item"].num_features
        hidden_channels=cfg['hidden_channels']
        latent_dim=cfg['latent_dim']
        emb_linear_transform=cfg.get('emb_linear_transform', False)

        self.emb_linear_transform = cfg.get('emb_linear_transform', False)
        self.user_linear = nn.Linear(user_in_channels, hidden_channels)
        self.item_linear = nn.Linear(item_in_channels, hidden_channels)

        if user_in_channels != item_in_channels:
            logger.warning(f"Inconsistent input feature dimensions: user={user_in_channels}, item={item_in_channels}")
            logger.warning(f"Forcing linear transformation")
            self.emb_linear_transform = True

        elif user_in_channels == item_in_channels == hidden_channels:
            logger.warning(f"Consistent input feature dimensions: user={user_in_channels}, item={item_in_channels}, hidden={hidden_channels}")
            logger.warning(f"Disabling linear transformation")
            self.emb_linear_transform = False

        input_channels = hidden_channels if self.emb_linear_transform else user_in_channels

        self.encoder = encoder(input_channels, hidden_channels, latent_dim, cfg)
        self.decoder = decoder()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, data):
        if self.emb_linear_transform:
            user_x = self.user_linear(data["user"].x)
            item_x = self.item_linear(data["item"].x)
        else:
            user_x = data["user"].x
            item_x = data["item"].x

        x = torch.cat([user_x, item_x], dim=0)
        edge_index = data['user', 'interacts', 'item'].edge_index

        mu, logvar = self.encoder(x, edge_index)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, z, edge_index):
        return self.decoder(z, edge_index)

    def summary(self):
        logger.info("Model Summary:")
        logger.info("----------------")
        logger.info("Encoder:")
        logger.info(pprint.pformat(self.encoder))
        logger.info("Decoder:")
        logger.info(pprint.pformat(self.decoder))

class VGAEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_dim, cfg):
        """
        in_channels: dimension of the node input features (after projecting items and using user embeddings)
        hidden_channels: hidden units for first GCN layer
        latent_dim: dimension of mu and logvar
        """
        super().__init__()
        GNNLayer = GNN_LAYER_REGISTRY[cfg['gnn_layer_cls']]
        self.convs = nn.ModuleList()
        self.convs.append(GNNLayer(in_channels, hidden_channels))
        for _ in range(cfg['n_layer'] - 1):
            self.convs.append(GNNLayer(hidden_channels, hidden_channels))
        self.conv_mu = GNNLayer(hidden_channels, latent_dim)
        self.conv_logvar = GNNLayer(hidden_channels, latent_dim)
        self.batch_norm = BatchNorm(hidden_channels) if cfg.get("batch_norm", False) else None
        self.skip = cfg.get('skip_connection', False)

    def forward(self, x, edge_index):
        for conv in self.convs:
            if self.skip:
                x_res = x
            x = conv(x, edge_index)
            if self.batch_norm:
                x = self.batch_norm(x)
            x = F.leaky_relu(x)
            if self.skip:
                x = x + x_res
        mu = self.conv_mu(x, edge_index)
        logvar = self.conv_logvar(x, edge_index)
        return mu, logvar


class InnerProductDecoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z, edge_index, sigmoid=True):
        edge_feat_user = z[edge_index[0]]
        edge_feat_item = z[edge_index[1]]
        preds = (edge_feat_user * edge_feat_item).sum(dim=-1)
        if sigmoid:
            preds = torch.sigmoid(preds)
        return preds
    
    def forward_all(self, z, sigmoid=True):
        "Use it cautiously, as it computes all pairwise interactions"
        A_pred = (z @ z.T).view(-1)
        if sigmoid:
            A_pred = torch.sigmoid(A_pred)
        return A_pred

