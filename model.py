import pprint
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, to_hetero, BatchNorm, Linear
from torchsummary import summary
from src.gnn_layers import GNNLayer, SageLayer, GATLayer

import logging

logger = logging.getLogger(__name__)



class GAE(nn.Module):
    def __init__(self, data, encoder, decoder, cfg):
        """
        Generalized VGAE/GAE model
        input: 
            data = input data
            encoder = encoder function
            decoder = decoder function
            cfg 
        """
        super().__init__()
        self.cfg = cfg
        self.variational = cfg.get('variational', True)  # Toggle for VGAE or GAE

        user_emb_dim = cfg['user_emb_dim']
        item_emb_dim = cfg['item_emb_dim']

        self.user_embedding = nn.Embedding(data["user"].num_nodes, user_emb_dim)
        self.item_embedding = nn.Embedding(data["item"].num_nodes, item_emb_dim)

        # Xavier initialization
        torch.nn.init.xavier_uniform_(self.user_embedding.weight)
        torch.nn.init.xavier_uniform_(self.item_embedding.weight)

        ## Linear projection layer if needed
        user_feat_dim = data["user"].num_features
        item_feat_dim = data["item"].num_features

        if cfg.get('user_feature_linear', False) and user_feat_dim > 0:
            self.user_feature_linear = nn.Linear(user_feat_dim, cfg['feature_linear_dim'])
            user_feat_dim = cfg['feature_linear_dim']

        if cfg.get('item_feature_linear', False) and item_feat_dim > 0:
            self.item_feature_linear = nn.Linear(item_feat_dim, cfg['feature_linear_dim'])
            item_feat_dim = cfg['feature_linear_dim']

        # Parameters counting
        user_in_channels = user_feat_dim + user_emb_dim
        item_in_channels = item_feat_dim + item_emb_dim
        input_channels = (user_in_channels, item_in_channels)
        hidden_channels = cfg['hidden_channels']
        latent_dim = cfg['latent_dim']
        logger.debug(f"Input channels: {input_channels}, Hidden channels: {hidden_channels}, Latent dim: {latent_dim}")

        self.encoder = encoder((-1, -1), hidden_channels, latent_dim, cfg)
        #self.encoder = to_hetero(self.encoder, metadata=data.metadata())
        self.encoder = self._wrap_layers(self.encoder, metadata=data.metadata(), aggr='mean')
        self.decoder = decoder()

    def _process_features(self, node_data, node_type, embedding, lin_project):
        if "x" not in node_data:
            return embedding

        feature = node_data.x

        if lin_project:
            embedding = self.user_feature_linear(feature) if node_type == "user" else self.item_feature_linear(feature)

        aggr_method = self.cfg.get('feature_aggr_method', 'concat')

        if aggr_method != "concat" and embedding.size(-1) != feature.size(-1):
            logger.warning(f"{node_type.capitalize()} embedding dim {embedding.size(-1)} and feature dim {feature.size(-1)} do not match. Using concat instead of {aggr_method}.")
            aggr_method = 'concat'

        if aggr_method == "concat":
            return torch.cat([embedding, feature], dim=-1)
        elif aggr_method == "add":
            return embedding + feature
        else:
            logger.fatal(f"Unknown aggregation method: {aggr_method}")
            raise ValueError(f"Unknown aggregation method: {aggr_method}")

    def _wrap_layers(self, module, metadata, aggr='mean'):
        """
        Recursively wraps all customer GNNlayers or MessagePassing submodules with to_hetero.
        """
        for name, child in module.named_children():
            # If the child is a custom GNN layer
            if isinstance(child, GNNLayer):
                # Replace the original child module with its heterogeneous version using to_hetero
                # This is necessary because to_hetero returns a new module instance
                setattr(module, name, to_hetero(child, metadata, aggr=aggr))
            # Recurse for children of children
            else:
                self._wrap_layers(child, metadata, aggr)
        return module

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, data):
        user_emb = self.user_embedding(data["user"].node_id)
        item_emb = self.item_embedding(data["item"].node_id)

        user_x = self._process_features(data["user"], "user", user_emb, self.cfg.get('user_feature_linear', False))
        item_x = self._process_features(data["item"], "item", item_emb, self.cfg.get('item_feature_linear', False))

        x_dict = {"user": user_x, "item": item_x}
        edge_index_dict = data.edge_index_dict

        if self.variational:
            mu_dict, logvar_dict = self.encoder(x_dict, edge_index_dict)
            z_user = self.reparameterize(mu_dict["user"], logvar_dict["user"])
            z_item = self.reparameterize(mu_dict["item"], logvar_dict["item"])
        else:
            z_out = self.encoder(x_dict, edge_index_dict)
            z_user = z_out["user"]
            z_item = z_out["item"]
            mu_dict, logvar_dict = None, None

        z_dict = {"user": z_user, "item": z_item}
        return z_dict, mu_dict, logvar_dict

    def decode(self, z_user, z_item, edge_index):
        return self.decoder(z_user, z_item, edge_index)



#### ENCODERS ####

class GraphEncoderBase(nn.Module, ABC):
    def __init__(self, in_channels, hidden_channels, latent_dim, cfg):
        super().__init__()
        self.variational = cfg.get("variational", True)
        batch_norm = cfg.get("batch_norm", False)
        skip_connection = cfg.get("skip_connection", None)
        dropout = cfg.get("dropout", 0.30)
        heads = cfg.get("heads", 1)
        n_layer = cfg["n_layer"]

        # Hidden layers
        self.layers = nn.ModuleList()
        self.layers.append(self._make_layer(in_channels, hidden_channels, batch_norm=batch_norm, skip_connection=skip_connection, dropout=dropout, heads=heads))
        for _ in range(1, n_layer):
            self.layers.append(self._make_layer(hidden_channels, hidden_channels, batch_norm=batch_norm, skip_connection=skip_connection, dropout=dropout, heads=heads))

        # Output layers
        if self.variational:
            self.conv_mu = self._make_layer(hidden_channels, latent_dim, batch_norm=batch_norm, skip_connection=None, dropout=0, heads=1, is_final_layer=True)
            self.conv_logvar = self._make_layer(hidden_channels, latent_dim, batch_norm=batch_norm, skip_connection=None, dropout=0, heads=1, is_final_layer=True)
        else:
            self.conv_out = self._make_layer(hidden_channels, latent_dim, batch_norm=batch_norm, skip_connection=skip_connection, dropout=0, heads=1, is_final_layer=True)

    @abstractmethod
    def _make_layer(self, in_channels, out_channels, batch_norm=True, skip='sum', dropout=0.3, heads=1, is_final_layer=False):
        """Return a GNNLayer (e.g. SageLayer, GATLayer)."""
        pass

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)

        if self.variational:
            mu = self.conv_mu(x, edge_index)
            logvar = self.conv_logvar(x, edge_index)
            return mu, logvar
        else:
            return self.conv_out(x, edge_index)


class SageEncoder(GraphEncoderBase):
    def _make_layer(self, in_channels, out_channels, batch_norm=True, skip_connection='sum', dropout=0.3, heads=1, is_final_layer=False):
        return SageLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            batch_norm=batch_norm,
            skip_connection=skip_connection,
            dropout=0 if is_final_layer else dropout,
            heads=1,
            is_final_layer=is_final_layer
        )


class GATEncoder(GraphEncoderBase):
    def _make_layer(self, in_channels, out_channels, batch_norm=True, skip_connection='sum', dropout=0.3, heads=1, is_final_layer=False):
        return GATLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            batch_norm=batch_norm,
            skip_connection=skip_connection if not is_final_layer else None,
            dropout=0 if is_final_layer else dropout,
            heads=heads,
            is_final_layer=is_final_layer
        )



# class SageEncoder(nn.Module):
#     def __init__(self, in_channels, hidden_channels, latent_dim, cfg):
#         super().__init__()
#         self.variational = cfg.get('variational', True)
#         self.layers = nn.ModuleList()
#         batch_norm = cfg.get("batch_norm", False)
#         skip = cfg.get('skip_connection', None)
#         self.dropout = cfg.get('dropout', 0.30)
#         self.layers.append(SageLayer(in_channels, hidden_channels, batch_norm=batch_norm, skip=skip, dropout=self.dropout))
#         for _ in range(1, cfg['n_layer']):
#             self.layers.append(SageLayer(hidden_channels, hidden_channels, batch_norm=batch_norm, skip=skip, dropout=self.dropout))
        
#         if self.variational:
#             self.conv_mu = SageLayer(hidden_channels, latent_dim, batch_norm=False, skip=None, dropout=0, is_final_layer=True)
#             self.conv_logvar = SageLayer(hidden_channels, latent_dim, batch_norm=False, skip=None, dropout=0, is_final_layer=True)
#         else:
#             self.conv_out = SageLayer(hidden_channels, latent_dim, batch_norm=False, skip=skip, dropout=0, is_final_layer=True)


#     def forward(self, x, edge_index):
#         for layer in self.layers:
#             x = layer(x, edge_index)
#         if self.variational:
#             mu = self.conv_mu(x, edge_index)
#             logvar = self.conv_logvar(x, edge_index)
#             return mu, logvar
#         else:
#             out = self.conv_out(x, edge_index)
#             return out


# class GATEncoder(nn.Module):
#     def __init__(self, in_channels, hidden_channels, latent_dim, cfg):
#         super().__init__()
#         self.variational = cfg.get('variational', True)
#         self.layers = nn.ModuleList()
#         batch_norm = cfg.get("batch_norm", False)
#         skip = cfg.get('skip_connection', None)
#         dropout = cfg.get('dropout', 0.30)
#         heads = cfg.get('heads', 1)
#         self.layers.append(GATLayer(in_channels, hidden_channels, batch_norm=batch_norm, skip=skip, dropout=dropout, heads=heads))
#         for _ in range(1, cfg['n_layer']):
#             self.layers.append(GATLayer(hidden_channels, hidden_channels, batch_norm=batch_norm, skip=skip, dropout=dropout, heads=heads))

#         if self.variational:
#             self.conv_mu = GATLayer(hidden_channels, latent_dim, batch_norm=False, skip=None, dropout=0, heads=1, is_final_layer=True)
#             self.conv_logvar = GATLayer(hidden_channels, latent_dim, batch_norm=False, skip=None, dropout=0, heads=1, is_final_layer=True)
#         else:
#             self.conv_out = GATLayer(hidden_channels, latent_dim, batch_norm=False, skip=skip, dropout=0, heads=1, is_final_layer=True)

#     def forward(self, x, edge_index):
#         for layer in self.layers:
#             x = layer(x, edge_index)
#         if self.variational:
#             mu = self.conv_mu(x, edge_index)
#             logvar = self.conv_logvar(x, edge_index)
#             return mu, logvar
#         else:
#             out = self.conv_out(x, edge_index)
#             return out


class InnerProductDecoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z_user, z_item, edge_index, sigmoid=True):
        # Split z into user_z and item_z

        # Decode using edge_index
        edge_feat_user = z_user[edge_index[0]]
        edge_feat_item = z_item[edge_index[1]]
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
    
    def batch_forward_all(self, z_user, z_item, sigmoid=True):
        "Compute all pairwise interactions in a batched manner to save memory"
        batch_size = 100  # Adjust based on memory constraints
        num_users = z_user.size(0)
        num_items = z_item.size(0)
        z_item_t = z_item.T
        preds = []
        for start in range(0, num_users, batch_size):
            end = min(start + batch_size, num_users)
            batch_user = z_user[start:end]
            batch_preds = torch.matmul(batch_user, z_item_t)
            if sigmoid:
                batch_preds = torch.sigmoid(batch_preds)
            preds.append(batch_preds)

        return torch.cat(preds, dim=0)



# class SageEncoder(nn.Module):
#     """
#     GraphSAGE-based encoder for (V)GAE models.

#     Args:
#         in_channels (int): Number of input features per node.
#         hidden_channels (int): Number of hidden units per layer.
#         latent_dim (int): Size of the latent embedding.
#         cfg (dict): Configuration dictionary. Keys include:
#             - 'n_layer': Number of layers.
#             - 'variational': If True, outputs (mu, logvar); else, outputs embedding.
#             - 'batch_norm': If True, applies BatchNorm after each layer.
#             - 'skip_connection': If True, adds skip connections.

#     Input shape:
#         x: Tensor of shape [num_nodes, in_channels]
#         edge_index: LongTensor of shape [2, num_edges]

#     Output shape:
#         If variational: (mu, logvar), each of shape [num_nodes, latent_dim]
#         Else: embedding of shape [num_nodes, latent_dim]
#     """
#     def __init__(self, in_channels, hidden_channels, latent_dim, cfg):
#         """
#         Initialize the SageEncoder.

#         Args:
#             in_channels (int): Number of input features per node.
#             hidden_channels (int): Number of hidden units per layer.
#             latent_dim (int): Size of the latent embedding.
#             cfg (dict): Configuration dictionary.
#         """
#         super().__init__()
#         self.variational = cfg.get('variational', True)
#         self.dropout = cfg.get('dropout', 0.30)
#         self.convs = nn.ModuleList()
#         self.convs.append(SAGEConv(in_channels, hidden_channels))
        
#         for _ in range(1, cfg['n_layer']):
#             self.convs.append(SAGEConv(hidden_channels, hidden_channels))

#         if self.variational:
#             self.conv_mu = SAGEConv(hidden_channels, latent_dim)
#             self.conv_logvar = SAGEConv(hidden_channels, latent_dim)
#         else:
#             self.conv_out = SAGEConv(hidden_channels, latent_dim)

#         self.bns = nn.ModuleList([BatchNorm(hidden_channels) for _ in range(cfg['n_layer'])]) if cfg.get("batch_norm", False) else None
#         self.skip = cfg.get('skip_connection', None)

#     def forward(self, x, edge_index):
#         """
#         Forward pass of the SageEncoder.

#         Args:
#             x (Tensor): Node features of shape [num_nodes, in_channels].
#             edge_index (LongTensor): Edge indices of shape [2, num_edges].

#         Returns:
#             If variational:
#                 mu (Tensor): Mean embeddings of shape [num_nodes, latent_dim].
#                 logvar (Tensor): Log-variance embeddings of shape [num_nodes, latent_dim].
#             Else:
#                 out (Tensor): Embeddings of shape [num_nodes, latent_dim].
#         """
#         for i, conv in enumerate(self.convs):
#             x_res = x.clone() if i > 0 and self.skip else None
#             x = conv(x, edge_index)
#             if self.bns is not None:
#                 x = self.bns[i](x)
#             x = F.leaky_relu(x)
#             if i > 0 and self.skip == 'sum':
#                 x = x + x_res
#             x = F.dropout(x, p=self.dropout, training=self.training)
            
#         if self.variational:
#             mu = self.conv_mu(x, edge_index)
#             logvar = self.conv_logvar(x, edge_index)
#             return mu, logvar
#         else:
#             out = self.conv_out(x, edge_index)
#             out = F.dropout(out, p=self.dropout, training=self.training)
#             return out




# class GATEncoder(nn.Module):
#     def __init__(self, in_channels, hidden_channels, latent_dim, cfg):
#         super().__init__()
#         self.variational = cfg.get('variational', True)
#         self.convs = nn.ModuleList()
#         self.self_linears = nn.ModuleList()

#         self.convs.append(GATConv(in_channels, hidden_channels, heads=1, add_self_loops=False, dropout=cfg['dropout']))
#         self.self_linears.append(nn.Linear(-1, hidden_channels))

#         for _ in range(1, cfg['n_layer']):
#             self.convs.append(GATConv(hidden_channels, hidden_channels, heads=1, add_self_loops=False, dropout=cfg['dropout']))
#             self.self_linears.append(nn.Linear(-1, hidden_channels))

#         if self.variational:
#             self.conv_mu = GATConv(hidden_channels, latent_dim, heads=1, add_self_loops=False, dropout=0) # We don't apply dropout here
#             self.conv_logvar = GATConv(hidden_channels, latent_dim, heads=1, add_self_loops=False, dropout=0)
#         else:
#             self.conv_out = GATConv(hidden_channels, latent_dim, heads=1, add_self_loops=False, dropout=cfg['dropout'])

#         self.bns = nn.ModuleList([BatchNorm(hidden_channels) for _ in range(cfg['n_layer'])]) if cfg.get("batch_norm", False) else None
#         self.skip = cfg.get('skip_connection', None)


#     def forward(self, x, edge_index):
#         "Forward pass"
#         for i, conv in enumerate(self.convs):
#             x_res = x.clone() if i > 0 and self.skip == 'sum' else None
#             x = conv(x, edge_index) + self.self_linears[i](x)
#             if self.bns is not None:
#                 x = self.bns[i](x)
#             x = F.leaky_relu(x)
#             if i > 0 and self.skip == 'sum':
#                 x = x + x_res
            
#         if self.variational:
#             mu = self.conv_mu(x, edge_index)
#             logvar = self.conv_logvar(x, edge_index)
#             return mu, logvar
#         else:
#             out = self.conv_out(x, edge_index)
#             return out


# class GATMultiheadEncoder(nn.Module):
#     def __init__(self, in_channels, hidden_channels, latent_dim, cfg):
#         super().__init__()
#         self.variational = cfg.get('variational', True)
#         self.convs = nn.ModuleList()
#         self.bns = nn.ModuleList() if cfg.get("batch_norm", False) else None
#         self.self_linears = nn.ModuleList()
#         self.linears = nn.ModuleList()

#         self.convs.append(GATConv(in_channels, hidden_channels, add_self_loops=False, heads=cfg['n_heads'], dropout=cfg['dropout']))
#         self.self_linears.append(nn.Linear(-1, hidden_channels))
#         self.linears.append(nn.Linear(-1, hidden_channels))
#         if self.bns is not None:
#             self.bns.append(BatchNorm(hidden_channels))

#         for _ in range(1, cfg['n_layer']):
#             self.convs.append(GATConv(hidden_channels, hidden_channels, heads=cfg['n_heads'], dropout=cfg['dropout']))
#             self.self_linears.append(nn.Linear(-1, hidden_channels))
#             self.linears.append(nn.Linear(-1, hidden_channels))
#             if self.bns is not None:
#                 self.bns.append(BatchNorm(hidden_channels))

#         if self.variational:
#             self.conv_mu = GATConv(hidden_channels, latent_dim, heads=1, concat=False, dropout=0)
#             self.conv_logvar = GATConv(hidden_channels, latent_dim, heads=1, concat=False, dropout=0)
#         else:
#             self.conv_out = GATConv(hidden_channels, latent_dim, heads=1, concat=False, dropout=cfg['dropout'])

#         self.skip = cfg.get('skip_connection', False)


#     def forward(self, x, edge_index):
#         for i, conv in enumerate(self.convs):
#             x_res = x if self.skip else None
#             x = conv(x, edge_index) + self.self_linears[i](x)
#             x = self.linears[i](x)
#             if self.bns is not None:
#                 x = self.bns[i](x)
#             x = F.leaky_relu(x)
#             if self.skip:
#                 x = x + x_res
#         if self.variational:
#             mu = self.conv_mu(x, edge_index)
#             logvar = self.conv_logvar(x, edge_index)
#             return mu, logvar
#         else:
#             out = self.conv_out(x, edge_index)
#             return out








