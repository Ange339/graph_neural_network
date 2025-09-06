import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, to_hetero ,BatchNorm

from torchsummary import summary

import logging

logger = logging.getLogger(__name__)


class VGAE(nn.Module):
    def __init__(self,data,encoder,decoder,cfg):
        """
        Classical VGAE model
        input: 
            data = input data
            encoder = encoder function
            decoder = decoder function
            cfg 
        """
        super().__init__()
        self.cfg = cfg

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

        self.encoder = encoder((-1,-1), hidden_channels, latent_dim, cfg)
        # self.encoder = encoder((-1,-1), hidden_channels, latent_dim, cfg)
        self.encoder = to_hetero(self.encoder, metadata=data.metadata())
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
        mu_dict, logvar_dict = self.encoder(x_dict, edge_index_dict)
        z_user = self.reparameterize(mu_dict["user"], logvar_dict["user"])
        z_item = self.reparameterize(mu_dict["item"], logvar_dict["item"])
        z_dict = {"user": z_user, "item": z_item}
        return z_dict, mu_dict, logvar_dict

    def decode(self, z_user, z_item, edge_index):
        return self.decoder(z_user, z_item, edge_index)




class VarSageEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_dim, cfg):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        
        for _ in range(1, cfg['n_layer']):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        self.conv_mu = SAGEConv(hidden_channels, latent_dim)
        self.conv_logvar = SAGEConv(hidden_channels, latent_dim)

        self.batch_norm = nn.ModuleList([BatchNorm(hidden_channels) for _ in range(cfg['n_layer'])]) if cfg.get("batch_norm", False) else None
        self.skip = cfg.get('skip_connection', False)


    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x_res = x if self.skip else None
            x = conv(x, edge_index)
            if self.batch_norm is not None:
                x = self.batch_norm[i](x)
            x = F.leaky_relu(x)
            if self.skip:
                x = x + x_res
        mu = self.conv_mu(x, edge_index)
        logvar = self.conv_logvar(x, edge_index)
        return mu, logvar



class VarGATEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_dim, cfg):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList() if cfg.get("batch_norm", False) else None
        self.self_linears = nn.ModuleList()

        self.convs.append(GATConv(in_channels, hidden_channels, add_self_loops=False, dropout=cfg['dropout']))
        self.self_linears.append(nn.Linear(-1, hidden_channels))
        if self.bns is not None:
            self.bns.append(BatchNorm(hidden_channels))

        for _ in range(1, cfg['n_layer']):
            self.convs.append(GATConv(hidden_channels, hidden_channels, heads=cfg['n_heads'], dropout=cfg['dropout']))
            self.self_linears.append(nn.Linear(-1, hidden_channels))
            if self.bns is not None:
                self.bns.append(BatchNorm(hidden_channels))

        self.conv_mu = GATConv(hidden_channels, latent_dim, heads=1, concat=False, dropout=cfg['dropout'])
        self.conv_logvar = GATConv(hidden_channels, latent_dim, heads=1, concat=False, dropout=cfg['dropout'])

        self.skip = cfg.get('skip_connection', False)


    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x_res = x if self.skip else None
            x = conv(x, edge_index) + self.self_linears[i](x)
            if self.bns is not None:
                x = self.bns[i](x)
            x = F.leaky_relu(x)
            if self.skip:
                x = x + x_res
        mu = self.conv_mu(x, edge_index)
        logvar = self.conv_logvar(x, edge_index)
        return mu, logvar


class VarGATMultiheadEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_dim, cfg):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList() if cfg.get("batch_norm", False) else None
        self.self_linears = nn.ModuleList()
        self.linears = nn.ModuleList()

        self.convs.append(GATConv(in_channels, hidden_channels, add_self_loops=False, heads=cfg['n_heads'], dropout=cfg['dropout']))
        self.self_linears.append(nn.Linear(-1, hidden_channels))
        self.linears.append(nn.Linear(-1, hidden_channels))
        if self.bns is not None:
            self.bns.append(BatchNorm(hidden_channels))

        for _ in range(1, cfg['n_layer']):
            self.convs.append(GATConv(hidden_channels, hidden_channels, heads=cfg['n_heads'], dropout=cfg['dropout']))
            self.self_linears.append(nn.Linear(-1, hidden_channels))
            self.linears.append(nn.Linear(-1, hidden_channels))
            if self.bns is not None:
                self.bns.append(BatchNorm(hidden_channels))

        self.conv_mu = GATConv(hidden_channels, latent_dim, heads=1, concat=False, dropout=cfg['dropout'])
        self.conv_logvar = GATConv(hidden_channels, latent_dim, heads=1, concat=False, dropout=cfg['dropout'])

        self.skip = cfg.get('skip_connection', False)


    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x_res = x if self.skip else None
            x = conv(x, edge_index) + self.self_linears[i](x)
            x = self.linears[i](x)
            if self.bns is not None:
                x = self.bns[i](x)
            x = F.leaky_relu(x)
            if self.skip:
                x = x + x_res
        mu = self.conv_mu(x, edge_index)
        logvar = self.conv_logvar(x, edge_index)
        return mu, logvar



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






