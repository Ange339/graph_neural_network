import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, to_hetero, BatchNorm, Linear, Sequential
from abc import ABC, abstractmethod




class GNNLayer(nn.Module, ABC):
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 batch_norm=False, 
                 skip_connection=None,
                 dropout=0.0, 
                 heads=1, 
                 is_final_layer=False,
                 activation=F.leaky_relu):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batch_norm_flag = batch_norm
        self.skip_connection = skip_connection
        self.dropout = dropout
        self.heads = heads
        self.is_final_layer = is_final_layer
        self.activation = activation

        # Build convolution
        self.conv, conv_out_dim = self._build_conv(in_channels, out_channels, heads, dropout)

        # If multiple heads and GATConv → fuse into out_channels
        self.fuse_linear = Linear(-1, out_channels) if heads > 1 and isinstance(self.conv, GATConv) else None

        # BatchNorm (optional)
        self.batch_norm = BatchNorm(conv_out_dim) if batch_norm else None

        # Skip connection (optional)
        self.skip_linear = Linear(-1, conv_out_dim) if skip_connection else None

        # # Store actual output dim (important for GAT with multiple heads)
        # self.final_output_dim = conv_out_dim

    @abstractmethod
    def _build_conv(self, in_channels, out_channels, heads, dropout):
        """Return (conv_layer, output_dim). Must be implemented by subclasses."""
        pass

    def forward(self, x, edge_index):
        # Residual path
        x_res = x if self.skip_linear else None

        # Graph convolution
        x = self.conv(x, edge_index)

        # If multiple heads → fuse into out_channels
        if self.fuse_linear:
            x = self.fuse_linear(x)            
        
        # Skip connection
        if self.skip_connection == 'sum':
            x = x + self.skip_linear(x_res)
        elif self.skip_connection == "concat":
            x = torch.cat([x, self.skip_linear(x_res)], dim=-1)

        # If last layer → return raw output
        if self.is_final_layer:
            return x

        # BatchNorm
        if self.batch_norm:
            x = self.batch_norm(x)

        # Activation
        if self.activation:
            x = self.activation(x)

        # Dropout
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class SageLayer(GNNLayer):
    def _build_conv(self, in_channels, out_channels, heads, dropout):
        conv = SAGEConv(in_channels, out_channels)
        return conv, out_channels


class GATLayer(GNNLayer):
    def _build_conv(self, in_channels, out_channels, heads, dropout):
        # Use concat=True → output dim = heads * out_channels
        gat_conv = GATConv(in_channels, out_channels,
                           heads=heads, concat=True,
                           add_self_loops=False, dropout=dropout)

        return gat_conv, out_channels

        # conv_out_dim = heads * out_channels

        # # If multiple heads → fuse into out_channels
        # if heads > 1:
        #     fusion = Linear(-1, out_channels)

        #     conv = nn.Sequential(gat_conv, fusion)
        #     return conv, out_channels
        # else:
        #     return gat_conv, out_channels




# class GNNLayer(nn.Module, ABC):
#     def __init__(self, in_channels, out_channels, batch_norm=False, skip=None, dropout=0.0, heads=1, is_final_layer=False):
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.batch_norm_flag = batch_norm
#         self.skip = skip
#         self.dropout = dropout
#         self.heads = heads
#         self.is_final_layer = is_final_layer

#     @abstractmethod
#     def forward(self, x, edge_index):
#         pass



# class SageLayer(GNNLayer):
#     def __init__(self, in_channels, out_channels, batch_norm=False, skip=None, dropout=0.0, heads=1, is_final_layer=False):
#         super().__init__()
#         self.conv = SAGEConv(in_channels, out_channels)
#         self.batch_norm = BatchNorm(out_channels) if batch_norm else None
#         self.skip = skip
#         self.skip_linear = Linear(-1, out_channels) if skip else None
#         self.dropout = dropout
#         self.is_final_layer = is_final_layer


#     def forward(self, x, edge_index):
#         x_res = x if self.skip_linear else None
#         x = self.conv(x, edge_index)
#         if self.is_final_layer:
#             return x
#         if self.batch_norm:
#             x = self.batch_norm(x)
#         x = F.leaky_relu(x)
#         # if self.skip == 'sum':
#         #     x = x + self.skip_linear(x_res)
#         # elif self.skip == 'concat':
#         #     x = torch.cat([x, self.skip_linear(x_res)], dim=-1)
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         return x


# class GATLayer(nn.Module):
#     def __init__(self, in_channels, out_channels, batch_norm=False, skip=None, dropout=0.0, heads=1, is_final_layer=False):
#         super().__init__()
#         self.gat_conv = GATConv(in_channels, out_channels, heads=heads, concat=False, add_self_loops=False, dropout=dropout)
#         self.batch_norm = BatchNorm(out_channels) if batch_norm else None
#         self.skip_linear = Linear(-1, out_channels)
#         self.linear = Linear(in_channels * heads, out_channels) if heads > 1 else None
#         self.is_final_layer = is_final_layer


#     def forward(self, x, edge_index):
#         x_res = x if self.skip_linear else None
#         x = self.gat_conv(x, edge_index)
#         if self.linear is not None:
#             x = self.linear(x)
#         x = x + self.skip_linear(x)

#         if self.is_final_layer:
#             return x

#         if self.batch_norm:
#             x = self.batch_norm(x)
#         x = F.leaky_relu(x)
#         return x