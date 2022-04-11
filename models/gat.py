import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from models.transformer_conv import TransformerConv


class GatNet(nn.Module):
    def __init__(self, node_feat_size, conv_hidden_size, dropout=0.2):
        super(GatNet, self).__init__()
        self.conv1 = TransformerConv(node_feat_size, conv_hidden_size // 2, dropout=dropout, heads=2, beta=0.0)

        self.gate_layer = nn.Sequential(
            nn.Linear(2 * node_feat_size, node_feat_size, bias=False),
            nn.Sigmoid()
        )
        self.fuse_layer = nn.Sequential(
            nn.Linear(node_feat_size, node_feat_size, bias=False),
            nn.Tanh()
        )
        self.output_proj = nn.Linear(node_feat_size, node_feat_size)
        # self.layer_norm = nn.LayerNorm(node_feat_size)

    def forward(self, nodes, edge_index):
        """
        nodes: shape [*, node_feat_size]
        edge_index: shape [2, *]
        """
        x = self.conv1(nodes, edge_index)
        # x = self.output_proj(x)
        h = torch.cat([nodes, x], dim=-1)
        gate = self.gate_layer(h)
        output = gate * self.fuse_layer(x) + (1.0 - gate) * nodes
        # output = self.gate_layer(h) * self.fuse_layer(h)
        
        # output = self.layer_norm(x + nodes)
        return output