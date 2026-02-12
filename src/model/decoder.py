# -*- coding: utf-8 -*-
"""Graph Decoders"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCN
import torch.nn.functional as F

class BetaStructureDecoder(nn.Module):
    def __init__(self,
                 in_dim,
                 hid_dim=64,
                 num_layers=1,
                 dropout=0.,
                 act=torch.nn.functional.relu,
                 backbone=GCN,
                 backbone_norm="layer_norm",
                 **kwargs):
        super(BetaStructureDecoder, self).__init__()
        self.proj_alpha = backbone(in_channels=in_dim,
                                   hidden_channels=hid_dim,
                                   num_layers=num_layers,
                                   out_channels=hid_dim,
                                   dropout=dropout,
                                   act=act,
                                   norm=backbone_norm, 
                                   **kwargs)              
        self.proj_beta = backbone(in_channels=in_dim,
                                  hidden_channels=hid_dim,
                                  num_layers=num_layers,
                                  out_channels=hid_dim,
                                  dropout=dropout,
                                  act=act,
                                  norm=backbone_norm, 
                                  **kwargs)

    def forward(self, x, edge_index):
        h_alpha = self.proj_alpha(x, edge_index)
        h_beta = self.proj_beta(x, edge_index)
        pred_alpha = h_alpha @ h_alpha.T
        pred_beta = h_beta @ h_beta.T
        alpha = F.softplus(pred_alpha) + 1.0 + 1e-4
        beta = F.softplus(pred_beta) + 1.0 + 1e-4
        return alpha, beta