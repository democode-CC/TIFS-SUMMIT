import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()

        self.conv1 = GATConv(args.in_dim, args.hidden_dim)
        self.conv2 = GATConv(args.hidden_dim, args.out_dim)

        self.hidden_dim_mu = int(args.hidden_dim / 2)
        self.hidden_dim_sigma = int(args.hidden_dim / 2)
        def xavier_init(layer):
            nn.init.xavier_normal_(layer.weight)

        self.linear_mu = nn.Linear(args.hidden_dim,  self.hidden_dim_mu)
        self.linear_sigma = nn.Linear(args.hidden_dim,  self.hidden_dim_sigma)
        xavier_init(self.linear_mu)
        xavier_init(self.linear_sigma)


    def encode(self,x, edge_index):
        x1 = self.conv1(x, edge_index)
        x1_ = F.relu(x1)
        mu = self.linear_mu(x1_)
        sigma = F.elu(self.linear_sigma(x1_)) + 1
        return x1, mu, sigma

    def forward(self, x, edge_index, return_all_emb=False):
        x1, _, _ = self.encode(x,edge_index)
        x1_ = F.relu(x1)
        x2 = self.conv2(x1_, edge_index)

        if return_all_emb:
            return x1, x2

        return x2


    def decode(self, z, pos_edge_index, neg_edge_index=None):
        if neg_edge_index is not None:
            edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
            logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

        else:
            edge_index = pos_edge_index
            logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

        return logits


