import torch.nn as nn
from layers.gin_conv import GINConv

class GINBlock(nn.Module):
    def __init__(self, in_dim, out_dim, eps=0.0):
        super().__init__()

        mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

        self.conv = GINConv(mlp, eps=eps, train_eps=True)
        self.bn = nn.BatchNorm1d(out_dim)
        self.act = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.bn(x)
        return self.act(x)
