import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class GINConv(MessagePassing):
    def __init__(self, mlp, eps=0.0, train_eps=False):
        super().__init__(aggr="add")  
        self.mlp = mlp

        if train_eps:
            self.eps = nn.Parameter(torch.tensor(eps))
        else:
            self.register_buffer("eps", torch.tensor(eps))

    def forward(self, x, edge_index):
        out = self.propagate(edge_index, x=x)
        out = (1 + self.eps) * x + out
        return self.mlp(out)

    def message(self, x_j):
        return x_j
