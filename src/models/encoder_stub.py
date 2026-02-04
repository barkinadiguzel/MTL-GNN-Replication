import torch.nn as nn
from blocks.gin_block import GINBlock
from layers.graph_readout import GraphReadout

class GraphEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, pooling="sum"):
        super().__init__()

        self.layers = nn.ModuleList()
        self.layers.append(GINBlock(in_dim, hidden_dim))

        for _ in range(num_layers - 1):
            self.layers.append(GINBlock(hidden_dim, hidden_dim))

        self.readout = GraphReadout(mode=pooling)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for layer in self.layers:
            x = layer(x, edge_index)

        h_G = self.readout(x, batch)
        return h_G
