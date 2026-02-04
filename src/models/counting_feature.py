import torch
import torch.nn as nn

class CountingFeature(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed = nn.Linear(1, embed_dim)

    def forward(self, x, batch):
        num_nodes = torch.bincount(batch).float().unsqueeze(1)
        graph_emb = self.embed(num_nodes)
        return graph_emb
