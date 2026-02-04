import torch
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

class GraphReadout:
    def __init__(self, mode="sum"):
        assert mode in ["sum", "mean", "max"]
        self.mode = mode

    def __call__(self, x, batch):
        if self.mode == "sum":
            return global_add_pool(x, batch)
        if self.mode == "mean":
            return global_mean_pool(x, batch)
        return global_max_pool(x, batch)
