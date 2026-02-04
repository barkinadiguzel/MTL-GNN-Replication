import torch.nn as nn

class MultiTaskHead(nn.Module):
    def __init__(self, in_dim, task_dims):
        super().__init__()

        self.heads = nn.ModuleDict()
        for task_name, out_dim in task_dims.items():
            self.heads[task_name] = nn.Linear(in_dim, out_dim)

    def forward(self, h_G):
        outputs = {}
        for task, head in self.heads.items():
            outputs[task] = head(h_G)
        return outputs
