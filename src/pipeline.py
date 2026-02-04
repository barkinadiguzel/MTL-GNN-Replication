import torch
from torch_geometric.data import Data

from config import *
from models.encoder_stub import GraphEncoder
from models.multitask_head_stub import MultiTaskHead
from utils.task_mask import apply_task_mask
from utils.graph_visualization import visualize_graph


def build_dummy_graph():
    x = torch.randn(5, INPUT_DIM)  # 5 atom
    edge_index = torch.tensor([
        [0, 1, 2, 3, 3, 4],
        [1, 0, 3, 2, 4, 3]
    ])
    batch = torch.zeros(x.size(0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, batch=batch)


def main():
    data = build_dummy_graph()

    encoder = GraphEncoder(
        in_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_GIN_LAYERS,
        pooling=POOLING_TYPE
    )

    head = MultiTaskHead(
        in_dim=HIDDEN_DIM,
        task_dims=TASKS
    )

    h_G = encoder(data)
    task_outputs = head(h_G)

    task_mask = {
        "toxicity": torch.tensor([1.0]),
        "solubility": torch.tensor([0.0]),  # bu molekülde yok
        "logP": torch.tensor([1.0])
    }

    masked_outputs = apply_task_mask(task_outputs, task_mask)

    print("Graph embedding h_G:")
    print(h_G)

    print("\nMasked task outputs:")
    for task, out in masked_outputs.items():
        print(task, out)

    visualize_graph(data)


if __name__ == "__main__":
    main()
