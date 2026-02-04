import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx

def visualize_graph(data, title="Molecule Graph"):
    G = to_networkx(data, to_undirected=True)

    plt.figure(figsize=(4, 4))
    nx.draw(
        G,
        with_labels=True,
        node_size=600,
        node_color="lightblue",
        edge_color="gray"
    )
    plt.title(title)
    plt.show()
