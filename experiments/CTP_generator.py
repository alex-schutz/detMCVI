#!/usr/bin/python3

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay


def plot_nx_graph(G: nx.Graph, origin, goal):
    # Plot graph
    node_colour = []
    for node in G.nodes:
        c = "tab:purple"
        if node == goal:
            c = "lightgreen"
        elif node == origin:
            c = "orange"
        node_colour.append(c)
    edge_labels = []
    probs = nx.get_edge_attributes(G, "blocked_prob")
    weights = nx.get_edge_attributes(G, "weight")
    edge_labels = {
        e: (f"p: {probs[e]}\n{w}" if e in probs else f"{w}") for e, w in weights.items()
    }
    edge_colour = ["blue" if edge in probs.keys() else "black" for edge in G.edges]
    pos = nx.get_node_attributes(G, "pos")
    nx.draw(
        G,
        with_labels=True,
        node_size=500,
        node_color=node_colour,
        edge_color=edge_colour,
        pos=pos,
    )
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels, font_color="blue")
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    plt.show()


def distance(e, nodes):
    dx = nodes[e[0]][0] - nodes[e[1]][0]
    dy = nodes[e[0]][1] - nodes[e[1]][1]
    return np.sqrt(dx * dx + dy * dy)


# Define world parameters
xmax = 10  # max x-grid coordinate
ymax = 10  # max y-grid coordinate
V = 8  # number of nodes
prop_stoch = 0.5  # proportion of stochastic edges in the graph
use_edge_weights = True

# Fix seed for graph generation
np.random.seed(42)

# Generate random points in a grid
node_pos = np.random.choice(xmax * ymax, V, replace=False)
grid_nodes = [(i // (ymax + 1), i % (ymax + 1)) for i in node_pos]

# Define origin and goal nodes
origin = grid_nodes.index(min(grid_nodes, key=lambda point: point[0] + point[1]))
goal = grid_nodes.index(max(grid_nodes, key=lambda point: point[0] + point[1]))

# Apply Delaunay triangulation
delaunay = Delaunay(grid_nodes)
simplices = delaunay.simplices

# Add to networkx object
G = nx.Graph()
for i, node in enumerate(grid_nodes):
    G.add_node(i, pos=node)
for path in simplices:
    nx.add_path(G, path)
weights = {}
for e in G.edges():
    weights[e] = round(distance(e, grid_nodes), 2) if use_edge_weights else 1
nx.set_edge_attributes(G, values=weights, name="weight")

# Define stochastic edges and probabilities
num_stoch_edges = round(prop_stoch * len(G.edges))
stoch_edge_idx = np.random.choice(len(G.edges), num_stoch_edges, replace=False)
edge_probs: dict[tuple[int, int], float] = {}
for i, e in enumerate(G.edges):
    if i in stoch_edge_idx:
        edge_probs[e] = round(np.random.uniform(), 2)
nx.set_edge_attributes(G, edge_probs, "blocked_prob")

# Add deterministic high cost edge from origin to goal
G.add_edge(origin, goal, weight=1000)


print(
    "const std::vector<int> CTPNodes = {", ", ".join(map(str, range(V))), "};", sep=""
)
edge_cpp = [
    f"{{{{{min(e)}, {max(e)}}}, {w}}}"
    for e, w in nx.get_edge_attributes(G, "weight").items()
]
print(
    "const std::unordered_map<std::pair<int, int>, double, pairhash> CTPEdges = {",
    ", ".join(edge_cpp),
    "};",
    sep="",
)
stoch_edge_cpp = [
    f"{{{{{min(e)}, {max(e)}}}, {w}}}"
    for e, w in nx.get_edge_attributes(G, "blocked_prob").items()
]
print(
    "const std::unordered_map<std::pair<int, int>, double, pairhash> CTPStochEdges = {",
    ", ".join(stoch_edge_cpp),
    "};",
    sep="",
)
print(f"const int CTPOrigin = {origin};")
print(f"const int CTPGoal = {goal};")

plot_nx_graph(G, origin, goal)
