#!/usr/bin/python3

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay
import sys

PREAMBLE = """#pragma once
#include <unordered_map>
#include <vector>

struct pairhash {
 public:
  template <typename T, typename U>
  std::size_t operator()(const std::pair<T, U>& x) const {
    return std::hash<T>()(x.first) ^ std::hash<U>()(x.second);
  }
};
"""


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


def distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return np.sqrt(dx * dx + dy * dy)


def edge_distance(e, nodes):
    return distance(nodes[e[0]], nodes[e[1]])


def generate_graph(
    n_nodes: int,
    seed: int,
    use_edge_weights=False,
    prop_stoch=0.4,
    plot=False,
    grid_size=10,
) -> tuple[nx.Graph, int, int, bool]:
    # Define world parameters
    xmax = grid_size  # max x-grid coordinate
    ymax = grid_size  # max y-grid coordinate

    # Fix seed for graph generation
    np.random.seed(seed)

    # Generate random points in a grid
    node_pos = np.random.choice(xmax * ymax, n_nodes, replace=False)
    grid_nodes = [(i // (ymax + 1), i % (ymax + 1)) for i in node_pos]

    # Define origin and goal nodes based on furthest apart nodes
    goal = int(
        np.argmax(
            [distance(grid_nodes[0], grid_nodes[i]) for i in range(len(grid_nodes))]
        )
    )
    origin = int(
        np.argmax(
            [distance(grid_nodes[goal], grid_nodes[i]) for i in range(len(grid_nodes))]
        )
    )

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
        weights[e] = round(edge_distance(e, grid_nodes), 2) if use_edge_weights else 1
    nx.set_edge_attributes(G, values=weights, name="weight")

    # Define stochastic edges and probabilities
    num_stoch_edges = round(prop_stoch * len(G.edges))
    stoch_edge_idx = np.random.choice(len(G.edges), num_stoch_edges, replace=False)
    edge_probs: dict[tuple[int, int], float] = {}
    for i, e in enumerate(G.edges):
        if i in stoch_edge_idx:
            edge_probs[e] = round(np.random.uniform(), 2)
    nx.set_edge_attributes(G, edge_probs, "blocked_prob")

    # basic solvability check
    solvable = nx.has_path(G, origin, goal)

    if plot:
        plot_nx_graph(G, origin, goal)

    return G, origin, goal, solvable


def graph_to_cpp(G: nx.Graph, origin, goal, file):
    print(PREAMBLE, file=file)

    print(
        "const std::vector<int64_t> CTPNodes = {",
        ", ".join(map(str, range(len(G.nodes)))),
        "};",
        sep="",
        file=file,
    )
    edge_cpp = [
        f"{{{{{min(e)}, {max(e)}}}, {w}}}"
        for e, w in nx.get_edge_attributes(G, "weight").items()
    ]
    print(
        "const std::unordered_map<std::pair<int64_t, int64_t>, double, pairhash> CTPEdges = {",
        ", ".join(edge_cpp),
        "};",
        sep="",
        file=file,
    )
    stoch_edge_cpp = [
        f"{{{{{min(e)}, {max(e)}}}, {w}}}"
        for e, w in nx.get_edge_attributes(G, "blocked_prob").items()
    ]
    print(
        "const std::unordered_map<std::pair<int64_t, int64_t>, double, pairhash> CTPStochEdges = {",
        ", ".join(stoch_edge_cpp),
        "};",
        sep="",
        file=file,
    )
    print(f"const int64_t CTPOrigin = {origin};", file=file)
    print(f"const int64_t CTPGoal = {goal};", file=file)


def generate_delaunay_graph_set(location_count: int, set_size: int, seed: int):
    """
    Delaunay graphs with 100 vertices whose coordinates are randomly chosen over the region [1, 100] x [1, 100] on the plane.
    Edge lengths are set to the Euclidean distance between their end vertices and the two farthest vertices of the graph are designated as the starting and termination vertices, respectively.
    Each grid edge has a 0.25 probability of being stochastic and marks of stochastic edges are sampled from the uniform distribution. (Aksakalli et al. 2016)
    """
    problem_set = []
    for n in range(set_size):
        while True:
            G, origin, goal, solvable = generate_graph(
                location_count, seed, True, 0.25, False, 100
            )
            if solvable:
                problem_set.append((G, origin, goal, seed))
                break
            else:
                seed += 1
        seed += 1
    return problem_set


# def generate_grid_graph_set(grid_size: int):
#     """
#     Grid graphs where imax = jmax = 10.
#     The starting and termination vertices are taken as s = (5, 10) and t = (5, 1).
#     As in Delaunay graphs, each edge has a 0.25 probability of being stochastic with uniform marks. (Aksakalli et al. 2016)
#     """


if __name__ == "__main__":
    seed = np.random.randint(0, 9999999)
    N = 15
    G, origin, goal, solvable = generate_graph(N, seed, True, plot=True)
    print("seed:", seed, "nodes:", N, "solvable:", solvable)
    graph_to_cpp(G, origin, goal, sys.stdout)
