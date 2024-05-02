import heapq
import math


class Node:
    def __init__(self, state, g_value, h_value):
        self.state = state
        self.g_value = g_value
        self.h_value = h_value

    def __lt__(self, other):
        return (self.g_value + self.h_value) < (other.g_value + other.h_value)


def ao_star(mdp, epsilon=1e-6):
    S, A, P, R, gamma = mdp
    open_list = []
    closed_list = {}

    def compute_heuristic(state):
        # Compute heuristic function h(s) for state s (e.g., using value iteration)
        # For simplicity, let's use a trivial heuristic for this example
        return 0

    def expand(node):
        state = node.state
        for action in A:
            for next_state in S:
                transition_prob = P(next_state, state, action)
                if transition_prob > 0:
                    next_g_value = node.g_value + gamma * transition_prob * R(
                        state, action, next_state
                    )
                    next_h_value = compute_heuristic(next_state)
                    next_node = Node(next_state, next_g_value, next_h_value)
                    yield next_node

    initial_node = Node(initial_state, 0, compute_heuristic(initial_state))
    heapq.heappush(open_list, initial_node)

    while open_list:
        current_node = heapq.heappop(open_list)

        if current_node.state in closed_list:
            continue

        if current_node.g_value - closed_list.get(current_node.state, 0) < epsilon:
            return current_node.state

        closed_list[current_node.state] = current_node.g_value

        for successor in expand(current_node):
            heapq.heappush(open_list, successor)

    return None  # No solution found


# Example usage:
# Define your MDP as a tuple (S, A, P, R, gamma)
# Then call ao_star(mdp) to solve it


def lao_star(mdp):
    V_l = h
    fringe_set = set(s0)
    interior_set = set()
    curr_graph = interior_set.union(fringe_set)
    solution_graph = set(s0)
    while has_non_goal_state(fringe_set.intersection(solution_graph)):
        # Expand a fringe state of the best partial policy
        s = choose_non_goal_state(fringe_set.intersection(solution_graph))
        fringe_set.remove(s)
        successors = generate_successors(s)
        new_successors = {s for s in successors if s not in interior_set}
        fringe_set = fringe_set.union(new_successors)
        interior_set = interior_set.union(new_successors)
        curr_graph = interior_set.union(fringe_set)
        # Update state values and mark greedy actions
        # Z = {s and all states in curr_graph from which s can be reached via greedy best actions}
        # AO*: back up values from the bottom up -> calculate successor values, go back up the graph
        # rebuild solution_graph based on best actions in graph


class Node:
    def __init__(self, belief: int) -> None:
        self.belief = belief
        self.value = heuristic(belief)
        self.successors: dict[int, Node] = {}
        self.best_action = None

    def expand(self) -> None:
        # TODO: iterate through actions and add successor states

        # TODO: check for previous states to close loops (LAO* only)
        pass

    def back_up_from_successors(self) -> None:
        if not self.successors:
            return
        best_action = None
        best_val = -math.inf
        for action, s in self.successors.items():
            s.back_up_from_successors()
            R = reward(self.state, action)
            if s.value + R > best_val:
                best_val = s.value + R
                best_action = action
        self.value = best_val
        self.best_action = best_action


class Graph:
    def __init__(self, node_edge_list: list[tuple[Node, int | None]]) -> None:
        self.nodes = []
        self.edges = {}
        for node, edge in node_edge_list:
            self.nodes += [node]
            self.edges.update({node: edge})


def ao_star2():
    initial_node = Node(initial_belief, compute_heuristic(initial_belief))
    fringe_set = {initial_node}

    solution_graph = Graph([(initial_node, None)])

    curr_soln_fringe = [
        s for s in fringe_set if s in solution_graph.nodes and not is_goal_state(s)
    ]
    while curr_soln_fringe:
        s = curr_soln_fringe[-1]
        fringe_set.remove(s)

        s.expand()
        fringe_set = fringe_set.union(
            {suc for suc in s.successors if not is_goal_state(suc)}
        )  # TODO LAO*: if not already in graph

        # back up nodes in the tree
        initial_node.back_up_from_successors()

        solution_graph = build_greedy_graph(initial_node)
        curr_soln_fringe = [
            s for s in fringe_set if s in solution_graph.nodes and not is_goal_state(s)
        ]
    return solution_graph.edges
