#include "Bound.h"

#include <algorithm>
#include <cassert>
#include <limits>

namespace MCVI {

static bool CmpPair(const std::pair<int64_t, double>& p1,
                    const std::pair<int64_t, double>& p2) {
  return p1.second < p2.second;
}

double UpperBoundEvaluation(const BeliefDistribution& belief,
                            const PathToTerminal& solver, double gamma,
                            int64_t belief_depth, int64_t max_depth) {
  double V_upper_bound = 0.0;
  for (const auto& [state, prob] : belief) {
    const auto [action, reward] = solver.path(state, max_depth);
    V_upper_bound += std::pow(gamma, belief_depth) * reward * prob;
  }

  return V_upper_bound;
}

double FindRLower(SimInterface* pomdp, const BeliefDistribution& b0,
                  int64_t num_actions, double epsilon, int64_t max_depth) {
  std::unordered_map<int64_t, double> action_min_reward;
  for (int64_t action = 0; action < num_actions; ++action) {
    double min_reward = std::numeric_limits<double>::infinity();
    for (const auto& [s, prob] : b0) {
      State state = s;
      int64_t step = 0;
      while ((step < max_depth) &&
             (std::pow(pomdp->GetDiscount(), step) > epsilon)) {
        const auto [sNext, obs, reward, done] = pomdp->Step(state, action);
        if (reward < min_reward) {
          action_min_reward[action] = reward;
          min_reward = reward;
        }
        if (done) break;
        state = sNext;
        ++step;
      }
    }
  }
  const double max_min_reward =
      std::max_element(std::begin(action_min_reward),
                       std::end(action_min_reward), CmpPair)
          ->second;
  return max_min_reward / (1 - pomdp->GetDiscount());
}

std::vector<std::tuple<State, double, int64_t>> PathToTerminal::getEdges(
    const State& state) const {
  if (terminalStates.contains(state)) return {{{}, 0, -1}};
  std::vector<std::tuple<State, double, int64_t>> edges;
  for (int64_t a = 0; a < pomdp->GetSizeOfA(); ++a) {
    const auto& [sNext, o, reward, done] = pomdp->Step(state, a);
    edges.push_back({sNext, -reward, a});
    if (done) {
      terminalStates.insert(sNext);
      return {{{}, 0, -1}};
    }
  }
  return edges;
}

std::tuple<int64_t, double> PathToTerminal::path(const State& source,
                                                 int64_t max_depth) const {
  if (!paths.contains(source)) {
    // NB: shortest path calculation does not account for discount
    const auto [costs, pred] = calculate(source, max_depth);
    const auto best_sink =
        costs.contains({})
            ? State()
            : std::max_element(costs.cbegin(), costs.cend(),
                               [](const std::pair<State, double>& p1,
                                  const std::pair<State, double>& p2) {
                                 return p1.second < p2.second;
                               })
                  ->first;
    const std::vector<std::pair<State, int64_t>> p =
        reconstructPath(best_sink, pred);
    for (size_t i = 0; i < p.size() - 1; ++i) {
      paths[p.at(i).first] = {p.at(i).second, p.at(i + 1).first};
    }
    paths[p.back().first] = {p.back().second, {}};
  }

  const double gamma = pomdp->GetDiscount();
  double sum_reward = 0.0;
  double discount = 1.0;
  State state = source;
  while (true) {
    const int64_t action = paths.at(state).first;
    if (action == -1) break;
    const auto [sNext, obs, reward, done] = pomdp->Step(state, action);
    sum_reward += discount * reward;
    discount *= gamma;
    state = sNext;
  }
  return {paths.at(source).first, sum_reward};
}

StateMap<std::shared_ptr<PathToTerminal::PathNode>>
PathToTerminal::buildPathTree() const {
  int64_t id = -1;
  std::shared_ptr<PathNode> rootNode =
      createPathNode(id, -1, {});  // dummy root
  StateMap<std::shared_ptr<PathNode>> startNodes;

  for (const auto& [source, _] : paths) {
    std::vector<std::pair<State, int64_t>> path;
    State state = source;
    while (true) {
      const auto p = paths.at(state);
      path.push_back({state, p.first});
      state = p.second;
      if (p.first == -1 || p.second.size() == 0) break;
    }
    std::shared_ptr<PathNode> nextNode = rootNode;

    for (auto it = path.rbegin(); it != path.rend(); ++it) {
      const int64_t action = it->second;
      const State& state = it->first;
      if (it == path.rbegin()) {
        assert(action == -1);
        assert(state.size() == 0);
        continue;
      }

      std::shared_ptr<PathNode> currentNode =
          findOrCreateNode(nextNode, id, action);
      currentNode->states.insert(state);
      nextNode = currentNode;
    }
    startNodes[source] = nextNode;
  }

  // disconnect the dummy root node
  for (const auto& child : rootNode->prevNodes) child->nextNode = nullptr;

  return startNodes;
}

std::shared_ptr<PathToTerminal::PathNode> PathToTerminal::createPathNode(
    int64_t& id, int64_t action,
    const std::unordered_set<State, StateHash, StateEqual>& states) const {
  return std::make_shared<PathNode>(
      PathNode{id++, action, states, nullptr, {}});
}

std::shared_ptr<PathToTerminal::PathNode> PathToTerminal::findActionChild(
    std::shared_ptr<PathNode> node, int64_t action) const {
  for (const auto& n : node->prevNodes)
    if (n->action == action) return n;
  return nullptr;
}

std::shared_ptr<PathToTerminal::PathNode> PathToTerminal::findOrCreateNode(
    std::shared_ptr<PathNode> nextNode, int64_t& id, int64_t action) const {
  const auto node = findActionChild(nextNode, action);
  if (!node) {
    std::shared_ptr<PathNode> newNode = createPathNode(id, action, {});
    newNode->nextNode = nextNode;
    nextNode->prevNodes.push_back(newNode);
    return newNode;
  }
  return node;
}

bool PathToTerminal::is_terminal(const State& source, int64_t max_depth) const {
  if (!paths.contains(source)) path(source, max_depth);
  State state = source;
  while (true) {
    if (terminalStates.contains(state)) return true;
    if (!paths.contains(state)) return false;
    if (paths.at(state).first == -1) return false;
    state = paths.at(state).second;
  }
}

}  // namespace MCVI
