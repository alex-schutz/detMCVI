#include "Bound.h"

#include <algorithm>
#include <cassert>
#include <limits>

namespace MCVI {

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
                  double epsilon, int64_t max_depth) {
  const int64_t default_action = 0;
  const double gamma = pomdp->GetDiscount();
  double belief_value = 0.0;

  for (const auto& [s, prob] : b0) {
    double sum_reward = 0;
    State state = s;
    int64_t step = 0;
    while ((step < max_depth) && (std::pow(gamma, step) > epsilon)) {
      const auto [sNext, obs, reward, done] =
          pomdp->Step(state, default_action);
      sum_reward += std::pow(gamma, step) * reward;
      if (done) break;
      state = sNext;
      ++step;
    }
    belief_value += sum_reward * prob;
  }
  return belief_value;
}

std::vector<std::tuple<State, double, int64_t>> PathToTerminal::getEdges(
    const State& state) const {
  if (terminalStates.contains(state)) return {{{}, 0, -1}};
  std::vector<std::tuple<State, double, int64_t>> edges;
  for (int64_t a = 0; a < pomdp->GetSizeOfA(); ++a) {
    const auto& [sNext, o, reward, done] = pomdp->Step(state, a);
    edges.push_back({sNext, -reward, a});
    if (done) terminalStates.insert(sNext);
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

double CalculateUpperBound(const BeliefDistribution& belief,
                           int64_t belief_depth, int64_t eval_depth,
                           const PathToTerminal& heuristic, SimInterface* sim) {
  const auto H_Uval = sim->GetHeuristicUpper(belief, eval_depth - belief_depth);
  if (H_Uval.has_value())
    return std::pow(sim->GetDiscount(), belief_depth) * H_Uval.value();
  return UpperBoundEvaluation(belief, heuristic, sim->GetDiscount(),
                              belief_depth, eval_depth);
}

double CalculateLowerBound(const BeliefDistribution& belief,
                           int64_t belief_depth, int64_t eval_depth,
                           const BoundFunction& func, SimInterface* sim) {
  const auto H_Lval = sim->GetHeuristicLower(belief, eval_depth);
  if (H_Lval.has_value())
    return std::pow(sim->GetDiscount(), belief_depth) * H_Lval.value();
  return func(belief, belief_depth, eval_depth, sim);
}

}  // namespace MCVI
