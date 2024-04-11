#include "Bound.h"

#include <algorithm>
#include <cassert>
#include <limits>

namespace MCVI {

static bool CmpPair(const std::pair<int64_t, double>& p1,
                    const std::pair<int64_t, double>& p2) {
  return p1.second < p2.second;
}

std::tuple<int64_t, double> UpperBoundEvaluation(
    const BeliefDistribution& belief, const PathToTerminal& solver,
    int64_t max_depth) {
  double V_upper_bound = 0.0;
  std::unordered_map<int64_t, double> action_vals;
  for (const auto& [state, prob] : belief) {
    const auto [action, reward] = solver.path(state, max_depth);
    V_upper_bound += prob * reward;
    action_vals[action] += prob * reward;
  }
  const int64_t best_action =
      std::max_element(std::begin(action_vals), std::end(action_vals), CmpPair)
          ->first;

  return {best_action, V_upper_bound};
}

double FindRLower(SimInterface* pomdp, const BeliefDistribution& b0,
                  int64_t num_actions, double epsilon, int64_t max_depth) {
  std::unordered_map<int64_t, double> action_min_reward;
  for (int64_t action = 0; action < num_actions; ++action) {
    double min_reward = std::numeric_limits<double>::infinity();
    for (const auto& [s, prob] : b0) {
      int64_t state = s;
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

std::vector<std::tuple<int64_t, double, int64_t>> PathToTerminal::getEdges(
    int64_t state) const {
  std::vector<std::tuple<int64_t, double, int64_t>> edges;
  for (int64_t a = 0; a < pomdp->GetSizeOfA(); ++a) {
    const auto& [sNext, o, reward, done] = pomdp->Step(state, a);
    edges.push_back({sNext, -reward, a});
    if (done) terminalStates.insert(sNext);
  }
  if (terminalStates.contains(state)) edges.push_back({-1, 0, -1});
  return edges;
}

std::tuple<int64_t, double> PathToTerminal::path(int64_t source,
                                                 int64_t max_depth) const {
  auto path = paths.find(source);
  if (path == paths.end()) {
    // NB: shortest path calculation does not account for discount
    const auto [costs, pred] = calculate(source, max_depth);
    const auto best_sink =
        costs.contains(-1)
            ? -1
            : std::max_element(costs.cbegin(), costs.cend(),
                               [](const std::pair<int64_t, double>& p1,
                                  const std::pair<int64_t, double>& p2) {
                                 return p1.second < p2.second;
                               })
                  ->first;
    const std::vector<std::pair<int64_t, int64_t>> p =
        reconstructPath(best_sink, pred);
    paths[source] = p;
    path = paths.find(source);
  }

  const double gamma = pomdp->GetDiscount();
  double sum_reward = 0.0;
  double discount = 1.0;
  for (const auto& [state, action] : path->second) {
    const auto [sNext, obs, reward, done] = pomdp->Step(state, action);
    sum_reward += discount * reward;
    discount *= gamma;
  }
  return {path->second[0].second, sum_reward};
}

std::unordered_map<int64_t, std::shared_ptr<PathToTerminal::PathNode>>
PathToTerminal::buildPathTree() const {
  std::shared_ptr<PathNode> rootNode = createPathNode(-1, {});  // dummy root
  std::unordered_map<int64_t, std::shared_ptr<PathNode>> startNodes;

  for (const auto& [source, path] : paths) {
    std::shared_ptr<PathNode> nextNode = rootNode;

    for (auto it = path.rbegin(); it != path.rend(); ++it) {
      int64_t action = it->second;
      int64_t state = it->first;
      if (it == path.rbegin()) {
        assert(action == -1);
        assert(state == -1);
        continue;
      }

      std::shared_ptr<PathNode> currentNode =
          findOrCreateNode(nextNode, action);
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
    int64_t action, const std::unordered_set<int64_t>& states) const {
  return std::make_shared<PathNode>(PathNode{action, states, nullptr, {}});
}

std::shared_ptr<PathToTerminal::PathNode> PathToTerminal::findActionChild(
    std::shared_ptr<PathNode> node, int64_t action) const {
  for (const auto& n : node->prevNodes)
    if (n->action == action) return n;
  return nullptr;
}

std::shared_ptr<PathToTerminal::PathNode> PathToTerminal::findOrCreateNode(
    std::shared_ptr<PathNode> nextNode, int64_t action) const {
  const auto node = findActionChild(nextNode, action);
  if (!node) {
    std::shared_ptr<PathNode> newNode = createPathNode(action, {});
    newNode->nextNode = nextNode;
    nextNode->prevNodes.push_back(newNode);
    return newNode;
  }
  return node;
}

}  // namespace MCVI
