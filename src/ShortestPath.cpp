#include "ShortestPath.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <queue>
#include <stdexcept>

namespace MCVI {

std::vector<std::pair<State, int64_t>>
ShortestPathFasterAlgorithm::reconstructPath(
    const State& target,
    const StateMap<std::pair<State, int64_t>>& pred) const {
  std::vector<std::pair<State, int64_t>> path;
  std::pair<State, int64_t> current = {target, -1};
  while (true) {
    path.push_back(current);
    const auto currentPtr = pred.find(current.first);
    if (currentPtr == pred.cend()) break;
    current = currentPtr->second;
  }

  std::reverse(path.begin(), path.end());
  return path;
}

void ShortestPathFasterAlgorithm::initParams() const {
  d = InfMap(StateMap<double>({}));
  inQueue = {};
  depth = {};
  predecessor = {};
}

std::tuple<StateMap<double>, StateMap<std::pair<State, int64_t>>>
ShortestPathFasterAlgorithm::calculate(const State& source, int64_t N) const {
  initParams();

  std::deque<State> q;
  q.push_back(source);
  d[source] = 0.0;
  inQueue[source] = true;
  depth[source] = 0;

  while (!q.empty()) {
    const auto u = q.front();
    q.pop_front();
    inQueue[u] = false;

    // Process each edge
    for (auto [v, w, label] : getEdges(u)) {
      if (v != u) depth[v] = std::max(depth[v], depth[u] + 1);
      if (depth[v] > N) return {d.map(), predecessor};  // max depth reached
      if (d[u] + w < d[v]) {
        d[v] = d[u] + w;
        predecessor[v] = {u, label};

        // Add v to queue
        if (!inQueue[v]) {
          if (q.empty() || d[v] < d[q.front()])
            q.push_front(v);
          else
            q.push_back(v);
          inQueue[v] = true;
        }
      }
    }
  }
  return {d.map(), predecessor};
}

static bool CmpPair(const std::pair<State, double>& p1,
                    const std::pair<State, double>& p2) {
  return p1.second < p2.second;
}

std::pair<double, std::vector<std::pair<int64_t, State>>>
MaximiseReward::getMaxReward(const State& init_state, int64_t max_depth,
                             double discount_factor) const {
  std::unordered_map<State, double, StateHash, StateEqual> rewards;
  std::unordered_map<State, std::vector<std::pair<int64_t, State>>, StateHash,
                     StateEqual>
      paths;
  rewards[init_state] = 0;

  for (int64_t depth = 0; depth < max_depth; ++depth) {
    std::unordered_map<State, double, StateHash, StateEqual> next_rewards;
    std::unordered_map<State, std::vector<std::pair<int64_t, State>>, StateHash,
                       StateEqual>
        next_paths;
    bool all_terminal = true;

    for (const auto& [state, rw] : rewards) {
      for (const auto& [action, next_state, immediate_rw, state_terminal] :
           getSuccessors(state)) {
        if (state_terminal) {
          if (next_rewards.find(state) == next_rewards.end() ||
              rw > next_rewards[state]) {
            next_rewards[state] = rw;
            next_paths[state] = paths[state];  // Path remains the same
          }
        } else {
          all_terminal = false;
          const double new_reward =
              rw + std::pow(discount_factor, depth) * immediate_rw;
          if (next_rewards.find(next_state) == next_rewards.end() ||
              new_reward > next_rewards[next_state]) {
            next_rewards[next_state] = new_reward;
            next_paths[next_state] = paths[state];
            next_paths[next_state].push_back({action, next_state});
          }
        }
      }
    }
    rewards = next_rewards;
    paths = next_paths;
    if (all_terminal) break;
  }

  const auto best_final_state_ptr =
      std::max_element(rewards.begin(), rewards.end(), CmpPair);
  if (best_final_state_ptr == rewards.end())
    throw std::logic_error("Could not find maximal path");
  return std::make_pair(best_final_state_ptr->second,
                        paths.at(best_final_state_ptr->first));
}

}  // namespace MCVI
