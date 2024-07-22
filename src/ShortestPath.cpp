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

std::pair<double, std::vector<std::tuple<int64_t, State, double>>>
MaximiseReward::getMaxReward(const State& init_state, int64_t max_depth) const {
  return GetCachedOrSearch(init_state, max_depth);
}

static bool CmpPathValue(
    const std::pair<double, std::vector<std::tuple<int64_t, State, double>>>&
        p1,
    const std::pair<double, std::vector<std::tuple<int64_t, State, double>>>&
        p2) {
  return p1.first < p2.first;
}

std::pair<double, std::vector<std::tuple<int64_t, State, double>>>
MaximiseReward::GetCachedOrSearch(const State& state,
                                  int64_t depth_to_go) const {
  if (depth_to_go <= 0) return {0, {}};
  const auto state_cache = cache.find(state);
  if (state_cache == cache.end()) return Search(state, depth_to_go);

  const auto pos = state_cache->second.find(depth_to_go);
  if (pos == state_cache->second.end()) return Search(state, depth_to_go);

  // follow the path to reconstruct
  const auto& [action, successor, immediate_rw] = pos->second;
  const auto& [next_reward, next_path] =
      GetCachedOrSearch(successor, depth_to_go - 1);
  const double total_rw = immediate_rw + discount_factor * next_reward;
  auto new_path = next_path;
  new_path.insert(new_path.begin(),
                  std::make_tuple(action, successor, immediate_rw));

  return {total_rw, new_path};
}

std::pair<double, std::vector<std::tuple<int64_t, State, double>>>
MaximiseReward::Search(const State& state, int64_t depth_to_go) const {
  if (depth_to_go <= 0) return {0, {}};

  std::vector<
      std::pair<double, std::vector<std::tuple<int64_t, State, double>>>>
      paths;
  for (const auto& [action, successor, immediate_rw, state_terminal] :
       getSuccessors(state)) {
    if (state_terminal) return {0, {}};

    const auto& [next_reward, next_path] =
        GetCachedOrSearch(successor, depth_to_go - 1);
    const double total_rw = immediate_rw + discount_factor * next_reward;
    auto new_path = next_path;
    new_path.insert(new_path.begin(),
                    std::make_tuple(action, successor, immediate_rw));
    paths.push_back({total_rw, new_path});
  }

  const auto best_path =
      std::max_element(paths.begin(), paths.end(), CmpPathValue);
  if (best_path == paths.end())
    throw std::runtime_error("Failed to find best path for remaining depth " +
                             std::to_string(depth_to_go));
  cache[state][depth_to_go] = best_path->second.at(0);
  return *best_path;
}

}  // namespace MCVI
