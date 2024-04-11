#include "ShortestPath.h"

#include <algorithm>
#include <iostream>
#include <queue>
#include <stdexcept>

namespace MCVI {

std::vector<std::pair<int64_t, int64_t>>
ShortestPathFasterAlgorithm::reconstructPath(
    int64_t target,
    const std::unordered_map<int64_t, std::pair<int64_t, int64_t>>& pred)
    const {
  std::vector<std::pair<int64_t, int64_t>> path;
  std::pair<int64_t, int64_t> current = {target, -1};
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
  d = {};
  inQueue = {};
  depth = {};
  predecessor = {};
}

std::tuple<std::unordered_map<int64_t, double>,
           std::unordered_map<int64_t, std::pair<int64_t, int64_t>>>
ShortestPathFasterAlgorithm::calculate(int64_t source, int64_t N) const {
  initParams();

  std::deque<int64_t> q;
  q.push_back(source);
  d[source] = 0.0;
  inQueue[source] = true;
  depth[source] = 0;

  while (!q.empty()) {
    int64_t u = q.front();
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

}  // namespace MCVI
