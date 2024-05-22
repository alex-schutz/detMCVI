#pragma once

// #include "CTP_graph.h"
#include "../statespace.h"
#include "ShortestPath.h"
#include "SimInterface.h"
#include "auto_generated_graph.h"

static bool CmpPair(const std::pair<std::pair<int64_t, int64_t>, double>& p1,
                    const std::pair<std::pair<int64_t, int64_t>, double>& p2) {
  return p1.second < p2.second;
}

class GraphPath : public ShortestPathFasterAlgorithm {
 private:
  std::unordered_map<std::pair<int64_t, int64_t>, double, pairhash>
      _edges;  // bidirectional, smallest node is first, double is weight

 public:
  GraphPath(
      std::unordered_map<std::pair<int64_t, int64_t>, double, pairhash> edges)
      : _edges(edges) {}

  std::vector<std::tuple<int64_t, double, int64_t>> getEdges(
      int64_t node) const override {
    std::vector<std::tuple<int64_t, double, int64_t>> out;
    for (const auto& e : _edges) {
      if (e.first.first == node)
        out.push_back({e.first.second, e.second, e.first.second});
      else if (e.first.second == node)
        out.push_back({e.first.first, e.second, e.first.first});
    }
    return out;
  }
};

class CTP : public MCVI::SimInterface {
 protected:
  std::mt19937_64& rng;
  std::vector<int64_t> nodes;
  std::unordered_map<std::pair<int64_t, int64_t>, double, pairhash>
      edges;  // bidirectional, smallest node is first, double is weight
  std::unordered_map<std::pair<int64_t, int64_t>, double, pairhash>
      stoch_edges;  // probability of being blocked
  int64_t goal;
  int64_t origin;
  StateSpace stateSpace;
  int64_t max_obs_width;
  std::vector<std::string> actions;
  std::vector<std::string> observations;
  double _idle_reward;
  double _bad_action_reward;
  mutable std::unordered_map<int64_t, bool> goal_reachable;

 public:
  CTP(std::mt19937_64& rng)
      : rng(rng),
        nodes(CTPNodes),
        edges(CTPEdges),
        stoch_edges(CTPStochEdges),
        goal(CTPGoal),
        origin(CTPOrigin),
        stateSpace(initStateSpace()),
        max_obs_width(initObsWidth()),
        actions(initActions()),
        observations(initObs()),
        _idle_reward(initIdleReward()),
        _bad_action_reward(initBadReward()) {}

  int64_t GetSizeOfObs() const override { return nodes.size() * max_obs_width; }
  int64_t GetSizeOfA() const override { return actions.size(); }
  double GetDiscount() const override { return 0.98; }
  int64_t GetNbAgent() const override { return 1; }
  const std::vector<std::string>& getActions() const { return actions; }
  const std::vector<std::string>& getObs() const { return observations; }
  int64_t getGoal() const { return goal; }
  bool IsTerminal(int64_t sI) const override {
    return stateSpace.getStateFactorElem(sI, "loc") == goal;
  }

  std::tuple<int64_t, int64_t, double, bool> Step(int64_t sI,
                                                  int64_t aI) override {
    int64_t sNext;
    const double reward = applyActionToState(sI, aI, sNext);
    const int64_t oI = observeState(sNext);
    const bool finished = checkFinished(sI, aI, sNext);
    // sI_next, oI, Reward, Done
    return std::tuple<int64_t, int64_t, double, bool>(sNext, oI, reward,
                                                      finished);
  }

  int64_t SampleStartState() override {
    std::uniform_real_distribution<> unif(0, 1);
    std::map<std::string, int64_t> state;
    // agent starts at special initial state (for init observation)
    state["loc"] = (int64_t)nodes.size();
    // stochastic edge status
    for (const auto& [edge, p] : stoch_edges)
      state[edge2str(edge)] = (unif(rng)) < p ? 0 : 1;
    return stateSpace.stateIndex(state);
  }

  void visualiseGraph(std::ostream& os) {
    if (!os) throw std::logic_error("Unable to write graph to file");

    os << "graph G {" << std::endl;

    for (const auto& i : nodes) {
      os << "  " << i << " [label=\"" << i << "\"";
      if (i == origin) os << ", fillcolor=\"#ff7f0e\", style=filled";
      if (i == goal) os << ", fillcolor=\"#2ca02c\", style=filled";
      os << "];" << std::endl;
    }

    for (const auto& [edge, weight] : edges) {
      auto stochEdge = stoch_edges.find(edge);
      if (stochEdge != stoch_edges.end()) {
        os << "  " << edge.first << " -- " << edge.second << " [label=\""
           << stochEdge->second << " : " << weight << "\", style=dashed];"
           << std::endl;
      } else {
        os << "  " << edge.first << " -- " << edge.second << " [label=\""
           << weight << "\"];" << std::endl;
      }
    }

    os << "}" << std::endl;
  }

 protected:
  std::string edge2str(std::pair<int64_t, int64_t> e) const {
    return "e" + std::to_string(e.first) + "_" + std::to_string(e.second);
  }

  // Return all stochastic edges adjacent to `node`
  std::vector<std::pair<int64_t, int64_t>> AdjacentStochEdges(
      int64_t node) const {
    std::vector<std::pair<int64_t, int64_t>> edges;
    for (const auto& [edge, p] : stoch_edges) {
      if (edge.first == node || edge.second == node) {
        edges.push_back(edge);
      }
    }

    auto compareEdges = [node](const std::pair<int64_t, int64_t>& edge1,
                               const std::pair<int64_t, int64_t>& edge2) {
      int64_t other1 = (edge1.first == node) ? edge1.second : edge1.first;
      int64_t other2 = (edge2.first == node) ? edge2.second : edge2.first;
      return other1 < other2;
    };

    std::sort(edges.begin(), edges.end(), compareEdges);

    return edges;
  }

 private:
  std::vector<std::string> initActions() const {
    std::vector<std::string> acts;
    for (const auto& n : nodes) acts.push_back(std::to_string(n));
    acts.push_back("decide_goal_unreachable");
    return acts;
  }

  StateSpace initStateSpace() const {
    std::map<std::string, std::vector<int64_t>> state_factors;
    // agent location
    state_factors["loc"] = nodes;
    state_factors["loc"].push_back(
        nodes.size());  // special state for init observation
    // stochastic edge status
    for (const auto& [edge, _] : stoch_edges)
      state_factors[edge2str(edge)] = {0, 1};  // 0 = blocked, 1 = unblocked
    StateSpace ss(state_factors);
    std::cout << "State space size: " << ss.size() << std::endl;
    return ss;
  }

  int64_t initObsWidth() const {
    size_t max_stoch_edges_at_node = 0;
    for (const auto& node : nodes)
      max_stoch_edges_at_node =
          std::max(max_stoch_edges_at_node, AdjacentStochEdges(node).size());

    return std::pow(2, max_stoch_edges_at_node);
  }

  std::vector<std::string> initObs() const {
    std::vector<std::string> obs;  // Observation space can be very large!
    return obs;
  }

  bool nodesAdjacent(int64_t a, int64_t b, int64_t state) const {
    if (a == (int64_t)nodes.size() || b == (int64_t)nodes.size()) return false;
    if (a == b) return true;
    const auto edge = a < b ? std::pair(a, b) : std::pair(b, a);
    if (edges.find(edge) == edges.end()) return false;  // edge does not exist

    // check if edge is stochastic
    const auto stoch_ptr = stoch_edges.find(edge);
    if (stoch_ptr == stoch_edges.end()) return true;  // deterministic edge

    // check if edge is unblocked
    return stateSpace.getStateFactorElem(state, edge2str(edge)) ==
           1;  // traversable
  }

  double applyActionToState(int64_t state, int64_t action,
                            int64_t& sNext) const {
    sNext = state;
    const int64_t loc = stateSpace.getStateFactorElem(state, "loc");
    if (loc == (int64_t)nodes.size()) {  // special initial state
      sNext = stateSpace.updateStateFactor(state, "loc", origin);
      return 0;
    }
    if (loc == goal) return 0;  // goal is absorbing

    if (loc == action) return _idle_reward;  // idling

    if (actions.at(action) == "decide_goal_unreachable")
      return goalUnreachable(state) ? 0 : _bad_action_reward;

    // invalid move
    if (!nodesAdjacent(loc, action, state)) return _bad_action_reward;

    // moving
    sNext = stateSpace.updateStateFactor(state, "loc", action);
    return action < loc ? -edges.at({action, loc}) : -edges.at({loc, action});
  }

  int64_t observeState(int64_t state) const {
    int64_t observation = 0;

    int64_t loc = stateSpace.getStateFactorElem(state, "loc");
    if (loc == (int64_t)nodes.size())
      loc = origin;  // observe initial state as if at origin

    // stochastic edge status
    int64_t n = 0;
    for (const auto& edge : AdjacentStochEdges(loc)) {
      if (stateSpace.getStateFactorElem(state, edge2str(edge)))
        observation |= ((int64_t)1 << n);
      ++n;
    }

    observation += loc * max_obs_width;

    return observation;
  }

  bool checkFinished(int64_t sI, int64_t aI, int64_t sNext) const {
    if (stateSpace.getStateFactorElem(sI, "loc") == (int64_t)nodes.size())
      return false;
    if (actions.at(aI) == "decide_goal_unreachable" && goalUnreachable(sI))
      return true;
    return stateSpace.getStateFactorElem(sNext, "loc") == goal;
  }

  bool goalUnreachable(int64_t state) const {
    // check if goal is reachable from the origin (cached)
    const int64_t origin_state =
        stateSpace.updateStateFactor(state, "loc", origin);
    const auto ret = goal_reachable.find(origin_state);
    if (ret != goal_reachable.end()) return !ret->second;

    // find shortest path to goal, return true if none exists
    std::unordered_map<std::pair<int64_t, int64_t>, double, pairhash>
        state_edges;
    for (const auto& e : edges) {
      if (!stoch_edges.contains(e.first) ||
          stateSpace.getStateFactorElem(state, edge2str(e.first)) == 1)
        state_edges.insert(e);
    }
    auto gp = GraphPath(state_edges);
    const auto [costs, pred] = gp.calculate(origin, state_edges.size() + 1);
    const bool reaches_goal = costs.contains(goal);
    goal_reachable[origin_state] = reaches_goal;

    return !reaches_goal;
  }

  double initIdleReward() const {
    const double max_edge =
        std::max_element(edges.begin(), edges.end(), CmpPair)->second;
    return -5 * max_edge;
  }

  double initBadReward() const {
    const double max_edge =
        std::max_element(edges.begin(), edges.end(), CmpPair)->second;
    return -50 * max_edge;
  }
};
