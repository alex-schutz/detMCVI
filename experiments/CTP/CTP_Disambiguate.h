#pragma once

#include <algorithm>
#include <cassert>
#include <random>

#include "CTP.h"
#include "Cache.h"
#include "ShortestPath.h"
#include "SimInterface.h"

#define USE_HEURISTIC_BOUNDS 0

class CTP_Disambiguate : public MCVI::SimInterface {
 protected:
  std::mt19937_64& rng;
  std::vector<int64_t> nodes;
  std::unordered_map<std::pair<int64_t, int64_t>, double, pairhash>
      edges;  // bidirectional, smallest node is first, double is weight
  std::unordered_map<std::pair<int64_t, int64_t>, double, pairhash>
      stoch_edges;  // probability of being blocked
  int64_t origin;
  int64_t goal;
  std::map<std::string, size_t> state_factor_sizes;
  std::vector<std::string> actions;
  std::vector<std::string> observations;
  double _idle_reward;
  double _bad_action_reward;
  double _disambiguate_reward;
  mutable MCVI::LRUCache<MCVI::State, bool, MCVI::StateHash, MCVI::StateEqual>
      goal_reachable;
  mutable MCVI::LRUCache<MCVI::State, double, MCVI::StateHash, MCVI::StateEqual>
      state_value;

 public:
  CTP_Disambiguate(std::mt19937_64& rng, const std::vector<int64_t>& nodes,
                   const std::unordered_map<std::pair<int64_t, int64_t>, double,
                                            pairhash>& edges,
                   const std::unordered_map<std::pair<int64_t, int64_t>, double,
                                            pairhash>& stoch_edges,
                   int64_t origin, int64_t goal)
      : rng(rng),
        nodes(nodes),
        edges(edges),
        stoch_edges(stoch_edges),
        origin(origin),
        goal(goal),
        state_factor_sizes(initStateSpace()),
        actions(initActions()),
        observations(initObs()),
        _idle_reward(initIdleReward()),
        _bad_action_reward(initBadReward()),
        _disambiguate_reward(initDisambiguateReward()),
        goal_reachable(250000),
        state_value(250000) {}

  int64_t GetSizeOfObs() const override { return nodes.size(); }
  int64_t GetSizeOfA() const override { return actions.size(); }
  double GetDiscount() const override { return 1.0; }
  int64_t GetNbAgent() const override { return 1; }
  const std::vector<std::string>& getActions() const { return actions; }
  const std::vector<std::string>& getObs() const { return observations; }
  int64_t getGoal() const { return goal; }
  bool IsTerminal(const MCVI::State& sI) const override {
    return sI.at(sfIdx("loc")) == goal;
  }

#if (USE_HEURISTIC_BOUNDS == 1)
  std::optional<double> GetHeuristicUpper(const MCVI::StateMap<double>& belief,
                                          int64_t max_depth) const override {
    return heuristicUpper(belief, max_depth);
  }
  std::optional<double> GetHeuristicLower(const MCVI::StateMap<double>& belief,
                                          int64_t max_depth) const override {
    return heuristicLower(belief, max_depth);
  }
#endif

  std::tuple<MCVI::State, int64_t, double, bool> Step(const MCVI::State& sI,
                                                      int64_t aI) override {
    MCVI::State sNext;
    const double reward = applyActionToState(sI, aI, sNext);
    const int64_t oI = observeState(sNext);
    const bool finished = checkFinished(sI, aI, sNext);
    // sI_next, oI, Reward, Done
    return std::tuple<MCVI::State, int64_t, double, bool>(sNext, oI, reward,
                                                          finished);
  }

  MCVI::State SampleStartState() override {
    std::uniform_real_distribution<> unif(0, 1);
    std::map<std::string, int64_t> state;
    state["loc"] = origin;
    // stochastic edge status
    for (const auto& [edge, p] : stoch_edges)
      state[edge2str(edge)] = (unif(rng)) < p ? 0 : 1;
    return names2state(state);
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

  std::pair<double, bool> get_state_value(const MCVI::State& state,
                                          int64_t max_depth) const {
    const auto loc_idx = sfIdx("loc");
    MCVI::State eval_state = state;
    if (eval_state.at(loc_idx) == -1) eval_state[loc_idx] = origin;
    if (state_value.contains(eval_state))
      return {state_value.at(eval_state), true};

    // find cost to goal in this realisation of the graph
    std::unordered_map<std::pair<int64_t, int64_t>, double, pairhash>
        state_edges;
    for (const auto& e : edges) {
      if (!stoch_edges.contains(e.first) ||
          state.at(sfIdx(edge2str(e.first))) == 1)
        state_edges.insert(e);
    }
    auto gp = GraphPath(state_edges);
    const auto [costs, pred] = gp.calculate({goal}, max_depth);

    // Calculate discounted reward by following path
    const auto path = gp.reconstructPath({eval_state[loc_idx]}, pred);
    const double gamma = GetDiscount();
    double sum_reward = 0.0;
    double discount = 1.0;
    bool can_terminate = false;
    for (const auto& [loc, action] : path) {
      if (loc == MCVI::State({goal})) {
        can_terminate = true;
        break;
      }

      MCVI::State world_state = eval_state;
      world_state[loc_idx] = loc.at(0);
      MCVI::State sNext;
      const auto state_action = std::distance(
          nodes.begin(), std::find(nodes.begin(), nodes.end(), action));
      const double reward =
          applyActionToState(world_state, state_action, sNext);
      sum_reward += discount * reward;
      discount *= gamma;
    }

    if (!can_terminate) {  // cannot reach goal
      state_value[eval_state] = 0;
      return {0, true};
    }

    state_value[eval_state] = sum_reward;
    return {sum_reward, can_terminate};
  }

 protected:
  std::string edge2str(std::pair<int64_t, int64_t> e) const {
    return "e" + std::to_string(e.first) + "_" + std::to_string(e.second);
  }

  MCVI::State names2state(const std::map<std::string, int64_t>& names) const {
    assert(names.size() == state_factor_sizes.size());
    std::vector<int64_t> state;
    for (const auto& [name, state_elem] : names) {
      const auto sf_sz = state_factor_sizes.find(name);
      assert(sf_sz != state_factor_sizes.cend());
      state.push_back(state_elem);
    }
    return state;
  }

  int64_t sfIdx(const std::string& state_factor) const {
    const auto sf_sz = state_factor_sizes.find(state_factor);
    assert(sf_sz != state_factor_sizes.cend());
    return (int64_t)std::distance(state_factor_sizes.cbegin(), sf_sz);
  }

 private:
  std::vector<std::string> initActions() const {
    std::vector<std::string> acts;
    for (const auto& n : nodes) acts.push_back(std::to_string(n));
    acts.push_back("decide_goal_unreachable");
    return acts;
  }

  std::map<std::string, size_t> initStateSpace() const {
    std::map<std::string, size_t> state_factors;
    // agent location
    state_factors["loc"] = nodes.size();
    // stochastic edge status
    for (const auto& [edge, _] : stoch_edges)
      state_factors[edge2str(edge)] = 2;  // 0 = blocked, 1 = unblocked

    double p = 1.0;
    for (const auto& [sf, sz] : state_factors) p *= sz;
    std::cout << "State space size: " << p << std::endl;

    return state_factors;
  }

  std::vector<std::string> initObs() const {
    std::vector<std::string> obs;
    for (const auto& n : nodes) obs.push_back(std::to_string(n));
    return obs;
  }

  bool nodesConnected(int64_t a, int64_t b) const {
    if (a == b) return true;
    const auto edge = a < b ? std::pair(a, b) : std::pair(b, a);
    if (edges.find(edge) == edges.end()) return false;  // edge does not exist
    return true;
  }

  bool edgeUnblocked(int64_t a, int64_t b, const MCVI::State& state) const {
    const auto edge = a < b ? std::pair(a, b) : std::pair(b, a);

    // check if edge is stochastic
    const auto stoch_ptr = stoch_edges.find(edge);
    if (stoch_ptr == stoch_edges.end()) return true;  // deterministic edge

    // check if edge is unblocked
    return state.at(sfIdx(edge2str(edge))) == 1;  // traversable
  }

  double applyActionToState(const MCVI::State& state, int64_t action,
                            MCVI::State& sNext) const {
    sNext = state;
    const int64_t loc_idx = sfIdx("loc");
    const int64_t loc = state.at(loc_idx);
    if (loc == goal) return 0;  // goal is absorbing

    if (actions.at(action) == "decide_goal_unreachable")
      return goalUnreachable(state) ? 0 : _bad_action_reward;

    const int64_t dest_loc = nodes.at(action);

    if (loc == dest_loc) return _idle_reward;  // idling

    // invalid move
    if (!nodesConnected(loc, dest_loc)) return _bad_action_reward;

    // disambiguate stochastic edge (blocked)
    if (!edgeUnblocked(loc, dest_loc, state)) return _disambiguate_reward;

    // moving
    sNext[loc_idx] = dest_loc;
    return dest_loc < loc ? -edges.at({dest_loc, loc})
                          : -edges.at({loc, dest_loc});
  }

  int64_t observeState(const MCVI::State& state) const {
    int64_t loc = state.at(sfIdx("loc"));
    return std::distance(nodes.begin(),
                         std::find(nodes.begin(), nodes.end(), loc));
  }

  bool checkFinished(const MCVI::State& sI, int64_t aI,
                     const MCVI::State& sNext) const {
    const int64_t loc_idx = sfIdx("loc");
    if (actions.at(aI) == "decide_goal_unreachable" && goalUnreachable(sI))
      return true;
    return sNext.at(loc_idx) == goal;
  }

  bool goalUnreachable(const MCVI::State& state) const {
    // check if goal is reachable from the origin (cached)
    MCVI::State origin_state = state;
    origin_state[sfIdx("loc")] = origin;
    const auto ret = goal_reachable.find(origin_state);
    if (ret != goal_reachable.end()) return !ret->second.first;

    // find shortest path to goal, return true if none exists
    std::unordered_map<std::pair<int64_t, int64_t>, double, pairhash>
        state_edges;
    for (const auto& e : edges) {
      if (!stoch_edges.contains(e.first) ||
          state.at(sfIdx(edge2str(e.first))) == 1)
        state_edges.insert(e);
    }
    auto gp = GraphPath(state_edges);
    const auto [costs, pred] = gp.calculate({origin}, state_edges.size() + 1);
    const bool reaches_goal = costs.contains({goal});
    goal_reachable[origin_state] = reaches_goal;

    return !reaches_goal;
  }

  double initDisambiguateReward() const {
    const double min_edge =
        std::min_element(edges.begin(), edges.end(), CmpPair)->second;
    return -min_edge;
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

  std::vector<int64_t> sfToIndices(
      const std::vector<std::string>& sf_keys) const {
    std::vector<int64_t> indices;
    for (const auto& key : sf_keys) indices.push_back(sfIdx(key));
    return indices;
  }

  double sumStateValues(const MCVI::State& state,
                        const std::vector<int64_t>& sf_indices) const {
    return std::accumulate(
        sf_indices.begin(), sf_indices.end(), 0.0,
        [&state](double sum, int64_t index) { return sum + state[index]; });
  }

  MCVI::State findMaxSumElement(const MCVI::StateMap<double>& belief,
                                const std::vector<int64_t>& indices) const {
    MCVI::State maxElement;
    double maxSum = -std::numeric_limits<double>::infinity();

    for (const auto& [state, value] : belief) {
      double currentSum = sumStateValues(state, indices);
      if (currentSum > maxSum) {
        maxSum = currentSum;
        maxElement = state;
      }
    }

    return maxElement;
  }

  MCVI::State findMinSumElement(const MCVI::StateMap<double>& belief,
                                const std::vector<int64_t>& indices) const {
    MCVI::State minElement;
    double minSum = std::numeric_limits<double>::infinity();

    for (const auto& [state, value] : belief) {
      double currentSum = sumStateValues(state, indices);
      if (currentSum < minSum) {
        minSum = currentSum;
        minElement = state;
      }
    }

    return minElement;
  }

  // find an upper bound for the value of a belief
  double heuristicUpper(const MCVI::StateMap<double>& belief,
                        int64_t max_depth) const {
    std::vector<std::string> sf_keys;
    for (const auto& [e, p] : stoch_edges) sf_keys.push_back(edge2str(e));
    const auto sf_indices = sfToIndices(sf_keys);
    // get state with most traversable edges
    MCVI::State best_case_state = findMaxSumElement(belief, sf_indices);

    const auto it = state_value.find(best_case_state);
    if (it != state_value.end()) return it->second.first;

    const auto [val, _] = get_state_value(best_case_state, max_depth);
    state_value[best_case_state] = val;
    return val;
  }

  double heuristicLower(const MCVI::StateMap<double>& belief,
                        int64_t max_depth) const {
    std::vector<std::string> sf_keys;
    for (const auto& [e, p] : stoch_edges) sf_keys.push_back(edge2str(e));
    const auto sf_indices = sfToIndices(sf_keys);
    // get state with most traversable edges
    MCVI::State worst_case_state = findMinSumElement(belief, sf_indices);

    const auto it = state_value.find(worst_case_state);
    if (it != state_value.end()) return it->second.first;

    const auto [val, _] = get_state_value(worst_case_state, max_depth);
    state_value[worst_case_state] = val;
    return val;
  }
};
