#pragma once

#include <algorithm>
#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ShortestPath.h"
#include "SimInterface.h"

#define USE_HEURISTIC_BOUNDS 0

struct pairhash {
 public:
  template <typename T, typename U>
  std::size_t operator()(const std::pair<T, U>& x) const {
    return std::hash<T>()(x.first) ^ std::hash<U>()(x.second);
  }
};

static bool CmpPair(const std::pair<std::pair<int64_t, int64_t>, double>& p1,
                    const std::pair<std::pair<int64_t, int64_t>, double>& p2) {
  return p1.second < p2.second;
}

class GraphPath : public MCVI::ShortestPathFasterAlgorithm {
 private:
  std::unordered_map<std::pair<int64_t, int64_t>, double, pairhash>
      _edges;  // bidirectional, smallest node is first, double is weight

 public:
  GraphPath(
      std::unordered_map<std::pair<int64_t, int64_t>, double, pairhash> edges)
      : _edges(edges) {}

  std::vector<std::tuple<MCVI::State, double, int64_t>> getEdges(
      const MCVI::State& node) const override {
    std::vector<std::tuple<MCVI::State, double, int64_t>> out;
    for (const auto& e : _edges) {
      if (e.first.first == node.at(0))
        out.push_back({{e.first.second}, e.second, e.first.second});
      else if (e.first.second == node.at(0))
        out.push_back({{e.first.first}, e.second, e.first.first});
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
  int64_t origin;
  int64_t goal;
  std::map<std::string, size_t> state_factor_sizes;
  int64_t max_obs_width;
  std::vector<std::string> actions;
  std::vector<std::string> observations;
  double _idle_reward;
  double _bad_action_reward;
  mutable MCVI::StateMap<bool> goal_reachable;
  mutable MCVI::StateMap<double> state_value;

 public:
  CTP(std::mt19937_64& rng, const std::vector<int64_t>& nodes,
      const std::unordered_map<std::pair<int64_t, int64_t>, double, pairhash>&
          edges,
      const std::unordered_map<std::pair<int64_t, int64_t>, double, pairhash>&
          stoch_edges,
      int64_t origin, int64_t goal)
      : rng(rng),
        nodes(nodes),
        edges(edges),
        stoch_edges(stoch_edges),
        origin(origin),
        goal(goal),
        state_factor_sizes(initStateSpace()),
        max_obs_width(initObsWidth()),
        actions(initActions()),
        observations(initObs()),
        _idle_reward(initIdleReward()),
        _bad_action_reward(initBadReward()) {}

  int64_t GetSizeOfObs() const override { return nodes.size() * max_obs_width; }
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
    // agent starts at special initial state (for init observation)
    state["loc"] = -1;
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

  double applyActionToState(const MCVI::State& state, int64_t action,
                            MCVI::State& sNext) const override {
    sNext = state;
    const int64_t loc_idx = sfIdx("loc");
    const int64_t loc = state.at(loc_idx);
    if (loc == -1) {  // special initial state
      sNext[loc_idx] = origin;
      return 0;
    }
    if (loc == goal) return 0;  // goal is absorbing

    if (actions.at(action) == "decide_goal_unreachable")
      return goalUnreachable(state) ? 0 : _bad_action_reward;

    const int64_t dest_loc = nodes.at(action);
    if (loc == dest_loc) return _idle_reward;  // idling

    // invalid move
    if (!nodesAdjacent(loc, dest_loc, state)) return _bad_action_reward;

    // moving
    sNext[loc_idx] = dest_loc;
    return dest_loc < loc ? -edges.at({dest_loc, loc})
                          : -edges.at({loc, dest_loc});
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
      //   assert(sf_sz->second > state_elem);
      state.push_back(state_elem);
    }
    return state;
  }

  int64_t sfIdx(const std::string& state_factor) const {
    const auto sf_sz = state_factor_sizes.find(state_factor);
    assert(sf_sz != state_factor_sizes.cend());
    return (int64_t)std::distance(state_factor_sizes.cbegin(), sf_sz);
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

  std::map<std::string, size_t> initStateSpace() const {
    std::map<std::string, size_t> state_factors;
    // agent location
    state_factors["loc"] =
        nodes.size() + 1;  // special state for init observation
    // stochastic edge status
    for (const auto& [edge, _] : stoch_edges)
      state_factors[edge2str(edge)] = 2;  // 0 = blocked, 1 = unblocked

    double p = 1.0;
    for (const auto& [sf, sz] : state_factors) p *= sz;
    std::cout << "State space size: " << p << std::endl;

    return state_factors;
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

  bool nodesAdjacent(int64_t a, int64_t b, const MCVI::State& state) const {
    if (a == -1 || b == -1) return false;
    if (a == b) return true;
    const auto edge = a < b ? std::pair(a, b) : std::pair(b, a);
    if (edges.find(edge) == edges.end()) return false;  // edge does not exist

    // check if edge is stochastic
    const auto stoch_ptr = stoch_edges.find(edge);
    if (stoch_ptr == stoch_edges.end()) return true;  // deterministic edge

    // check if edge is unblocked
    return state.at(sfIdx(edge2str(edge))) == 1;  // traversable
  }

  int64_t observeState(const MCVI::State& state) const {
    int64_t observation = 0;

    int64_t loc = state.at(sfIdx("loc"));
    if (loc == -1) loc = origin;  // observe initial state as if at origin
    const int64_t loc_idx = std::distance(
        nodes.begin(), std::find(nodes.begin(), nodes.end(), loc));

    // stochastic edge status
    int64_t n = 0;
    for (const auto& edge : AdjacentStochEdges(loc)) {
      if (state.at(sfIdx(edge2str(edge)))) observation |= ((int64_t)1 << n);
      ++n;
    }

    observation += loc_idx * max_obs_width;

    return observation;
  }

  bool checkFinished(const MCVI::State& sI, int64_t aI,
                     const MCVI::State& sNext) const {
    const int64_t loc_idx = sfIdx("loc");
    if (sI.at(loc_idx) == -1) return false;
    if (actions.at(aI) == "decide_goal_unreachable" && goalUnreachable(sI))
      return true;
    return sNext.at(loc_idx) == goal;
  }

  bool goalUnreachable(const MCVI::State& state) const {
    // check if goal is reachable from the origin (cached)
    MCVI::State origin_state = state;
    origin_state[sfIdx("loc")] = origin;
    const auto ret = goal_reachable.find(origin_state);
    if (ret != goal_reachable.end()) return !ret->second;

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
    if (best_case_state.at(sfIdx("loc")) == -1)
      best_case_state[sfIdx("loc")] = origin;

    const auto it = state_value.find(best_case_state);
    if (it != state_value.cend()) return it->second;

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
    if (worst_case_state.at(sfIdx("loc")) == -1)
      worst_case_state[sfIdx("loc")] = origin;

    const auto it = state_value.find(worst_case_state);
    if (it != state_value.cend()) return it->second;

    const auto [val, _] = get_state_value(worst_case_state, max_depth);
    state_value[worst_case_state] = val;
    return val;
  }
};

struct CTPParams {
  std::string filename;               // CTP graph file
  int64_t nb_particles_b0 = 100000;   // num init belief samples
  int64_t max_belief_samples = 2000;  // downsampled belief
  int64_t max_node_size = 10000;      // num FSC nodes
  int64_t max_sim_depth = 100;        // trajectory depth
  double eval_epsilon = 0.005;        // trajectory cumulative discount limit
  double converge_thresh = 0.005;     // upper and lower bound diff
  int64_t max_iter = 10;              // MCVI iterations
  int64_t max_time_ms = 600000;       // MCVI computation time
};

CTPParams parseArgs(int argc, char** argv) {
  CTPParams params;
  std::stringstream ss;
  ss << "Usage: " << argv[0] << " <filename> [options]\n"
     << "Options:\n"
     << "  --nb_particles_b0 <int64_t>      Number of initial belief "
        "samples\n"
     << "  --max_belief_samples <int64_t>   Max downsampled belief\n"
     << "  --max_node_size <int64_t>        Max number of FSC nodes\n"
     << "  --max_sim_depth <int64_t>        Max trajectory depth\n"
     << "  --eval_epsilon <double>          Trajectory cumulative "
        "discount limit\n"
     << "  --converge_thresh <double>       Convergence threshold "
        "(upper and lower bound difference)\n"
     << "  --max_iter <int64_t>             MCVI iterations\n"
     << "  --max_time_ms <int64_t>          MCVI computation time\n"
     << "  --help                           Show this help message\n";

  if (argc < 2) {
    std::cerr << "Error: Missing filename argument.\n";
    std::cerr << ss.str();
    std::exit(1);
  }

  params.filename = argv[1];

  for (int i = 2; i < argc; ++i) {
    if (strcmp(argv[i], "--nb_particles_b0") == 0 && i + 1 < argc) {
      params.nb_particles_b0 = std::stoll(argv[++i]);
    } else if (strcmp(argv[i], "--max_belief_samples") == 0 && i + 1 < argc) {
      params.max_belief_samples = std::stoll(argv[++i]);
    } else if (strcmp(argv[i], "--max_node_size") == 0 && i + 1 < argc) {
      params.max_node_size = std::stoll(argv[++i]);
    } else if (strcmp(argv[i], "--max_sim_depth") == 0 && i + 1 < argc) {
      params.max_sim_depth = std::stoll(argv[++i]);
    } else if (strcmp(argv[i], "--eval_epsilon") == 0 && i + 1 < argc) {
      params.eval_epsilon = std::stod(argv[++i]);
    } else if (strcmp(argv[i], "--converge_thresh") == 0 && i + 1 < argc) {
      params.converge_thresh = std::stod(argv[++i]);
    } else if (strcmp(argv[i], "--max_iter") == 0 && i + 1 < argc) {
      params.max_iter = std::stoll(argv[++i]);
    } else if (strcmp(argv[i], "--max_time_ms") == 0 && i + 1 < argc) {
      params.max_time_ms = std::stoll(argv[++i]);
    } else if (strcmp(argv[i], "--help") == 0) {
      std::cout << ss.str();
      std::exit(0);
    }
  }

  return params;
}

void ctpGraphFromFile(
    const std::string& filename, std::vector<int64_t>& CTPNodes,
    std::unordered_map<std::pair<int64_t, int64_t>, double, pairhash>& CTPEdges,
    std::unordered_map<std::pair<int64_t, int64_t>, double, pairhash>&
        CTPStochEdges,
    int64_t& CTPOrigin, int64_t& CTPGoal) {
  std::ifstream file(filename);
  std::string line = "";
  std::string key = "";

  while (std::getline(file, line)) {
    std::istringstream iss(line);
    iss >> key;

    if (key == "CTPNodes:") {
      int64_t node;
      while (iss >> node) {
        CTPNodes.push_back(node);
      }
    } else if (key == "CTPEdges:") {
      while (true) {
        const std::streampos oldpos = file.tellg();
        if (!std::getline(file, line) || line.empty()) break;
        if (line.find(':') != std::string::npos) {
          file.seekg(oldpos);
          break;
        }
        std::istringstream edgeStream(line);
        int64_t from, to;
        double weight;
        edgeStream >> from >> to >> weight;
        CTPEdges[{from, to}] = weight;
      }
    } else if (key == "CTPStochEdges:") {
      while (true) {
        const std::streampos oldpos = file.tellg();
        if (!std::getline(file, line) || line.empty()) break;
        if (line.find(':') != std::string::npos) {
          file.seekg(oldpos);
          break;
        }
        std::istringstream edgeStream(line);
        int64_t from, to;
        double weight;
        edgeStream >> from >> to >> weight;
        CTPStochEdges[{from, to}] = weight;
      }
    } else if (key == "CTPOrigin:") {
      iss >> CTPOrigin;
    } else if (key == "CTPGoal:") {
      iss >> CTPGoal;
    }
  }
}
