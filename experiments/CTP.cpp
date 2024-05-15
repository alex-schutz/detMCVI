#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>

#include "AOStar.h"
// #include "CTP_graph.h"
#include "MCVI.h"
#include "SimInterface.h"
#include "auto_generated_graph.h"
#include "statespace.h"

using namespace MCVI;

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
 private:
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
    state["loc"] = -2;
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

 private:
  std::vector<std::string> initActions() const {
    std::vector<std::string> acts;
    for (const auto& n : nodes) acts.push_back(std::to_string(n));
    acts.push_back("decide_goal_unreachable");
    return acts;
  }

  std::string edge2str(std::pair<int64_t, int64_t> e) const {
    return "e" + std::to_string(e.first) + "_" + std::to_string(e.second);
  }

  StateSpace initStateSpace() const {
    std::map<std::string, std::vector<int64_t>> state_factors;
    // agent location
    state_factors["loc"] = nodes;
    state_factors["loc"].push_back(-2);  // special state for init observation
    // stochastic edge status
    for (const auto& [edge, _] : stoch_edges)
      state_factors[edge2str(edge)] = {0, 1};  // 0 = blocked, 1 = unblocked
    StateSpace ss(state_factors);
    std::cout << "State space size: " << ss.size() << std::endl;
    return ss;
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
    if (a == -2 || b == -2) return false;
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
    if (loc == -2) {  // special initial state
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
    if (loc == -2) loc = origin;  // observe initial state as if at origin

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
    if (stateSpace.getStateFactorElem(sI, "loc") == -2) return false;
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
    const double min_edge =
        std::min_element(edges.begin(), edges.end(), CmpPair)->second;
    return -5 * min_edge;
  }

  double initBadReward() const {
    const double min_edge =
        std::min_element(edges.begin(), edges.end(), CmpPair)->second;
    return -50 * min_edge;
  }
};

static double s_time_diff(const std::chrono::steady_clock::time_point& begin,
                          const std::chrono::steady_clock::time_point& end) {
  return (std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
              .count()) /
         1000.0;
}

void runMCVI(CTP* pomdp, const BeliefDistribution& init_belief,
             std::mt19937_64& rng, int64_t max_sim_depth, int64_t max_node_size,
             int64_t eval_depth, int64_t eval_epsilon, double converge_thresh,
             int64_t max_iter, int64_t max_computation_ms,
             int64_t max_eval_steps, int64_t n_eval_trials,
             int64_t nb_particles_b0) {
  // Initialise heuristic
  PathToTerminal ptt(pomdp);

  // Initialise the FSC
  std::cout << "Initialising FSC" << std::endl;
  const auto init_fsc = AlphaVectorFSC(max_node_size);
  //   const auto init_fsc =
  //       InitialiseFSC(ptt, init_belief, max_sim_depth, max_node_size,
  //       &pomdp);
  //   init_fsc.GenerateGraphviz(std::cerr, pomdp.getActions(), pomdp.getObs());

  // Run MCVI
  std::cout << "Running MCVI" << std::endl;
  const std::chrono::steady_clock::time_point mcvi_begin =
      std::chrono::steady_clock::now();
  auto planner = MCVIPlanner(pomdp, init_fsc, init_belief, ptt, rng);
  const auto [fsc, root] =
      planner.Plan(max_sim_depth, converge_thresh, max_iter, max_computation_ms,
                   eval_depth, eval_epsilon);
  const std::chrono::steady_clock::time_point mcvi_end =
      std::chrono::steady_clock::now();
  std::cout << "MCVI complete (" << s_time_diff(mcvi_begin, mcvi_end)
            << " seconds)" << std::endl;

  // Draw FSC plot
  std::fstream fsc_graph("fsc.dot", std::fstream::out);
  fsc.GenerateGraphviz(fsc_graph, pomdp->getActions(), pomdp->getObs());
  fsc_graph.close();

  // Simulate the resultant FSC
  std::cout << "Simulation with up to " << max_eval_steps
            << " steps:" << std::endl;
  planner.SimulationWithFSC(max_eval_steps);
  std::cout << std::endl;

  // Evaluate the FSC policy
  std::cout << "Evaluation of policy (" << max_eval_steps << " steps, "
            << n_eval_trials << " trials):" << std::endl;
  planner.EvaluationWithSimulationFSC(max_eval_steps, n_eval_trials,
                                      nb_particles_b0);
  std::cout << "detMCVI policy FSC contains " << fsc.NumNodes() << " nodes."
            << std::endl;
  std::cout << std::endl;

  // Draw the internal belief tree
  std::fstream belief_tree("belief_tree.dot", std::fstream::out);
  root->DrawBeliefTree(belief_tree);
  belief_tree.close();
}

void runAOStar(CTP* pomdp, const BeliefDistribution& init_belief,
               std::mt19937_64& rng, int64_t eval_depth, int64_t eval_epsilon,
               int64_t max_iter, int64_t max_computation_ms,
               int64_t max_eval_steps, int64_t n_eval_trials,
               int64_t nb_particles_b0) {
  // Initialise heuristic
  PathToTerminal ptt(pomdp);

  // Create root belief node
  std::shared_ptr<BeliefTreeNode> root = CreateBeliefTreeNode(
      init_belief, 0, ptt, eval_depth, eval_epsilon, pomdp);

  // Run AO*
  std::cout << "Running AO* on belief tree" << std::endl;
  const std::chrono::steady_clock::time_point ao_begin =
      std::chrono::steady_clock::now();
  RunAOStar(root, max_iter, max_computation_ms, ptt, eval_depth, eval_epsilon,
            pomdp);
  const std::chrono::steady_clock::time_point ao_end =
      std::chrono::steady_clock::now();
  std::cout << "AO* complete (" << s_time_diff(ao_begin, ao_end) << " seconds)"
            << std::endl;

  // Draw policy tree
  std::fstream policy_tree("greedy_policy_tree.dot", std::fstream::out);
  const int64_t n_greedy_nodes = root->DrawPolicyTree(policy_tree);
  policy_tree.close();

  // Evaluate policy
  std::cout << "Evaluation of alternative (AO* greedy) policy ("
            << max_eval_steps << " steps, " << n_eval_trials
            << " trials):" << std::endl;
  EvaluationWithGreedyTreePolicy(root, max_eval_steps, n_eval_trials,
                                 nb_particles_b0, pomdp, rng, ptt, "AO*");
  std::cout << "AO* greedy policy tree contains " << n_greedy_nodes << " nodes."
            << std::endl;
}

void parseCommandLine(int argc, char* argv[], int64_t& runtime_ms) {
  if (argc > 1) {
    for (int i = 1; i < argc; ++i) {
      if (std::string(argv[i]) == "--runtime" && i + 1 < argc) {
        runtime_ms = std::stoi(argv[i + 1]);
        break;
      }
    }
  }
}

int main(int argc, char* argv[]) {
  std::mt19937_64 rng(std::random_device{}());

  // Initialise the POMDP
  std::cout << "Initialising CTP" << std::endl;
  auto pomdp = CTP(rng);

  std::cout << "Observation space size: " << pomdp.GetSizeOfObs() << std::endl;

  std::fstream ctp_graph("ctp_graph.dot", std::fstream::out);
  pomdp.visualiseGraph(ctp_graph);
  ctp_graph.close();

  // Initial belief parameters
  const int64_t nb_particles_b0 = 100000;
  const int64_t max_belief_samples = 20000;

  // MCVI parameters
  const int64_t max_sim_depth = 30;
  const int64_t max_node_size = 10000;
  const int64_t eval_depth = 30;
  const int64_t eval_epsilon = 0.005;
  const double converge_thresh = 0.005;
  const int64_t max_iter = 500;
  int64_t max_time_ms = 10000;

  // Evaluation parameters
  const int64_t max_eval_steps = 30;
  const int64_t n_eval_trials = 10000;

  parseCommandLine(argc, argv, max_time_ms);

  // Sample the initial belief
  std::cout << "Sampling initial belief" << std::endl;
  auto init_belief = SampleInitialBelief(nb_particles_b0, &pomdp);
  std::cout << "Initial belief size: " << init_belief.size() << std::endl;
  if (max_belief_samples < init_belief.size()) {
    std::cout << "Downsampling belief" << std::endl;
    init_belief = DownsampleBelief(init_belief, max_belief_samples, rng);
  }

  // Run MCVI
  auto mcvi_ctp = new CTP(pomdp);
  runMCVI(mcvi_ctp, init_belief, rng, max_sim_depth, max_node_size, eval_depth,
          eval_epsilon, converge_thresh, max_iter, max_time_ms, max_eval_steps,
          n_eval_trials, 10 * nb_particles_b0);

  // Compare to AO*
  auto aostar_ctp = new CTP(pomdp);
  runAOStar(aostar_ctp, init_belief, rng, eval_depth, eval_epsilon, max_iter,
            max_time_ms, max_eval_steps, n_eval_trials, 10 * nb_particles_b0);

  return 0;
}
