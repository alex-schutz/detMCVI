#include <iostream>
#include <random>

#include "MCVI.h"
#include "SimInterface.h"
#include "delaunay.h"
#include "statespace.h"

using namespace MCVI;

struct pairhash {
 public:
  template <typename T, typename U>
  std::size_t operator()(const std::pair<T, U>& x) const {
    return std::hash<T>()(x.first) ^ std::hash<U>()(x.second);
  }
};

void visualiseGraph(
    const std::vector<Point>& points,
    const std::unordered_map<std::pair<int, int>, double, pairhash>& edges,
    const std::unordered_map<std::pair<int, int>, double, pairhash>&
        stoch_edges,
    int origin, int goal, std::ostream& os) {
  if (!os) throw std::logic_error("Unable to write graph to file");

  os << "graph G {" << std::endl;

  for (size_t i = 0; i < points.size(); ++i) {
    os << "  " << i << " [pos=\"" << points[i].x << "," << points[i].y
       << "!\", label=\"" << i << "\"";
    if (i == (size_t)origin) os << ", fillcolor=\"#ff7f0e\", style=filled";
    if (i == (size_t)goal) os << ", fillcolor=\"#2ca02c\", style=filled";
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

class CTP : public MCVI::SimInterface {
 private:
  mutable std::mt19937_64 rng;
  std::vector<int> nodes;
  std::unordered_map<std::pair<int, int>, double, pairhash>
      edges;  // bidirectional, smallest node is first, double is weight
  std::unordered_map<std::pair<int, int>, double, pairhash>
      stoch_edges;  // probability of being blocked
  int goal;
  int origin;
  std::vector<std::string> actions;
  std::vector<std::string> observations;
  double _move_reward = -1;
  double _idle_reward = -1;
  double _bad_action_reward = -50;

  StateSpace stateSpace;
  StateSpace observationSpace;

 public:
  CTP(int nodes, int stoch_edge_count, bool use_edge_weights = false,
      uint64_t seed = std::random_device{}())
      : rng(seed),
        actions(initActions(nodes)),
        stateSpace({}),
        observationSpace({}) {
    generateGraph(nodes, stoch_edge_count, use_edge_weights);
    initStateSpace();
    initObsSpace();
  }

  int GetSizeOfObs() const override { return observationSpace.size(); }
  int GetSizeOfA() const override { return actions.size(); }
  double GetDiscount() const override { return 0.95; }
  int GetNbAgent() const override { return 1; }
  const std::vector<std::string>& getActions() const { return actions; }
  const std::vector<std::string>& getObs() const { return observations; }

  std::tuple<int, int, double, bool> Step(int sI, int aI) override {
    int sNext;
    const double reward = applyActionToState(sI, aI, sNext);
    const int oI = observeState(sNext);
    const bool finished = stateSpace.getStateFactorElem(sNext, "loc") == goal;
    // sI_next, oI, Reward, Done
    return std::tuple<int, int, double, bool>(sNext, oI, reward, finished);
  }

  int SampleStartState() override {
    std::uniform_real_distribution<> unif(0, 1);
    std::map<std::string, int> state;
    // agent starts at origin
    state["loc"] = origin;
    // stochastic edge status
    for (const auto& [edge, p] : stoch_edges)
      state[edge2str(edge)] = (unif(rng)) < p ? 0 : 1;
    return stateSpace.stateIndex(state);
  }

 private:
  double nodeDistance(Point p1, Point p2) const {
    return std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2));
  }

  void addEdge(int n1, int n2, bool use_edge_weights,
               const std::vector<Point>& points) {
    const int p1 = (n1 < n2) ? n1 : n2;
    const int p2 = (n1 < n2) ? n2 : n1;
    double w =
        use_edge_weights ? nodeDistance(points[p1], points[p2]) : -_move_reward;
    edges[{p1, p2}] = w;
  }

  std::unordered_map<std::pair<int, int>, double, pairhash> chooseStochEdges(
      int k) const {
    std::vector<std::pair<int, int>> edge_keys;
    edge_keys.reserve(edges.size());
    for (const auto& elem : edges) edge_keys.push_back(elem.first);
    std::shuffle(edge_keys.begin(), edge_keys.end(), rng);
    if (edge_keys.size() > (size_t)k) edge_keys.resize(k);

    std::unordered_map<std::pair<int, int>, double, pairhash> stochastic_edges;
    std::uniform_real_distribution<> p(0, 1);
    for (const auto& key : edge_keys) stochastic_edges[key] = p(rng);

    return stochastic_edges;
  }

  void generateGraph(int n_nodes, int k, bool use_edge_weights) {
    // use 10 x 10 grid to generate points
    std::vector<Point> points;
    std::uniform_real_distribution<> d(0, 10);
    for (int i = 0; i < n_nodes; ++i) {
      const double x = d(rng);
      const double y = d(rng);
      nodes.push_back(i);
      points.push_back(Point(x, y));
    }
    std::cerr << "size " << points.size() << std::endl;

    for (const Triangle& tri : delaunayTriangulation(points)) {
      addEdge(tri.p1, tri.p2, use_edge_weights, points);
      addEdge(tri.p2, tri.p3, use_edge_weights, points);
      addEdge(tri.p3, tri.p1, use_edge_weights, points);
    }

    // Choose origin and goal at opposite sides of the map
    const auto extrema = std::minmax_element(
        points.cbegin(), points.cend(),
        [](const Point& a, const Point& b) { return a.y < b.y; });
    origin = (extrema.first - points.cbegin());  //  southernmost point
    goal = (extrema.second - points.cbegin());   //  northernmost point
    std::cerr << "size " << points.size() << " origin: " << origin
              << " goal: " << goal << std::endl;

    stoch_edges = chooseStochEdges(k);

    visualiseGraph(points, edges, stoch_edges, origin, goal, std::cout);
  }

  std::vector<std::string> initActions(int N) const {
    std::vector<std::string> acts;
    for (int n = 0; n < N; ++n) acts.push_back(std::to_string(n));
    return acts;
  }

  std::string edge2str(std::pair<int, int> e) const {
    return "e" + std::to_string(e.first) + "_" + std::to_string(e.second);
  }

  void initStateSpace() {
    std::map<std::string, std::vector<int>> state_factors;
    // agent location
    state_factors["loc"] = nodes;
    // stochastic edge status
    for (const auto& [edge, _] : stoch_edges)
      state_factors[edge2str(edge)] = {0, 1};  // 0 = blocked, 1 = unblocked
    stateSpace = StateSpace(state_factors);
  }

  // agent can observe any element from state space or -1 (unknown)
  void initObsSpace() {
    std::map<std::string, std::vector<int>> observation_factors;
    // agent location
    std::vector<int> agent_locs = nodes;
    agent_locs.push_back(-1);
    observation_factors["loc"] = agent_locs;
    // stochastic edge status
    for (const auto& [edge, _] : stoch_edges)
      observation_factors[edge2str(edge)] = {0, 1, -1};
    observationSpace = StateSpace(observation_factors);
    initObs();
  }

  std::string map2string(const std::map<std::string, int>& map) const {
    std::stringstream ss;
    for (auto it = map.begin(); it != map.end(); ++it) {
      ss << it->first << ": " << it->second;
      if (std::next(it) != map.end()) {
        ss << ", ";
      }
    }
    return ss.str();
  }

  void initObs() {
    for (size_t o = 0; o < observationSpace.size(); ++o)
      observations.push_back(map2string(observationSpace.at(o)));
  }

  bool nodesAdjacent(int a, int b, int state) const {
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

  double applyActionToState(int state, int action, int& sNext) const {
    sNext = state;
    const int loc = stateSpace.getStateFactorElem(state, "loc");
    if (!nodesAdjacent(loc, action, state)) return _bad_action_reward;

    sNext = stateSpace.updateStateFactor(state, "loc", action);
    if (loc == action) return _idle_reward;
    return action < loc ? -edges.at({action, loc}) : -edges.at({loc, action});
  }

  int observeState(int state) const {
    std::map<std::string, int> observation;

    const int loc = stateSpace.getStateFactorElem(state, "loc");
    observation["loc"] = loc;

    // stochastic edge status
    for (const auto& [edge, _] : stoch_edges) {
      if (loc == edge.first || loc == edge.second) {
        const int status = stateSpace.getStateFactorElem(state, edge2str(edge));
        observation[edge2str(edge)] = status;
      } else {
        observation[edge2str(edge)] = -1;
      }
    }
    return observationSpace.stateIndex(observation);
  }
};

int main() {
  // Initialise the POMDP
  auto pomdp = CTP(6, 6, true, 124543);

  const int64_t nb_particles_b0 = 10000;
  const int64_t max_node_size = 10000;

  // Sample the initial belief
  std::vector<int64_t> particles;
  for (int i = 0; i < nb_particles_b0; ++i)
    particles.push_back(pomdp.SampleStartState());
  const auto init_belief = BeliefParticles(particles);

  // Set the Q-learning policy
  const int64_t max_sim_depth = 15;
  const double learning_rate = 0.9;
  const int64_t nb_episode_size = 30;
  const int64_t nb_max_episode = 10;
  const int64_t nb_sim = 40;
  const double decay_Q_learning = 0.01;
  const double epsilon_Q_learning = 0.001;
  const auto q_policy = QLearningPolicy(
      learning_rate, decay_Q_learning, max_sim_depth, nb_max_episode,
      nb_episode_size, nb_sim, epsilon_Q_learning);

  // Initialise the FSC
  std::vector<int64_t> action_space;
  std::vector<int64_t> observation_space;
  for (int i = 0; i < pomdp.GetSizeOfA(); ++i) action_space.push_back(i);
  for (int i = 0; i < pomdp.GetSizeOfObs(); ++i) observation_space.push_back(i);
  const auto init_fsc =
      AlphaVectorFSC(max_node_size, action_space, observation_space);

  // Run MCVI
  auto planner = MCVIPlanner(&pomdp, init_fsc, init_belief, q_policy);
  const int64_t nb_sample = 1000;
  const double converge_thresh = 0.1;
  const int64_t max_iter = 30;
  const auto fsc =
      planner.Plan(max_sim_depth, nb_sample, converge_thresh, max_iter);

  fsc.GenerateGraphviz(std::cerr, pomdp.getActions(), pomdp.getObs());

  // Simulate the resultant FSC
  planner.SimulationWithFSC(20);

  return 0;
}
