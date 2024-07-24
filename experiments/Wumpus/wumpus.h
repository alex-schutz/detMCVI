#pragma once

#include <cassert>
#include <random>

#include "Sample.h"
#include "ShortestPath.h"
#include "SimInterface.h"

#define USE_HEURISTIC_BOUNDS 1

static bool CmpPair(const std::pair<MCVI::State, double>& p1,
                    const std::pair<MCVI::State, double>& p2) {
  return p1.second < p2.second;
}

class WumpusPath : public MCVI::ShortestPathFasterAlgorithm {
 private:
  int64_t grid_size;
  std::pair<int64_t, int64_t> wumpus_loc;
  std::vector<std::pair<int64_t, int64_t>> pit_locs;
  std::pair<int64_t, int64_t> gold_loc;

 public:
  WumpusPath(int64_t grid_size, std::pair<int64_t, int64_t> wumpus_loc,
             std::vector<std::pair<int64_t, int64_t>> pit_locs,
             std::pair<int64_t, int64_t> gold_loc)
      : grid_size(grid_size),
        wumpus_loc(wumpus_loc),
        pit_locs(pit_locs),
        gold_loc(gold_loc) {}

  std::vector<std::tuple<MCVI::State, double, int64_t>> getEdges(
      const MCVI::State& node) const override {
    std::vector<std::tuple<MCVI::State, double, int64_t>> out;

    // states = {x, y, dir, wumpus_alive, gold_obtained}
    const int64_t nx = node[0];
    const int64_t ny = node[1];
    const int64_t ndir = node[2];
    const int64_t nwumpus_alive = node[3];
    const int64_t ngold_obtained = node[4];

    {  // forward action (0) / shoot action (4)
      const int64_t x = (ndir == 1) ? nx + 1 : ((ndir == 3) ? nx - 1 : nx);
      const int64_t y = (ndir == 0) ? ny + 1 : ((ndir == 2) ? ny - 1 : ny);
      if (x < grid_size && x >= 0 && y < grid_size && y >= 0) {
        if (std::find(pit_locs.begin(), pit_locs.end(), std::make_pair(x, y)) ==
            pit_locs.end()) {  // not a pit
          if (nwumpus_alive && x == wumpus_loc.first &&
              y == wumpus_loc.second) {
            out.push_back({{nx, ny, ndir, false, ngold_obtained}, 11, 4});
          } else {
            out.push_back({{x, y, ndir, nwumpus_alive, ngold_obtained}, 1, 0});
          }
        }
      }
    }
    {  // turn left (1)
      out.push_back(
          {{nx, ny, (ndir - 1) % 4, nwumpus_alive, ngold_obtained}, 1, 1});
    }
    {  // turn right (2)
      out.push_back(
          {{nx, ny, (ndir + 1) % 4, nwumpus_alive, ngold_obtained}, 1, 2});
    }
    {  // pick up gold (3)
      if (!ngold_obtained && nx == gold_loc.first && ny == gold_loc.second)
        out.push_back({{nx, ny, ndir, nwumpus_alive, true}, 1, 3});
    }
    {  // exit (5)
      if (ngold_obtained && nx == 0 && ny == 0)
        out.push_back(
            {{nx, ny, ndir, nwumpus_alive, ngold_obtained}, -1000, 5});
    }

    return out;
  }

  std::pair<double, bool> bestPath(int64_t max_depth) const {
    const auto [costs, predecessors] = calculate({0, 0}, max_depth);
    const auto best_state =
        std::max_element(costs.begin(), costs.end(), CmpPair);
    if (best_state == costs.end())
      throw std::logic_error("Could not find path in Wumpus World");

    const auto path_to_gold = reconstructPath(best_state->first, predecessors);
    const bool can_reach_gold = path_to_gold.back().first[4];
    return {best_state->second, can_reach_gold};
  }
};

class Wumpus : public MCVI::SimInterface {
 private:
  int64_t grid_size;
  std::vector<std::string> actions = {
      "forward", "turn_90_deg_left", "turn_90_deg_right", "grab", "shoot",
      "climb"};
  std::vector<std::string> observations;

  double pit_probability;

  std::mt19937_64& rng;
  std::map<std::string, size_t> state_factor_sizes;

  double _failed_reward = -1000;
  double _success_reward = 1000;
  double _action_reward = -1;
  double _shoot_reward = -10;
  double _bad_action_reward = -1000;

 public:
  Wumpus(int32_t grid_size, std::mt19937_64& rng)
      : grid_size(grid_size),
        observations(initObs()),
        pit_probability(0.2),
        rng(rng),
        state_factor_sizes(initStateSpace()) {}

  int64_t GetSizeOfObs() const override { return observations.size(); }
  int64_t GetSizeOfA() const override { return actions.size(); }
  double GetDiscount() const override { return 1.0; }
  int64_t GetNbAgent() const override { return 1; }
  const std::vector<std::string>& getActions() const { return actions; }
  const std::vector<std::string>& getObs() const { return observations; }
  bool IsTerminal(const MCVI::State& sI) const override {
    return sI[sfIdx("player_state")] == 1;
  }

  void drawWumpusWorld(const MCVI::State& state) const {
    std::vector<std::vector<std::string>> grid(
        grid_size, std::vector<std::string>(grid_size, "         "));

    // Player position and direction
    int64_t player_pos = state[sfIdx("player_pos")];
    int64_t player_x = player_pos / grid_size;
    int64_t player_y = player_pos % grid_size;
    char player_dir;
    switch (state[sfIdx("player_dir")]) {
      case 1:
        player_dir = '>';
        break;
      case 2:
        player_dir = 'V';
        break;
      case 3:
        player_dir = '<';
        break;
      default:
        player_dir = 'A';
    }
    grid[player_x][player_y][7] = player_dir;

    // Player carries gold
    if (state[sfIdx("player_gold")] == 1) {
      grid[player_x][player_y][6] = 'g';
    }

    // Player carries arrow
    if (state[sfIdx("player_arrow")] == 1) {
      grid[player_x][player_y][8] = 'q';
    }

    // Wumpus position
    int64_t wumpus_x = state[sfIdx("wumpus_x")];
    int64_t wumpus_y = state[sfIdx("wumpus_y")];
    if (wumpus_x != -1 && wumpus_y != -1) {
      grid[wumpus_x][wumpus_y][1] = 'W';
    }

    // Gold position
    int64_t gold_x = state[sfIdx("gold_x")];
    int64_t gold_y = state[sfIdx("gold_y")];
    if (gold_x != -1 && gold_y != -1) {
      grid[gold_x][gold_y][0] = 'G';
    }

    // Pits
    for (int64_t x = 0; x < grid_size; ++x) {
      for (int64_t y = 0; y < grid_size; ++y) {
        if (state[sfIdx(coord2str(x, y) + "_pit")] == 1) {
          grid[x][y][4] = 'P';
        }
      }
    }

    // Draw the grid
    for (int64_t y = grid_size - 1; y >= 0; --y) {
      for (int64_t x = 0; x < grid_size; ++x) {
        std::cout << "+---";
      }
      std::cout << "+" << std::endl;
      for (int64_t x = 0; x < grid_size; ++x) {
        std::cout << "|" << grid[x][y][0] << grid[x][y][1] << grid[x][y][2];
      }
      std::cout << "|" << std::endl;
      for (int64_t x = 0; x < grid_size; ++x) {
        std::cout << "|" << grid[x][y][3] << grid[x][y][4] << grid[x][y][5];
      }
      std::cout << "|" << std::endl;
      for (int64_t x = 0; x < grid_size; ++x) {
        std::cout << "|" << grid[x][y][6] << grid[x][y][7] << grid[x][y][8];
      }
      std::cout << "|" << std::endl;
    }
    for (int64_t x = 0; x < grid_size; ++x) {
      std::cout << "+---";
    }
    std::cout << "+" << std::endl;
  }

#if (USE_HEURISTIC_BOUNDS == 1)
  std::optional<double> GetHeuristicUpper(const MCVI::StateMap<double>& belief,
                                          int64_t max_depth) const override {
    return heuristicUpper(belief, max_depth);
  }
#endif

  std::pair<double, bool> get_state_value(const MCVI::State& state,
                                          int64_t max_depth) const {
    std::pair<int64_t, int64_t> wumpus_loc = {state[sfIdx("wumpus_x")],
                                              state[sfIdx("wumpus_y")]};
    std::vector<std::pair<int64_t, int64_t>> pit_locs;
    std::pair<int64_t, int64_t> gold_loc = {state[sfIdx("gold_x")],
                                            state[sfIdx("gold_y")]};
    for (int64_t x = 0; x < grid_size; ++x)
      for (int64_t y = 0; y < grid_size; ++y)
        if (coordHasItem(state, x, y, "pit")) pit_locs.push_back({x, y});

    WumpusPath path(grid_size, wumpus_loc, pit_locs, gold_loc);
    return path.bestPath(max_depth);
  }

  std::tuple<MCVI::State, int64_t, double, bool> Step(const MCVI::State& sI,
                                                      int64_t aI) override {
    MCVI::State sNext;
    const double reward = applyActionToState(sI, aI, sNext);
    const int64_t oI = observeState(sI, sNext, aI);
    const bool finished = IsTerminal(sNext);
    // sI_next, oI, Reward, Done
    return std::tuple<MCVI::State, int64_t, double, bool>(sNext, oI, reward,
                                                          finished);
  }

  MCVI::State SampleStartState() override {
    static std::uniform_real_distribution<> pit_dist(0, 1);
    static std::uniform_int_distribution<int64_t> grid_dist(
        1, grid_size * grid_size - 1);

    std::map<std::string, int64_t> state_factors;

    // the entrance/exit are fixed at (0,0) but agent starts in special init
    // state for init observation
    state_factors["player_pos"] = 0;
    state_factors["player_dir"] = 1;  // Facing East
    state_factors["player_state"] = -1;
    state_factors["player_gold"] = 0;
    state_factors["player_arrow"] = 1;

    const int64_t wumpus_loc = grid_dist(rng);
    const int64_t gold_loc = grid_dist(rng);
    state_factors["wumpus_x"] = wumpus_loc / grid_size;
    state_factors["wumpus_y"] = wumpus_loc % grid_size;
    state_factors["gold_x"] = gold_loc / grid_size;
    state_factors["gold_y"] = gold_loc % grid_size;

    for (int x = 0; x < grid_size; ++x) {
      for (int y = 0; y < grid_size; ++y) {
        state_factors[coord2str(x, y) + "_pit"] =
            pit_dist(rng) < pit_probability;
      }
    }
    state_factors[coord2str(0, 0) + "_pit"] = 0;

    return names2state(state_factors);
  }

  double applyActionToState(const MCVI::State& sI, int64_t aI,
                            MCVI::State& sNext) const {
    sNext = sI;
    if (sI[sfIdx("player_state")] == -1) {  // initial state
      sNext[sfIdx("player_state")] = 0;
      return 0;
    } else if (sI[sfIdx("player_state")] == 1) {  // terminal state
      return 0;
    }

    const int64_t loc = sI[sfIdx("player_pos")];
    const int64_t dir = sI[sfIdx("player_dir")];
    const int64_t loc_x = loc / grid_size;
    const int64_t loc_y = loc % grid_size;
    const int64_t x_inc = (dir == 1) ? 1 : ((dir == 3) ? -1 : 0);
    const int64_t y_inc = (dir == 0) ? 1 : ((dir == 2) ? -1 : 0);

    if (actions[aI] == "forward") {
      // calc next coords
      const int64_t x = (loc_x + x_inc >= 0 && loc_x + x_inc < grid_size)
                            ? loc_x + x_inc
                            : loc_x;
      const int64_t y = (loc_y + y_inc >= 0 && loc_y + y_inc < grid_size)
                            ? loc_y + y_inc
                            : loc_y;
      const int64_t next_loc = x * grid_size + y;
      // move
      sNext[sfIdx("player_pos")] = next_loc;
      // check for wumpus/pit
      if ((sI[sfIdx("wumpus_x")] == x && sI[sfIdx("wumpus_y")] == y) ||
          coordHasItem(sI, x, y, "pit")) {
        sNext[sfIdx("player_state")] = 1;  // terminate
        return _failed_reward + _action_reward;
      }
      return _action_reward;

    } else if (actions[aI] == "turn_90_deg_left") {  // turn left
      sNext[sfIdx("player_dir")] = (sI[sfIdx("player_dir")] + 3) % 4;
      return _action_reward;

    } else if (actions[aI] == "turn_90_deg_right") {  // turn right
      sNext[sfIdx("player_dir")] = (sI[sfIdx("player_dir")] + 1) % 4;
      return _action_reward;

    } else if (actions[aI] == "grab" &&
               (sI[sfIdx("gold_x")] == loc_x &&
                sI[sfIdx("gold_y")] == loc_y)) {  // grab gold
      // remove gold from world
      sNext[sfIdx("gold_x")] = -1;
      sNext[sfIdx("gold_y")] = -1;
      // put gold in inventory
      sNext[sfIdx("player_gold")] = 1;
      return _action_reward;

    } else if (actions[aI] == "shoot" &&
               sI[sfIdx("player_arrow")] == 1) {  // shoot arrow
      // remove an arrow from inventory
      sNext[sfIdx("player_arrow")] = 0;
      // kill wumpus in that direction
      int64_t x = loc_x + x_inc;
      int64_t y = loc_y + y_inc;
      int64_t x_end = (x_inc != 0) ? (x_inc > 0 ? grid_size : -1) : loc_x;
      int64_t y_end = (y_inc != 0) ? (y_inc > 0 ? grid_size : -1) : loc_y;

      while ((x_inc != 0 && x != x_end) || (y_inc != 0 && y != y_end)) {
        if (sI[sfIdx("wumpus_x")] == x && sI[sfIdx("wumpus_y")] == y) {
          sNext[sfIdx("wumpus_x")] = -1;
          sNext[sfIdx("wumpus_y")] = -1;
          break;
        }
        x += x_inc;
        y += y_inc;
      }

      return _shoot_reward + _action_reward;

    } else if (actions[aI] == "climb" && loc == 0) {
      sNext[sfIdx("player_state")] = 1;  // terminate
      if (sI[sfIdx("player_gold")] == 1) return _success_reward;
      return _bad_action_reward;
    }

    return _bad_action_reward;
  }

  int64_t sfIdx(const std::string& state_factor) const {
    const auto sf_sz = state_factor_sizes.find(state_factor);
    if (sf_sz == state_factor_sizes.cend())
      throw std::logic_error("Could not find state factor " + state_factor);
    return (int64_t)std::distance(state_factor_sizes.cbegin(), sf_sz);
  }

 private:
  std::string coord2str(int64_t i, int64_t j) const {
    return std::to_string(i) + "_" + std::to_string(j);
  }

  bool coordHasItem(const MCVI::State& state, int64_t x, int64_t y,
                    const std::string& item) const {
    return state[sfIdx(coord2str(x, y) + "_" + item)];
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

  static inline bool isBitSet(int64_t num, size_t bit) {
    return 1 == ((num >> bit) & 1);
  }

  std::vector<std::string> initObs() const {
    std::vector<std::string> obs_components = {"stench", "breeze", "glitter",
                                               "bump", "scream"};

    std::vector<std::string> obs;
    for (int64_t b = 0; b < (1 << obs_components.size()); ++b) {
      std::string o = "*";
      for (size_t i = 0; i < obs_components.size(); ++i)
        if (isBitSet(b, i)) o += obs_components[i];
      obs.push_back(o);
    }
    return obs;
  }

  std::map<std::string, size_t> initStateSpace() const {
    std::map<std::string, size_t> state_factors;

    // For each square the following conditions can be met:
    // contains wumpus
    // contains pit
    // contains gold
    // Other variables are:
    // player position
    // game state

    for (int x = 0; x < grid_size; ++x) {
      for (int y = 0; y < grid_size; ++y) {
        state_factors[coord2str(x, y) + "_pit"] = 2;  // 0=no pit, 1=pit
      }
    }
    state_factors["player_pos"] =
        grid_size * grid_size;  // grid coord (NxM grid in the form of M*x+y)
    state_factors["player_dir"] = 4;  // 0=N, 1=E, 2=S, 3=W
    state_factors["player_gold"] = 2;
    state_factors["player_arrow"] = 2;
    state_factors["player_state"] = 3;  // -1=init, 0=playing, 1=terminal
    state_factors["wumpus_x"] = grid_size + 1;
    state_factors["wumpus_y"] = grid_size + 1;
    state_factors["gold_x"] = grid_size + 1;
    state_factors["gold_y"] = grid_size + 1;

    size_t p = 1;
    for (const auto& [sf, sz] : state_factors) p *= sz;
    std::cout << "State space size: " << p << std::endl;

    return state_factors;
  }

  int64_t observeState(const MCVI::State& sPrev, const MCVI::State& sI,
                       int64_t aI) const {
    std::string obs = "*";
    if (sI[sfIdx("player_state")] == 1) {  // terminal state
      return (int64_t)std::distance(
          observations.cbegin(),
          std::find(observations.cbegin(), observations.cend(), obs));
    }

    const int64_t loc = sI[sfIdx("player_pos")];
    const int64_t loc_x = loc / grid_size;
    const int64_t loc_y = loc % grid_size;

    for (int i = 0; i < 4; ++i) {  // adjacent wumpus
      const int64_t x = (i == 1) ? loc_x + 1 : ((i == 3) ? loc_x - 1 : loc_x);
      const int64_t y = (i == 0) ? loc_y + 1 : ((i == 2) ? loc_y - 1 : loc_y);
      if (x >= grid_size || x < 0 || y >= grid_size || y < 0) continue;
      if ((sI[sfIdx("wumpus_x")] == x && sI[sfIdx("wumpus_y")] == y)) {
        obs += "stench";
        break;
      }
    }

    for (int i = 0; i < 4; ++i) {  // adjacent pit
      const int64_t x = (i == 1) ? loc_x + 1 : ((i == 3) ? loc_x - 1 : loc_x);
      const int64_t y = (i == 0) ? loc_y + 1 : ((i == 2) ? loc_y - 1 : loc_y);
      if (x >= grid_size || x < 0 || y >= grid_size || y < 0) continue;
      if (coordHasItem(sI, x, y, "pit")) {
        obs += "breeze";
        break;
      }
    }

    if ((sI[sfIdx("gold_x")] == loc_x && sI[sfIdx("gold_y")] == loc_y))
      obs += "glitter";

    if (actions[aI] == "forward") {  // bump into wall
      const int64_t prev_loc = sPrev[sfIdx("player_pos")];
      const int64_t prev_loc_x = prev_loc / grid_size;
      const int64_t prev_loc_y = prev_loc % grid_size;
      const int64_t dir = sPrev[sfIdx("player_dir")];
      const int64_t x_inc = (dir == 1) ? 1 : ((dir == 3) ? -1 : 0);
      const int64_t y_inc = (dir == 0) ? 1 : ((dir == 2) ? -1 : 0);

      if (prev_loc_x + x_inc < 0 || prev_loc_x + x_inc >= grid_size ||
          prev_loc_y + y_inc < 0 || prev_loc_y + y_inc >= grid_size)
        obs += "bump";
    } else if (actions[aI] == "shoot" && sPrev[sfIdx("player_arrow")] == 1 &&
               sPrev[sfIdx("wumpus_x")] != -1 &&
               sPrev[sfIdx("wumpus_y")] != -1 && sI[sfIdx("wumpus_x")] == -1 &&
               sI[sfIdx("wumpus_y")] == -1) {  // hear wumpus scream
      obs += "scream";
    }

    return (int64_t)std::distance(
        observations.cbegin(),
        std::find(observations.cbegin(), observations.cend(), obs));
  }

  // find an upper bound for the value of a belief
  double heuristicUpper(const MCVI::StateMap<double>& belief,
                        int64_t max_depth) const {
    double val = 0;
    for (const auto& [s, p] : belief) {
      //   double d = 0;
      //   int64_t x = s[sfIdx("player_pos")] / grid_size;
      //   int64_t y = s[sfIdx("player_pos")] % grid_size;
      //   bool got_gold = s[sfIdx("player_gold")];
      //   bool no_wumpus = true;

      //   // wumpus is on gold, have to shoot first
      //   if (s[sfIdx("gold_x")] == s[sfIdx("wumpus_x")] &&
      //       s[sfIdx("gold_y")] == s[sfIdx("wumpus_y")]) {
      //     if (s[sfIdx("player_arrow")] == 1) {
      //       d += _shoot_reward + _action_reward;
      //       no_wumpus = true;
      //     } else
      //       no_wumpus = false;  // wumpus is on gold, can't shoot
      //   }
      //   // gold is available, go grab it
      //   if (no_wumpus && s[sfIdx("player_gold")] == 0 &&
      //       !coordHasItem(s, s[sfIdx("gold_x")], s[sfIdx("gold_y")], "pit"))
      //       {
      //     // l1 distance from player location to gold
      //     d += _action_reward * std::abs(x - s[sfIdx("gold_x")]) +
      //          _action_reward * std::abs(y - s[sfIdx("gold_y")]);
      //     d += _action_reward;  // pick up gold
      //     got_gold = true;
      //     // set player location to gold so we navigate to start
      //     x = s[sfIdx("gold_x")];
      //     y = s[sfIdx("gold_y")];
      //   }

      //   // l1 distance to return to start
      //   d += _action_reward * std::abs(x - 0) + _action_reward * std::abs(y -
      //   0); if (got_gold)
      //     d += _success_reward;
      //   else
      //     d = _bad_action_reward;
      //   val += d * p;
      val += get_state_value(s, max_depth).first * p;
    }

    return val;
  }
};
