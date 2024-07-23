#pragma once

#include <cassert>
#include <random>

#include "Sample.h"
#include "SimInterface.h"

class Wumpus : public MCVI::SimInterface {
 private:
  int64_t grid_size;
  std::vector<std::string> actions = {
      "forward", "turn_90_deg_left", "turn_90_deg_right", "grab", "shoot",
      "climb"};
  std::vector<std::string> observations;

  int wumpus_count;
  double pit_probability;
  int gold_count;

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
        wumpus_count(1),
        pit_probability(0.2),
        gold_count(1),
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

  std::tuple<MCVI::State, int64_t, double, bool> Step(const MCVI::State& sI,
                                                      int64_t aI) override {
    MCVI::State sNext;
    const double reward = applyActionToState(sI, aI, sNext);
    const int64_t oI = observeState(sNext, aI);
    const bool finished = IsTerminal(sNext);
    // sI_next, oI, Reward, Done
    return std::tuple<MCVI::State, int64_t, double, bool>(sNext, oI, reward,
                                                          finished);
  }

  MCVI::State SampleStartState() override {
    static std::uniform_real_distribution<> pit_dist(0, 1);
    std::map<std::string, int64_t> state_factors;

    // the entrance/exit are fixed at (0,0) but agent starts in special init
    // state for init observation
    state_factors["player_pos"] = 0;
    state_factors["player_dir"] = 1;  // Facing East
    state_factors["player_state"] = -1;
    state_factors["player_gold"] = 0;
    state_factors["player_arrow"] = wumpus_count;

    for (int x = 0; x < grid_size; ++x) {
      for (int y = 0; y < grid_size; ++y) {
        state_factors[coord2str(x, y) + "_wumpus"] = 0;
        state_factors[coord2str(x, y) + "_pit"] =
            pit_dist(rng) < pit_probability;
        state_factors[coord2str(x, y) + "_gold"] = 0;
      }
    }

    std::unordered_map<int64_t, double> unif_pmf;
    for (int i = 1; i < grid_size * grid_size; ++i) unif_pmf[i] = 1;
    const auto wumpus_locs = MCVI::weightedShuffle(unif_pmf, rng, wumpus_count);
    for (const auto& [wumpus_loc, d] : wumpus_locs)
      state_factors[coord2str(wumpus_loc / grid_size, wumpus_loc % grid_size) +
                    "_wumpus"] = 1;

    const auto gold_locs = MCVI::weightedShuffle(unif_pmf, rng, gold_count);
    for (const auto& [gold_loc, d] : gold_locs)
      state_factors[coord2str(gold_loc / grid_size, gold_loc % grid_size) +
                    "_gold"] = 1;

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
      if (coordHasItem(sI, x, y, "wumpus") || coordHasItem(sI, x, y, "pit")) {
        sNext[sfIdx("player_state")] = 1;  // terminate
        return _failed_reward + _action_reward;
      }
      return _action_reward;

    } else if (actions[aI] == "turn_90_deg_left") {  // turn left
      sNext[sfIdx("player_dir")] = (sNext[sfIdx("player_dir")] - 1) % 4;
      return _action_reward;

    } else if (actions[aI] == "turn_90_deg_right") {  // turn right
      sNext[sfIdx("player_dir")] = (sNext[sfIdx("player_dir")] + 1) % 4;
      return _action_reward;

    } else if (actions[aI] == "grab" &&
               coordHasItem(sI, loc_x, loc_y, "gold")) {  // grab gold
      // remove gold from world
      sNext[sfIdx(coord2str(loc_x, loc_y) + "_gold")] = 0;
      // put gold in inventory
      sNext[sfIdx("player_gold")] += 1;
      return _action_reward;

    } else if (actions[aI] == "shoot" && loc == 0 &&
               sI[sfIdx("player_arrow")] > 0) {  // shoot arrow
      // remove an arrow from inventory
      sNext[sfIdx("player_arrow")] -= 1;
      // kill wumpus in that direction
      int64_t x = loc_x + x_inc;
      int64_t y = loc_y + y_inc;
      int64_t x_end = (x_inc != 0) ? (x_inc > 0 ? grid_size : -1) : loc_x;
      int64_t y_end = (y_inc != 0) ? (y_inc > 0 ? grid_size : -1) : loc_y;

      while ((x_inc != 0 && x != x_end) || (y_inc != 0 && y != y_end)) {
        if (coordHasItem(sI, x, y, "wumpus")) {
          sNext[sfIdx(coord2str(x, y) + "_wumpus")] = 0;
          break;  // only kill the first wumpus encountered
        }
        x += x_inc;
        y += y_inc;
      }

      return _shoot_reward + _action_reward;

    } else if (actions[aI] == "climb" && loc == 0) {
      sNext[sfIdx("player_state")] = 1;  // terminate
      return _success_reward * sI[sfIdx("player_gold")] + _action_reward;
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
      std::string o = "";
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
        state_factors[coord2str(x, y) + "_wumpus"] =
            2;                                         // 0=no wumpus, 1=wumpus
        state_factors[coord2str(x, y) + "_pit"] = 2;   // 0=no pit, 1=pit
        state_factors[coord2str(x, y) + "_gold"] = 2;  // 0=no gold, 1=gold
      }
    }
    state_factors["player_pos"] =
        grid_size * grid_size;  // grid coord (NxM grid in the form of M*x+y)
    state_factors["player_dir"] = 4;  // 0=N, 1=E, 2=S, 3=W
    state_factors["player_gold"] = gold_count + 1;
    state_factors["player_arrow"] = wumpus_count + 1;
    state_factors["player_state"] = 3;  // -1=init, 0=playing, 1=terminal

    size_t p = 1;
    for (const auto& [sf, sz] : state_factors) p *= sz;
    std::cout << "State space size: " << p << std::endl;

    return state_factors;
  }

  int64_t observeState(const MCVI::State& sI, int64_t aI) const {
    std::string obs = "";
    if (sI[sfIdx("player_state")] == 1) {  // terminal state
      return (int64_t)std::distance(
          observations.cbegin(),
          std::find(observations.cbegin(), observations.cend(), obs));
    }

    const int64_t loc = sI[sfIdx("player_pos")];
    const int64_t dir = sI[sfIdx("player_dir")];
    const int64_t loc_x = loc / grid_size;
    const int64_t loc_y = loc % grid_size;
    const int64_t x_inc = (dir == 1) ? 1 : ((dir == 3) ? -1 : 0);
    const int64_t y_inc = (dir == 0) ? 1 : ((dir == 2) ? -1 : 0);

    for (int i = 0; i < 4; ++i) {  // adjacent wumpus
      const int64_t x =
          (dir == 1) ? loc_x + 1 : ((dir == 3) ? loc_x - 1 : loc_x);
      const int64_t y =
          (dir == 0) ? loc_y + 1 : ((dir == 2) ? loc_y - 1 : loc_y);
      if (x >= grid_size || x < 0 || y >= grid_size || y < 0) continue;
      if (coordHasItem(sI, x, y, "wumpus")) {
        obs += "stench";
        break;
      }
    }

    for (int i = 0; i < 4; ++i) {  // adjacent pit
      const int64_t x =
          (dir == 1) ? loc_x + 1 : ((dir == 3) ? loc_x - 1 : loc_x);
      const int64_t y =
          (dir == 0) ? loc_y + 1 : ((dir == 2) ? loc_y - 1 : loc_y);
      if (x >= grid_size || x < 0 || y >= grid_size || y < 0) continue;
      if (coordHasItem(sI, x, y, "pit")) {
        obs += "breeze";
        break;
      }
    }

    if (coordHasItem(sI, loc_x, loc_y, "gold")) obs += "glitter";

    if (actions[aI] == "forward") {  // bump into wall
      if (loc_x + x_inc < 0 || loc_x + x_inc >= grid_size ||
          loc_y + y_inc < 0 || loc_y + y_inc >= grid_size)
        obs += "bump";
    } else if (actions[aI] == "shoot" && loc == 0 &&
               sI[sfIdx("player_arrow")] > 0) {  // hear wumpus scream
      int64_t x = loc_x + x_inc;
      int64_t y = loc_y + y_inc;
      int64_t x_end = (x_inc != 0) ? (x_inc > 0 ? grid_size : -1) : loc_x;
      int64_t y_end = (y_inc != 0) ? (y_inc > 0 ? grid_size : -1) : loc_y;

      while ((x_inc != 0 && x != x_end) || (y_inc != 0 && y != y_end)) {
        if (coordHasItem(sI, x, y, "wumpus")) {
          obs += "scream";
          break;
        }
        x += x_inc;
        y += y_inc;
      }
    }

    return (int64_t)std::distance(
        observations.cbegin(),
        std::find(observations.cbegin(), observations.cend(), obs));
  }
};
