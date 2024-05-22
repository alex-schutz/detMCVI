#pragma once

#include <random>

#include "../statespace.h"
#include "SimInterface.h"

class Battleships : public MCVI::SimInterface {
 private:
  int64_t grid_size;
  StateSpace stateSpace;
  std::vector<std::string> actions;
  std::vector<std::string> observations;
  std::mt19937_64& rng;

  int ship_count_multiplier;

  double _hit_reward = 1;
  double _miss_reward = 0;
  double _bad_action_reward = -10;

 public:
  Battleships(int32_t grid_size, int ship_count_multiplier,
              std::mt19937_64& rng)
      : grid_size(grid_size),
        stateSpace(initStateSpace()),
        actions(initActions()),
        observations(initObs()),
        ship_count_multiplier(ship_count_multiplier),
        rng(rng) {}

  int64_t GetSizeOfObs() const override { return observations.size(); }
  int64_t GetSizeOfA() const override { return actions.size(); }
  double GetDiscount() const override { return 1.0; }
  int64_t GetNbAgent() const override { return 1; }
  const std::vector<std::string>& getActions() const { return actions; }
  const std::vector<std::string>& getObs() const { return observations; }
  bool IsTerminal(int64_t sI) const override {
    for (int64_t i = 0; i < grid_size; ++i)
      for (int64_t j = 0; j < grid_size; ++j)
        if (stateSpace.getStateFactorElem(sI, coord2str(i, j)) == 1)
          return false;
    return true;
  }

  std::tuple<int64_t, int64_t, double, bool> Step(int64_t sI,
                                                  int64_t aI) override {
    int64_t sNext;
    const double reward = applyActionToState(sI, aI, sNext);
    const int64_t oI = observeState(sNext, aI);
    const bool finished = IsTerminal(sNext);
    // sI_next, oI, Reward, Done
    return std::tuple<int64_t, int64_t, double, bool>(sNext, oI, reward,
                                                      finished);
  }

  int64_t SampleStartState() override {
    return stateSpace.stateIndex(generateBattleshipConfig());
  }

 private:
  std::string coord2str(int64_t i, int64_t j) const {
    return std::to_string(i) + "_" + std::to_string(j);
  }

  std::vector<std::string> initActions() const {
    std::vector<std::string> acts;
    for (int64_t i = 0; i < grid_size; ++i)
      for (int64_t j = 0; j < grid_size; ++j) acts.push_back(coord2str(i, j));
    return acts;
  }

  StateSpace initStateSpace() const {
    std::map<std::string, std::vector<int64_t>> state_factors;
    // tile status
    for (int64_t i = 0; i < grid_size; ++i)
      for (int64_t j = 0; j < grid_size; ++j)
        state_factors[coord2str(i, j)] = {0, 1, -1};  // empty, active, sunk
    StateSpace ss(state_factors);
    std::cout << "State space size: " << ss.size() << std::endl;
    return ss;
  }

  std::vector<std::string> initObs() const {
    std::vector<std::string> obs = {"miss", "hit"};
    return obs;
  }

  int64_t observeState(int64_t sI, int64_t aI) const {
    if (stateSpace.getStateFactorElem(sI, actions.at(aI)) == -1) return 1;
    return 0;
  }

  double applyActionToState(int64_t sI, int64_t aI, int64_t& sNext) const {
    sNext = sI;
    if (stateSpace.getStateFactorElem(sI, actions.at(aI)) == 1) {
      sNext = stateSpace.updateStateFactor(sNext, actions.at(aI), -1);
      return _hit_reward;
    }
    if (stateSpace.getStateFactorElem(sI, actions.at(aI)) == -1)  // already hit
      return _bad_action_reward;
    return _miss_reward;
  }

  // No ships may be adjacent or diagonally adjacent in a valid configuration
  bool tile_valid(const std::map<std::string, int64_t>& grid, int64_t row,
                  int64_t col) {
    for (int dr = -1; dr <= 1; ++dr) {
      for (int dc = -1; dc <= 1; ++dc) {
        const int64_t r = row + dr;
        const int64_t c = col + dc;
        if (r >= 0 && r < grid_size && c >= 0 && c < grid_size &&
            (grid.at(coord2str(r, c)) != 0))
          return false;
      }
    }
    return true;
  }

  bool valid_ship_placement(const std::map<std::string, int64_t>& grid,
                            int64_t row, int64_t col, int ship_size,
                            bool is_horizontal) {
    if (is_horizontal) {
      if (col + ship_size > grid_size) return false;
      for (int i = 0; i < ship_size; ++i)
        if (!tile_valid(grid, row, col + i)) return false;
    } else {  // vertical
      if (row + ship_size > grid_size) return false;
      for (int i = 0; i < ship_size; ++i)
        if (!tile_valid(grid, row + i, col)) return false;
    }

    return true;
  }

  void place_ship(std::map<std::string, int64_t>& grid, int64_t row,
                  int64_t col, int ship_size, bool is_horizontal) {
    if (is_horizontal) {
      for (int i = 0; i < ship_size; ++i) grid[coord2str(row, col + i)] = 1;
    } else {  // vertical
      for (int i = 0; i < ship_size; ++i) grid[coord2str(row + i, col)] = 1;
    }
  }

  std::tuple<int64_t, int64_t, bool> generate_random_position() {
    static std::uniform_int_distribution<int64_t> unif(0, grid_size - 1);
    static std::uniform_int_distribution<int64_t> coin(0, 1);
    return {unif(rng), unif(rng), coin(rng)};
  }

  bool attempt_to_place_ship(std::map<std::string, int64_t>& grid,
                             int ship_size) {
    const int max_attempts = 100;
    for (int i = 0; i < max_attempts; ++i) {
      const auto [row, col, direction] = generate_random_position();
      if (valid_ship_placement(grid, row, col, ship_size, direction)) {
        place_ship(grid, row, col, ship_size, direction);
        return true;
      }
    }
    return false;
  }

  // Find a legal arrangement of ships for a square grid of size `grid_size`
  // Ship sizes are 2, 3, 4 and 5, and the number of each type is given by
  // `ship_count_multiplier`
  std::map<std::string, int64_t> generateBattleshipConfig() {
    std::map<std::string, int64_t> grid;
    for (int i = 0; i < grid_size; ++i)
      for (int j = 0; j < grid_size; ++j) grid[coord2str(i, j)] = 0;

    std::vector<int> ship_sizes = {2, 3, 4, 5};

    for (const auto& ship_sz : ship_sizes) {
      for (int n = 0; n < ship_count_multiplier; ++n) {
        if (!attempt_to_place_ship(grid, ship_sz)) {
          throw std::runtime_error(
              "Failed to place all ships, please try again with different "
              "parameters or a larger grid.");
        }
      }
    }

    return grid;
  }
};
