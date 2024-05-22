#pragma once

#include <random>

#include "../statespace.h"
#include "SimInterface.h"

class Battleships : public MCVI::SimInterface {
 private:
  int64_t grid_size;
  std::vector<std::string> actions;
  std::vector<std::string> observations;

  int ship_count_multiplier;
  std::vector<int> ship_sizes = {2, 3, 4, 5};

  StateSpace stateSpace;
  std::mt19937_64& rng;

  double _hit_reward = 1;
  double _miss_reward = 0;
  double _bad_action_reward = -10;

 public:
  Battleships(int32_t grid_size, int ship_count_multiplier,
              std::mt19937_64& rng)
      : grid_size(grid_size),
        actions(initActions()),
        observations(initObs()),
        ship_count_multiplier(ship_count_multiplier),
        stateSpace(initStateSpace()),
        rng(rng) {}

  int64_t GetSizeOfObs() const override { return observations.size(); }
  int64_t GetSizeOfA() const override { return actions.size(); }
  double GetDiscount() const override { return 1.0; }
  int64_t GetNbAgent() const override { return 1; }
  const std::vector<std::string>& getActions() const { return actions; }
  const std::vector<std::string>& getObs() const { return observations; }
  bool IsTerminal(int64_t sI) const override {
    for (const auto& ship_sz : ship_sizes) {
      for (int64_t n = 0; n < ship_count_multiplier; ++n) {
        for (int i = 0; i < ship_sz; ++i) {
          if (stateSpace.getStateFactorElem(
                  sI, ship2str(ship_sz, n) + "_" + std::to_string(i)) == 0)
            return false;
        }
      }
    }
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
    std::map<std::string, int64_t> state_factors;
    const auto ship_locs = generateBattleshipConfig();

    for (const auto& [ship_sz, n, row, col, ori] : ship_locs) {
      state_factors[ship2str(ship_sz, n) + "_row"] = row;
      state_factors[ship2str(ship_sz, n) + "_col"] = col;
      state_factors[ship2str(ship_sz, n) + "_ori"] = ori;
      for (int i = 0; i < ship_sz; ++i) {
        state_factors[ship2str(ship_sz, n) + "_" + std::to_string(i)] = 0;
      }
    }

    return stateSpace.stateIndex(state_factors);
  }

 private:
  std::string coord2str(int64_t i, int64_t j) const {
    return std::to_string(i) + "_" + std::to_string(j);
  }

  std::string ship2str(int64_t sz, int64_t rep) const {
    return std::to_string(sz) + "_" + std::to_string(rep);
  }

  std::vector<std::string> initActions() const {
    std::vector<std::string> acts;
    for (int64_t i = 0; i < grid_size; ++i)
      for (int64_t j = 0; j < grid_size; ++j) acts.push_back(coord2str(i, j));
    return acts;
  }

  std::pair<int64_t, int64_t> decompose_action(int64_t aI) const {
    return {aI / grid_size, aI % grid_size};
  }

  StateSpace initStateSpace() const {
    std::map<std::string, std::vector<int64_t>> state_factors;
    std::vector<int64_t> grid_coords;
    for (int64_t i = 0; i < grid_size; ++i) grid_coords.push_back(i);
    // ship location and orientation
    std::cerr << ship_sizes.size() << " " << ship_count_multiplier << std::endl;
    for (const auto& ship_sz : ship_sizes) {
      for (int64_t n = 0; n < ship_count_multiplier; ++n) {
        state_factors[ship2str(ship_sz, n) + "_row"] = grid_coords;
        state_factors[ship2str(ship_sz, n) + "_col"] = grid_coords;
        state_factors[ship2str(ship_sz, n) + "_ori"] = {0, 1};
        for (int i = 0; i < ship_sz; ++i) {
          state_factors[ship2str(ship_sz, n) + "_" + std::to_string(i)] = {
              0, 1};  // unhit, hit
        }
      }
    }

    StateSpace ss(state_factors);
    std::cout << "State space size: " << ss.size() << std::endl;
    return ss;
  }

  bool ship_present(int64_t sI, int ship_sz, int64_t ship_n, int64_t row,
                    int64_t col) const {
    const int64_t R =
        stateSpace.getStateFactorElem(sI, ship2str(ship_sz, ship_n) + "_row");
    const int64_t C =
        stateSpace.getStateFactorElem(sI, ship2str(ship_sz, ship_n) + "_col");
    const bool is_horiz =
        stateSpace.getStateFactorElem(sI, ship2str(ship_sz, ship_n) + "_ori");

    if (is_horiz) {
      return (R == row && C <= col && C + ship_sz - 1 >= col);
    } else {
      return (C == col && R <= row && R + ship_sz - 1 >= row);
    }
  }

  int64_t hit_ship(int64_t sI, int ship_sz, int64_t ship_n, int64_t row,
                   int64_t col) const {
    const int64_t R =
        stateSpace.getStateFactorElem(sI, ship2str(ship_sz, ship_n) + "_row");
    const int64_t C =
        stateSpace.getStateFactorElem(sI, ship2str(ship_sz, ship_n) + "_col");
    const bool is_horiz =
        stateSpace.getStateFactorElem(sI, ship2str(ship_sz, ship_n) + "_ori");

    const int64_t hit_no = (is_horiz) ? col - C : row - R;
    if (hit_no < 0 || hit_no >= ship_sz)
      throw std::runtime_error("Invalid hit");
    return stateSpace.updateStateFactor(
        sI, ship2str(ship_sz, ship_n) + "_" + std::to_string(hit_no), 1);
  }

  std::vector<std::string> initObs() const {
    std::vector<std::string> obs = {"miss", "hit"};
    return obs;
  }

  int64_t observeState(int64_t sI, int64_t aI) const {
    const auto [row, col] = decompose_action(aI);
    for (const auto& ship_sz : ship_sizes) {
      for (int64_t n = 0; n < ship_count_multiplier; ++n) {
        if (ship_present(sI, ship_sz, n, row, col)) return 1;
      }
    }
    return 0;
  }

  double applyActionToState(int64_t sI, int64_t aI, int64_t& sNext) const {
    sNext = sI;
    const auto [row, col] = decompose_action(aI);
    for (const auto& ship_sz : ship_sizes) {
      for (int64_t n = 0; n < ship_count_multiplier; ++n) {
        if (ship_present(sI, ship_sz, n, row, col)) {
          sNext = hit_ship(sI, ship_sz, n, row, col);
          if (sNext == sI) return _bad_action_reward;  // already hit
          return _hit_reward;
        }
      }
    }

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

  std::tuple<int64_t, int64_t, bool> attempt_to_place_ship(
      std::map<std::string, int64_t>& grid, int ship_size) {
    const int max_attempts = 100;
    for (int i = 0; i < max_attempts; ++i) {
      const auto [row, col, direction] = generate_random_position();
      if (valid_ship_placement(grid, row, col, ship_size, direction)) {
        place_ship(grid, row, col, ship_size, direction);
        return {row, col, direction};
      }
    }
    return {-1, -1, 0};  // failure
  }

  // Find a legal arrangement of ships for a square grid of size `grid_size`
  // Ship sizes are 2, 3, 4 and 5, and the number of each type is given by
  // `ship_count_multiplier`
  std::vector<std::tuple<int, int, int64_t, int64_t, bool>>
  generateBattleshipConfig() {
    std::vector<std::tuple<int, int, int64_t, int64_t, bool>>
        ship_locs;  // ship_sz, n, row, col, is_horiz

    std::map<std::string, int64_t> grid;
    for (int i = 0; i < grid_size; ++i)
      for (int j = 0; j < grid_size; ++j) grid[coord2str(i, j)] = 0;

    for (const auto& ship_sz : ship_sizes) {
      for (int n = 0; n < ship_count_multiplier; ++n) {
        const auto ship_loc = attempt_to_place_ship(grid, ship_sz);
        if (std::get<0>(ship_loc) < 0) {
          throw std::runtime_error(
              "Failed to place all ships, please try again with different "
              "parameters or a larger grid.");
        }
        ship_locs.push_back({ship_sz, n, std::get<0>(ship_loc),
                             std::get<1>(ship_loc), std::get<2>(ship_loc)});
      }
    }

    return ship_locs;
  }
};
