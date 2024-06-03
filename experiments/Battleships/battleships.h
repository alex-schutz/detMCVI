#pragma once

#include <cassert>
#include <random>

#include "SimInterface.h"

class Battleships : public MCVI::SimInterface {
 private:
  int64_t grid_size;
  std::vector<std::string> actions;
  std::vector<std::string> observations;

  int ship_count_multiplier;
  std::vector<int> ship_sizes = {2, 3, 4, 5};
  std::map<std::string, size_t> state_factor_sizes;

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
        state_factor_sizes(initStateSpace()),
        rng(rng) {}

  int64_t GetSizeOfObs() const override { return observations.size(); }
  int64_t GetSizeOfA() const override { return actions.size(); }
  double GetDiscount() const override { return 1.0; }
  int64_t GetNbAgent() const override { return 1; }
  const std::vector<std::string>& getActions() const { return actions; }
  const std::vector<std::string>& getObs() const { return observations; }
  bool IsTerminal(const MCVI::State& sI) const override {
    for (const auto& ship_sz : ship_sizes) {
      for (int64_t n = 0; n < ship_count_multiplier; ++n) {
        for (int i = 0; i < ship_sz; ++i) {
          if (sI.at(sfIdx(ship2str(ship_sz, n) + "_" + std::to_string(i))) == 0)
            return false;
        }
      }
    }
    return true;
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

    return names2state(state_factors);
  }

 private:
  std::string coord2str(int64_t i, int64_t j) const {
    return std::to_string(i) + "_" + std::to_string(j);
  }

  std::string ship2str(int64_t sz, int64_t rep) const {
    return std::to_string(sz) + "_" + std::to_string(rep);
  }

  MCVI::State names2state(const std::map<std::string, int64_t>& names) const {
    assert(names.size() == state_factor_sizes.size());
    std::vector<int64_t> state;
    for (const auto& [name, state_elem] : names) {
      const auto sf_sz = state_factor_sizes.find(name);
      assert(sf_sz != state_factor_sizes.cend());
      assert(sf_sz->second > state_elem);
      state.push_back(state_elem);
    }
    return state;
  }

  int64_t sfIdx(const std::string& state_factor) const {
    const auto sf_sz = state_factor_sizes.find(state_factor);
    assert(sf_sz != state_factor_sizes.cend());
    return (int64_t)std::distance(state_factor_sizes.cbegin(), sf_sz);
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

  std::map<std::string, size_t> initStateSpace() const {
    std::map<std::string, size_t> state_factors;
    std::vector<int64_t> grid_coords;
    for (int64_t i = 0; i < grid_size; ++i) grid_coords.push_back(i);
    // ship location and orientation
    for (const auto& ship_sz : ship_sizes) {
      for (int64_t n = 0; n < ship_count_multiplier; ++n) {
        state_factors[ship2str(ship_sz, n) + "_row"] = grid_coords.size();
        state_factors[ship2str(ship_sz, n) + "_col"] = grid_coords.size();
        state_factors[ship2str(ship_sz, n) + "_ori"] =
            2;  // vertical, horizontal
        for (int i = 0; i < ship_sz; ++i) {
          state_factors[ship2str(ship_sz, n) + "_" + std::to_string(i)] =
              2;  // unhit, hit
        }
      }
    }

    size_t p = 1;
    for (const auto& [sf, sz] : state_factors) p *= sz;
    std::cout << "State space size: " << p << std::endl;

    return state_factors;
  }

  bool ship_present(const MCVI::State& sI, int ship_sz, int64_t ship_n,
                    int64_t row, int64_t col) const {
    const int64_t R = sI.at(sfIdx(ship2str(ship_sz, ship_n) + "_row"));
    const int64_t C = sI.at(sfIdx(ship2str(ship_sz, ship_n) + "_col"));
    const bool is_horiz = sI.at(sfIdx(ship2str(ship_sz, ship_n) + "_ori"));

    if (is_horiz) {
      return (R == row && C <= col && C + ship_sz - 1 >= col);
    } else {
      return (C == col && R <= row && R + ship_sz - 1 >= row);
    }
  }

  MCVI::State hit_ship(const MCVI::State& sI, int ship_sz, int64_t ship_n,
                       int64_t row, int64_t col) const {
    const int64_t R = sI.at(sfIdx(ship2str(ship_sz, ship_n) + "_row"));
    const int64_t C = sI.at(sfIdx(ship2str(ship_sz, ship_n) + "_col"));
    const bool is_horiz = sI.at(sfIdx(ship2str(ship_sz, ship_n) + "_ori"));

    const int64_t hit_no = (is_horiz) ? col - C : row - R;
    if (hit_no < 0 || hit_no >= ship_sz)
      throw std::runtime_error("Invalid hit");
    MCVI::State out_state = sI;
    out_state[sfIdx(ship2str(ship_sz, ship_n) + "_" + std::to_string(hit_no))] =
        1;
    return out_state;
  }

  std::vector<std::string> initObs() const {
    std::vector<std::string> obs = {"miss", "hit"};
    return obs;
  }

  int64_t observeState(const MCVI::State& sI, int64_t aI) const {
    const auto [row, col] = decompose_action(aI);
    for (const auto& ship_sz : ship_sizes) {
      for (int64_t n = 0; n < ship_count_multiplier; ++n) {
        if (ship_present(sI, ship_sz, n, row, col)) return 1;
      }
    }
    return 0;
  }

  double applyActionToState(const MCVI::State& sI, int64_t aI,
                            MCVI::State& sNext) const {
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
                  int64_t col) const {
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
                            bool is_horizontal) const {
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
                  int64_t col, int ship_size, bool is_horizontal) const {
    if (is_horizontal) {
      for (int i = 0; i < ship_size; ++i) grid[coord2str(row, col + i)] = 1;
    } else {  // vertical
      for (int i = 0; i < ship_size; ++i) grid[coord2str(row + i, col)] = 1;
    }
  }

  std::tuple<int64_t, int64_t, bool> generate_random_position() const {
    static std::uniform_int_distribution<int64_t> unif(0, grid_size - 1);
    static std::uniform_int_distribution<int64_t> coin(0, 1);
    return {unif(rng), unif(rng), coin(rng)};
  }

  std::tuple<int64_t, int64_t, bool> attempt_to_place_ship(
      std::map<std::string, int64_t>& grid, int ship_size) const {
    const auto [row, col, direction] = generate_random_position();
    if (valid_ship_placement(grid, row, col, ship_size, direction)) {
      place_ship(grid, row, col, ship_size, direction);
      return {row, col, direction};
    }
    return {-1, -1, 0};  // failure
  }

  bool generate_config(
      std::vector<std::tuple<int, int, int64_t, int64_t, bool>>& config) const {
    std::map<std::string, int64_t> grid;
    for (int i = 0; i < grid_size; ++i)
      for (int j = 0; j < grid_size; ++j) grid[coord2str(i, j)] = 0;

    for (const auto& ship_sz : ship_sizes) {
      for (int n = 0; n < ship_count_multiplier; ++n) {
        const auto ship_loc = attempt_to_place_ship(grid, ship_sz);
        if (std::get<0>(ship_loc) < 0) return false;
        config.push_back({ship_sz, n, std::get<0>(ship_loc),
                          std::get<1>(ship_loc), std::get<2>(ship_loc)});
      }
    }
    return true;
  }

  // Find a legal arrangement of ships for a square grid of size `grid_size`
  // Ship sizes are 2, 3, 4 and 5, and the number of each type is given by
  // `ship_count_multiplier`
  std::vector<std::tuple<int, int, int64_t, int64_t, bool>>
  generateBattleshipConfig() {
    std::vector<std::tuple<int, int, int64_t, int64_t, bool>>
        ship_locs;  // ship_sz, n, row, col, is_horiz

    const int max_attempts = 100;
    for (int i = 0; i < max_attempts; ++i) {
      if (generate_config(ship_locs))
        return ship_locs;
      else
        ship_locs.clear();
    }
    throw std::runtime_error(
        "Failed to place all ships, please try again with different "
        "parameters or a larger grid.");
  }
};
