#pragma once

#include <algorithm>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "Cache.h"
#include "ShortestPath.h"
#include "SimInterface.h"

#define USE_HEURISTIC_BOUNDS 1

class Maze : public MCVI::SimInterface,
             public MCVI::ShortestPathFasterAlgorithm {
 private:
  std::vector<std::string> _maze;
  std::vector<std::string> actions = {"up", "right", "down", "left"};
  std::vector<std::string> observations;  // observe the surrounding walls

  std::mt19937_64& rng;

  double _success_reward = 100;
  double _move_reward = -1;

  int64_t state_space_sz;
  mutable MCVI::LRUCache<MCVI::State, double, MCVI::StateHash, MCVI::StateEqual>
      state_value_cache;

 public:
  Maze(std::vector<std::string> maze, std::mt19937_64& rng)
      : _maze(maze),
        observations(initObs()),
        rng(rng),
        state_space_sz(countBlankSpaces(_maze) + 1),
        state_value_cache(250000) {}

  int64_t GetSizeOfObs() const override { return observations.size(); }
  int64_t GetSizeOfA() const override { return actions.size(); }
  double GetDiscount() const override { return 1.0; }
  int64_t GetNbAgent() const override { return 1; }
  const std::vector<std::string>& getActions() const { return actions; }
  const std::vector<std::string>& getObs() const { return observations; }
  bool IsTerminal(const MCVI::State& sI) const override { return sI[0] == 0; }

  void drawState(const MCVI::State& sI) const {
    const auto m = indexToPlayerLocation(_maze, sI[0]);
    return printMaze(m);
  }

#if (USE_HEURISTIC_BOUNDS == 1)
  std::optional<double> GetHeuristicUpper(const MCVI::StateMap<double>& belief,
                                          int64_t max_depth) const override {
    return heuristicUpper(belief, max_depth);
  }
#endif

  std::pair<double, bool> get_state_value(const MCVI::State& state,
                                          int64_t max_depth) const {
    const auto f = state_value_cache.find(state);
    if (f != state_value_cache.cend()) return {f->second.first, true};
    const auto b = bestPath(state, max_depth);
    if (b.second) state_value_cache.put(state, b.first);
    return b;
  }

  std::tuple<MCVI::State, int64_t, double, bool> Step(const MCVI::State& sI,
                                                      int64_t aI) override {
    MCVI::State sNext;
    const double reward = applyActionToState(sI, aI, sNext);
    const int64_t oI = observeState(sNext);
    const bool finished = IsTerminal(sNext);
    // sI_next, oI, Reward, Done
    return std::tuple<MCVI::State, int64_t, double, bool>(sNext, oI, reward,
                                                          finished);
  }

  MCVI::State SampleStartState() override {
    // Start in any available position other than the goal
    static std::uniform_int_distribution<int64_t> ss_dist(1,
                                                          state_space_sz - 1);
    return {ss_dist(rng)};
  }

  double applyActionToState(const MCVI::State& sI, int64_t aI,
                            MCVI::State& sNext) const {
    sNext = sI;
    if (IsTerminal(sI)) return 0;
    const auto curr_maze = indexToPlayerLocation(_maze, sI[0]);
    const std::pair<int64_t, int64_t> curr_loc = findPlayerLocation(curr_maze);
    if (curr_loc.first == -1 || curr_loc.second == -1)
      throw std::logic_error("Cannot find player location for state " +
                             std::to_string(sI[0]));

    auto next_maze = curr_maze;
    const std::string action = actions[aI];
    if (action == "up") {
      if (curr_loc.first > 0 &&
          (curr_maze[curr_loc.first - 1][curr_loc.second] == ' ' ||
           curr_maze[curr_loc.first - 1][curr_loc.second] == 'G')) {
        next_maze[curr_loc.first - 1][curr_loc.second] = '*';
        next_maze[curr_loc.first][curr_loc.second] = ' ';
      }
    } else if (action == "down") {
      if (curr_loc.first < (int64_t)_maze.size() - 1 &&
          (curr_maze[curr_loc.first + 1][curr_loc.second] == ' ' ||
           curr_maze[curr_loc.first + 1][curr_loc.second] == 'G')) {
        next_maze[curr_loc.first + 1][curr_loc.second] = '*';
        next_maze[curr_loc.first][curr_loc.second] = ' ';
      }
    } else if (action == "left") {
      if (curr_loc.second > 0 &&
          (curr_maze[curr_loc.first][curr_loc.second - 1] == ' ' ||
           curr_maze[curr_loc.first][curr_loc.second - 1] == 'G')) {
        next_maze[curr_loc.first][curr_loc.second - 1] = '*';
        next_maze[curr_loc.first][curr_loc.second] = ' ';
      }
    } else if (action == "right") {
      if (curr_loc.second < (int64_t)_maze[0].size() - 1 &&
          (curr_maze[curr_loc.first][curr_loc.second + 1] == ' ' ||
           curr_maze[curr_loc.first][curr_loc.second + 1] == 'G')) {
        next_maze[curr_loc.first][curr_loc.second + 1] = '*';
        next_maze[curr_loc.first][curr_loc.second] = ' ';
      }
    }

    // Player location is same as goal
    if (findGoalLocation(next_maze).first == -1) {
      sNext = {0};
      return _success_reward;
    }

    sNext = {playerLocationToIndex(next_maze)};
    return _move_reward;
  }

 private:
  void printMaze(const std::vector<std::string>& maze) const {
    for (const std::string& row : maze) {
      std::cout << row << std::endl;
    }
  }

  std::pair<int64_t, int64_t> findPlayerLocation(
      const std::vector<std::string>& maze) const {
    for (int64_t r = 0; r < (int64_t)maze.size(); ++r) {
      size_t col = maze[r].find('*');
      if (col != std::string::npos) {
        return {r, static_cast<int64_t>(col)};
      }
    }
    return {-1, -1};  // not found
  }

  std::pair<int64_t, int64_t> findGoalLocation(
      const std::vector<std::string>& maze) const {
    for (int64_t r = 0; r < (int64_t)maze.size(); ++r) {
      size_t col = maze[r].find('G');
      if (col != std::string::npos) {
        return {r, static_cast<int64_t>(col)};
      }
    }
    return {-1, -1};  // not found
  }

  int64_t playerLocationToIndex(const std::vector<std::string>& maze) const {
    if (findPlayerLocation(maze).first == -1) return -1;
    if (findGoalLocation(maze).first == -1) return 0;

    int64_t index = 1;  // Start at 1 since G is index 0

    for (int64_t r = 0; r < (int64_t)maze.size(); ++r) {
      for (int64_t c = 0; c < (int64_t)maze[r].size(); ++c) {
        if (maze[r][c] == ' ' || maze[r][c] == '*') {
          if (maze[r][c] == '*') return index;
          ++index;
        }
      }
    }

    return -1;  // No player location is found
  }

  std::vector<std::string> indexToPlayerLocation(
      const std::vector<std::string>& maze, int64_t index) const {
    std::vector<std::string> blankMaze = maze;

    if (index == 0) {  // player is at goal location
      const auto goal_loc = findGoalLocation(maze);
      if (goal_loc.first == -1 || goal_loc.second == -1)
        return blankMaze;  // bad
      blankMaze[goal_loc.first][goal_loc.second] = '*';
      return blankMaze;
    }

    int64_t currentIndex = 1;
    for (int64_t r = 0; r < (int64_t)blankMaze.size(); ++r) {
      for (int64_t c = 0; c < (int64_t)blankMaze[r].size(); ++c) {
        if (blankMaze[r][c] == ' ') {
          if (index == currentIndex) {
            blankMaze[r][c] = '*';
            return blankMaze;
          }
          ++currentIndex;
        }
      }
    }

    return blankMaze;
  }

  static inline bool isBitSet(int64_t num, size_t bit) {
    return 1 == ((num >> bit) & 1);
  }

  std::vector<std::string> initObs() const {
    std::vector<std::string> obs_components = {"above", "right", "below",
                                               "left"};

    std::vector<std::string> obs;
    for (int64_t b = 0; b < (1 << obs_components.size()); ++b) {
      std::string o = "*";
      for (size_t i = 0; i < obs_components.size(); ++i)
        if (isBitSet(b, i)) o += obs_components[i];
      obs.push_back(o);
    }
    return obs;
  }

  int64_t observeState(const MCVI::State& sI) const {
    std::string obs = "*";
    const auto curr_maze = indexToPlayerLocation(_maze, sI[0]);
    const std::pair<int64_t, int64_t> curr_loc = findPlayerLocation(curr_maze);
    if (curr_loc.first == -1 || curr_loc.second == -1)
      throw std::logic_error("Cannot find player location for state " +
                             std::to_string(sI[0]));
    if (curr_loc.first > 0 &&
        curr_maze[curr_loc.first - 1][curr_loc.second] != ' ' &&
        curr_maze[curr_loc.first - 1][curr_loc.second] != 'G')
      obs += "above";
    if (curr_loc.second < (int64_t)_maze[0].size() - 1 &&
        curr_maze[curr_loc.first][curr_loc.second + 1] != ' ' &&
        curr_maze[curr_loc.first][curr_loc.second + 1] != 'G')
      obs += "right";
    if (curr_loc.first < (int64_t)_maze.size() - 1 &&
        curr_maze[curr_loc.first + 1][curr_loc.second] != ' ' &&
        curr_maze[curr_loc.first + 1][curr_loc.second] != 'G')
      obs += "below";
    if (curr_loc.second > 0 &&
        curr_maze[curr_loc.first][curr_loc.second - 1] != ' ' &&
        curr_maze[curr_loc.first][curr_loc.second - 1] != 'G')
      obs += "left";

    return (int64_t)std::distance(
        observations.cbegin(),
        std::find(observations.cbegin(), observations.cend(), obs));
  }

  double heuristicUpper(const MCVI::StateMap<double>& belief,
                        int64_t max_depth) const {
    double val = 0;
    for (const auto& [s, p] : belief) {
      val += get_state_value(s, max_depth).first * p;
    }
    return val;
  }

 public:
  std::vector<std::tuple<MCVI::State, double, int64_t>> getEdges(
      const MCVI::State& state) const {
    if (IsTerminal(state)) return {};
    std::vector<std::tuple<MCVI::State, double, int64_t>> successors;
    for (int64_t a = 0; a < GetSizeOfA(); ++a) {
      MCVI::State sNext;
      const auto& reward = applyActionToState(state, a, sNext);
      successors.push_back({sNext, -reward, a});
    }
    return successors;
  }

 private:
  std::pair<double, bool> bestPath(const MCVI::State& state,
                                   int64_t max_depth) const {
    const auto [costs, predecessors] = calculate(state, max_depth);
    const auto path_to_goal = reconstructPath({0}, predecessors);
    const bool can_reach_goal =
        findGoalLocation(
            indexToPlayerLocation(_maze, path_to_goal.back().first[0]))
            .first == -1;

    if (!can_reach_goal) return {max_depth * _move_reward, false};

    return {-costs.at({0}), can_reach_goal};
  }

  int64_t countBlankSpaces(const std::vector<std::string>& maze) {
    int64_t count = 0;
    for (const auto& row : maze) {
      for (char ch : row) {
        if (ch == ' ') ++count;
      }
    }
    return count;
  }

 public:
  void toSARSOP(std::ostream& os) {
    const size_t num_states = state_space_sz;
    std::vector<MCVI::State> state_enum;
    for (int64_t s = 0; s < state_space_sz; ++s) state_enum.push_back({s});
    const double n = std::sqrt((num_states + 1) / 2);
    os << "discount: " << std::exp(std::log(0.01) / (4.0 * n * n)) << std::endl;
    os << "values: reward" << std::endl;
    os << "states: " << num_states << std::endl;
    os << "actions: " << GetSizeOfA() << std::endl;
    os << "observations: " << GetSizeOfObs() << std::endl << std::endl;

    // Initial belief
    os << "start: " << std::endl;
    os << 0 << " ";  // goal state
    double sum = 0.0;
    for (int64_t s = 1; s < state_space_sz; ++s) {
      const double target = s * 1.0 / (state_space_sz - 1);
      os << target - sum << " ";
      sum += target - sum;
    }
    os << std::endl << std::endl;

    // Transition probabilities  T : <action> : <start-state> : <end-state> %f
    // Observation probabilities O : <action> : <end-state> : <observation> %f
    // Reward     R: <action> : <start-state> : <end-state> : <observation> %f
    for (size_t sI = 0; sI < state_enum.size(); ++sI) {
      for (int64_t a = 0; a < GetSizeOfA(); ++a) {
        MCVI::State sNext;
        double reward = applyActionToState(state_enum[sI], a, sNext);
        if (IsTerminal(state_enum[sI])) {
          reward = 0;
          sNext = state_enum[sI];
        }
        const int64_t obs = observeState(state_enum[sI]);
        const size_t eI = std::distance(
            state_enum.begin(),
            std::find(state_enum.begin(), state_enum.end(), sNext));
        os << "T : " << a << " : " << sI << " : " << eI << " 1.0" << std::endl;
        os << "O : " << a << " : " << sI << " : " << obs << " 1.0" << std::endl;
        os << "R : " << a << " : " << sI << " : " << eI << " : " << obs << " "
           << reward << std::endl;
      }
    }
  }
};

void ReadMazeParams(const std::string& filename,
                    std::vector<std::string>& maze) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Unable to open file: " + filename);
  }

  std::string line;
  while (std::getline(file, line)) {
    maze.push_back(line);
  }

  file.close();
}
