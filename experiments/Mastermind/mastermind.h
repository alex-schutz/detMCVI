#pragma once

#include <cassert>
#include <random>

#include "SimInterface.h"
#include "state_space.h"

class Mastermind : public MCVI::SimInterface {
 private:
  int colour_count;
  int peg_count;
  StateSpace actions;
  StateSpace observations;

  std::mt19937_64& rng;
  std::map<std::string, size_t> state_factor_sizes;

  double _success_reward = 100;
  double _guess_reward = -10;

 public:
  Mastermind(int colour_count, int peg_count, std::mt19937_64& rng)
      : colour_count(colour_count),
        peg_count(peg_count),
        actions(initActions()),
        observations(initObs()),
        rng(rng),
        state_factor_sizes(initStateSpace()) {}

  int64_t GetSizeOfObs() const override { return observations.size(); }
  int64_t GetSizeOfA() const override { return actions.size(); }
  double GetDiscount() const override { return 1.0; }
  int64_t GetNbAgent() const override { return 1; }
  bool IsTerminal(const MCVI::State& sI) const override {
    return sI[sfIdx("success")] == 1;
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
    static std::uniform_int_distribution<int64_t> colour_dist(0,
                                                              colour_count - 1);

    std::map<std::string, int64_t> state_factors;

    state_factors["success"] = 0;
    for (int h = 0; h < peg_count; ++h) {
      state_factors[std::to_string(h)] = colour_dist(rng);
    }

    return names2state(state_factors);
  }

  double applyActionToState(const MCVI::State& sI, int64_t aI,
                            MCVI::State& sNext) const {
    sNext = sI;
    int correct_pegs = 0;
    const auto guess = actions.at(aI);
    for (int h = 0; h < peg_count; ++h) {
      const std::string key = std::to_string(h);
      if (guess.at(key) == sI[sfIdx(key)]) ++correct_pegs;
    }

    if (correct_pegs == peg_count) {
      sNext[sfIdx("success")] = 1;
      return _success_reward;
    }

    return _guess_reward;
  }

  int64_t sfIdx(const std::string& state_factor) const {
    const auto sf_sz = state_factor_sizes.find(state_factor);
    if (sf_sz == state_factor_sizes.cend())
      throw std::logic_error("Could not find state factor " + state_factor);
    return (int64_t)std::distance(state_factor_sizes.cbegin(), sf_sz);
  }

 private:
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

  StateSpace initActions() const {
    std::map<std::string, std::vector<int64_t>> guess_factors;
    std::vector<int64_t> v;
    for (int c = 0; c < colour_count; ++c) v.push_back(c);
    for (int h = 0; h < peg_count; ++h) guess_factors[std::to_string(h)] = v;
    return StateSpace(guess_factors);
  }

  StateSpace initObs() const {
    std::map<std::string, std::vector<int64_t>> obs_factors;
    std::vector<int64_t> v;
    for (int h = 0; h <= peg_count; ++h) v.push_back(h);
    obs_factors["correct_colour"] = v;
    obs_factors["correct_colour_and_loc"] = v;
    return StateSpace(obs_factors);
  }

  std::map<std::string, size_t> initStateSpace() const {
    std::map<std::string, size_t> state_factors;

    for (int h = 0; h < peg_count; ++h) {
      state_factors[std::to_string(h)] = colour_count;
    }
    state_factors["success"] = 2;  // false/true

    size_t p = 1;
    for (const auto& [sf, sz] : state_factors) p *= sz;
    std::cout << "State space size: " << p << std::endl;

    return state_factors;
  }

  int64_t observeState(const MCVI::State& sI, int64_t aI) const {
    // initialise observation
    std::map<std::string, int64_t> obs_factors;
    obs_factors["correct_colour"] = 0;
    obs_factors["correct_colour_and_loc"] = 0;

    const auto guess = actions.at(aI);
    std::map<int64_t, int> guess_counts;
    std::map<int64_t, int> secret_counts;

    // First pass: calculate red pegs and count remaining colors
    for (int h = 0; h < peg_count; ++h) {
      const std::string key = std::to_string(h);
      if (guess.at(key) == sI[sfIdx(key)]) {
        obs_factors["correct_colour_and_loc"] += 1;
      } else {
        guess_counts[guess.at(key)]++;
        secret_counts[sI[sfIdx(key)]]++;
      }
    }

    // Second pass: calculate white pegs
    for (const auto& [colour, count] : guess_counts) {
      const auto s = secret_counts.find(colour);
      if (s != secret_counts.end())
        obs_factors["correct_colour"] += std::min(count, s->second);
    }

    return observations.stateIndex(obs_factors);
  }

  std::vector<MCVI::State> enumerateStates(
      size_t max_size = std::numeric_limits<int64_t>::max()) const {
    std::vector<MCVI::State> enum_states;
    std::vector<size_t> sizes;
    for (const auto& pair : state_factor_sizes) sizes.push_back(pair.second);

    // Initialize a state vector with the first state (all zeros)
    MCVI::State current_state(state_factor_sizes.size(), 0);
    enum_states.push_back(current_state);

    // Generate all combinations of state factors
    while (true) {
      // Find the rightmost factor that can be incremented
      size_t factor_index = state_factor_sizes.size();
      while (factor_index > 0) {
        factor_index--;
        if (current_state[factor_index] + 1 < (int64_t)sizes[factor_index]) {
          current_state[factor_index]++;
          break;
        } else {
          // Reset this factor and carry over to the next factor
          current_state[factor_index] = 0;
        }
      }

      // If we completed a full cycle (all factors are zero again), we are done
      if (factor_index == 0 && current_state[0] == 0) {
        break;
      }
      // warn if overflow
      if (enum_states.size() >= max_size)
        throw std::runtime_error("Maximum size exceeded.");
      enum_states.push_back(current_state);
    }
    return enum_states;
  }

 public:
  void toSARSOP(std::ostream& os) {
    std::vector<MCVI::State> state_enum = enumerateStates();
    const size_t num_states = state_enum.size();
    os << "discount: " << std::exp(std::log(0.01) / (24)) << std::endl;
    os << "values: reward" << std::endl;
    os << "states: " << num_states << std::endl;
    os << "actions: " << GetSizeOfA() << std::endl;
    os << "observations: " << GetSizeOfObs() << std::endl << std::endl;

    // Initial belief
    os << "start include: ";
    for (size_t s = 0; s < num_states; ++s)
      if (state_enum.at(s)[sfIdx("success")] == 0) os << s << " ";
    os << std::endl;

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
        const int64_t obs = observeState(state_enum[sI], a);
        const size_t eI = std::distance(
            state_enum.begin(),
            std::find(state_enum.begin(), state_enum.end(), sNext));
        os << "T : " << a << " : " << sI << " : " << eI << " 1.0" << std::endl;
        os << "O : " << a << " : " << sI << " : " << obs << " 1.0" << std::endl;
        os << "R : " << a << " : " << sI << " : " << eI << " : * " << reward
           << std::endl;
      }
    }
  }
};

void ReadMastermindParams(const std::string& filename, int& colour_count,
                          int& peg_count) {
  std::ifstream file(filename);
  if (!file.is_open())
    throw std::runtime_error("Unable to open file: " + filename);
  if (!(file >> colour_count))
    throw std::runtime_error("Error reading the colour_count from file");
  if (!(file >> peg_count))
    throw std::runtime_error("Error reading the peg_count from file");
  file.close();
}
