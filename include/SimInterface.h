/* This file has been written and/or modified by the following people:
 *
 * Yang You
 * Alex Schutz
 *
 */

#pragma once

#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <string>

#include "StateVector.h"

namespace MCVI {

class SimInterface {
 private:
  /* data */
 public:
  SimInterface(){};
  virtual ~SimInterface(){};

  // ------- obligate functions ----------
  virtual std::tuple<State, int64_t, double, bool> Step(
      const State& sI,
      int64_t aI) = 0;  // sI_next, oI, Reward, Done
  virtual State SampleStartState() = 0;
  virtual int64_t GetSizeOfObs() const = 0;
  virtual int64_t GetSizeOfA() const = 0;
  virtual double GetDiscount() const = 0;
  virtual int64_t GetNbAgent() const = 0;
  virtual bool IsTerminal(const State& sI) const = 0;

  // --------------------------------------------------------

  virtual std::optional<double> GetHeuristic(
      const MCVI::StateMap<double>& /*belief*/, int64_t /*max_depth*/) const {
    return std::nullopt;
  }

  // Maybe add visulization functions? :)
};

}  // namespace MCVI
