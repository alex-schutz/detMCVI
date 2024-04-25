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
#include <sstream>
#include <string>

#include "PomdpInterface.h"

namespace MCVI {

class SimInterface {
 private:
  /* data */
 public:
  SimInterface(){};
  virtual ~SimInterface(){};

  // ------- obligate functions ----------
  virtual std::tuple<int64_t, int64_t, double, bool> Step(
      int64_t sI, int64_t aI) = 0;  // sI_next, oI, Reward, Done
  virtual int64_t SampleStartState() = 0;
  virtual int64_t GetSizeOfObs() const = 0;
  virtual int64_t GetSizeOfA() const = 0;
  virtual double GetDiscount() const = 0;
  virtual int64_t GetNbAgent() const = 0;
  virtual bool IsTerminal(int64_t sI) const = 0;

  // --------------------------------------------------------

  /// @brief Return an action chosen randomly
  virtual int64_t RandomAction() const {
    std::mt19937_64 rng;
    std::uniform_int_distribution<> action_dist(0, GetSizeOfA() - 1);
    return action_dist(rng);
  }

  // Maybe add visulization functions? :)
};

}  // namespace MCVI
