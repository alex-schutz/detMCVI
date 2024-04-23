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
  virtual std::tuple<int, int, double, bool> Step(
      int sI, int aI) = 0;  // sI_next, oI, Reward, Done
  virtual int SampleStartState() = 0;
  virtual int GetSizeOfObs() const = 0;
  virtual int GetSizeOfA() const = 0;
  virtual double GetDiscount() const = 0;
  virtual int GetNbAgent() const = 0;
  virtual bool IsTerminal(int sI) const = 0;

  // --------------------------------------------------------

  /// @brief Return an action chosen randomly
  virtual int RandomAction() const {
    std::mt19937_64 rng;
    std::uniform_int_distribution<> action_dist(0, GetSizeOfA() - 1);
    return action_dist(rng);
  }

  // Maybe add visulization functions? :)
};

}  // namespace MCVI
