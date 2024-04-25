/* This file has been written and/or modified by the following people:
 *
 * Yang You
 * Alex Schutz
 *
 */

#pragma once

#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>

#include "PomdpInterface.h"

namespace MCVI {

class ParsedPOMDPSparse : public PomdpInterface {
 private:
  // set of states
  std::vector<std::string> States;
  int64_t S_size;

  // set of actions
  std::vector<std::string> Actions;
  int64_t A_size;

  // set of observations
  std::vector<std::string> Observations;
  int64_t Obs_size;

  // initial belief
  std::vector<double> b0;

  std::map<int64_t, double> b0_sparse;

  // transition function as A -> S -> P(S)
  std::vector<std::vector<std::map<int64_t, double>>> TransFuncVecs;

  // observation as A -> S' -> O -> proba
  std::vector<std::vector<std::map<int64_t, double>>> ObsFuncVecs;

  // reward function as A -> S -> reward
  std::vector<std::vector<double>> RewardFuncVecs;

  // discount factor
  double discount;

 public:
  // builds a POMDP from a file
  ParsedPOMDPSparse(const std::string filename);
  // destroys a POMDP
  ~ParsedPOMDPSparse();
  // get discount value
  double GetDiscount() const;
  int64_t GetSizeOfS() const;
  int64_t GetSizeOfA() const;
  int64_t GetSizeOfObs() const;
  double TransFunc(int64_t sI, int64_t aI, int64_t s_newI) const;
  double ObsFunc(int64_t oI, int64_t s_newI, int64_t aI) const;
  double Reward(int64_t sI, int64_t aI) const;
  const std::std::vector<std::string> &GetAllStates() const;
  const std::std::vector<std::string> &GetAllActions() const;
  const std::std::vector<std::string> &GetAllObservations() const;
  // for sparse representation
  const std::map<int64_t, double> *GetTransProbDist(int64_t sI,
                                                    int64_t aI) const;
  const std::map<int64_t, double> *GetObsFuncProbDist(int64_t s_newI,
                                                      int64_t aI) const;
  const std::map<int64_t, double> *GetInitBeliefSparse() const;
};

}  // namespace MCVI
