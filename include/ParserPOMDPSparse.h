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
  int S_size;

  // set of actions
  std::vector<std::string> Actions;
  int A_size;

  // set of observations
  std::vector<std::string> Observations;
  int Obs_size;

  // initial belief
  std::vector<double> b0;

  std::map<int, double> b0_sparse;

  // transition function as A -> S -> P(S)
  std::vector<std::vector<std::map<int, double>>> TransFuncVecs;

  // observation as A -> S' -> O -> proba
  std::vector<std::vector<std::map<int, double>>> ObsFuncVecs;

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
  int GetSizeOfS() const;
  int GetSizeOfA() const;
  int GetSizeOfObs() const;
  double TransFunc(int sI, int aI, int s_newI) const;
  double ObsFunc(int oI, int s_newI, int aI) const;
  double Reward(int sI, int aI) const;
  const std::std::vector<std::string> &GetAllStates() const;
  const std::std::vector<std::string> &GetAllActions() const;
  const std::std::vector<std::string> &GetAllObservations() const;
  // for sparse representation
  const std::map<int, double> *GetTransProbDist(int sI, int aI) const;
  const std::map<int, double> *GetObsFuncProbDist(int s_newI, int aI) const;
  const std::map<int, double> *GetInitBeliefSparse() const;
};

}  // namespace MCVI
