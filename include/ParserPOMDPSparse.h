/* This file has been written and/or modified by the following people:
 *
 * Yang You
 * 
 */

#ifndef _PARSERPOMDPSPARSE_H_
#define _PARSERPOMDPSPARSE_H_ 1

#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <sstream>
#include "PomdpInterface.h"
using namespace std;

class ParsedPOMDPSparse : public PomdpInterface
{
private:
    // set of states
    vector<string> States;
    int S_size;

    // set of actions
    vector<string> Actions;
    int A_size;

    // set of observations
    vector<string> Observations;
    int Obs_size;

    // initial belief
    vector<double> b0;

    map<int, double> b0_sparse;

    // transition function as A -> S -> P(S)
    vector<vector<map<int, double>>> TransFuncVecs;

    // observation as A -> S' -> O -> proba
    vector<vector<map<int, double>>> ObsFuncVecs;

    // reward function as A -> S -> reward
    vector<vector<double>> RewardFuncVecs;

    // discount factor
    double discount;

public:
    // builds a POMDP from a file
    ParsedPOMDPSparse(const string filename);
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
    const std::vector<string> &GetAllStates() const;
    const std::vector<string> &GetAllActions() const;
    const std::vector<string> &GetAllObservations() const;
    // for sparse representation
    const map<int, double> *GetTransProbDist(int sI, int aI) const;
    const map<int, double> *GetObsFuncProbDist(int s_newI, int aI) const;
    const map<int, double> *GetInitBeliefSparse() const;
};

#endif
