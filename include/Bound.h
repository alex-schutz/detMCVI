/* This file has been written and/or modified by the following people:
 *
 * Yang You
 * Alex Schutz
 *
 */

#ifndef _BOUND_H_
#define _BOUND_H_

#include <unordered_map>

#include "PomdpInterface.h"
using namespace std;

using Belief = unordered_map<int, double>;

/** @brief Return an upper bound for the value of the belief.
 *
 * The upper bound value of the belief is the expected sum of each possible
 * state's MDP value
 *
 * @param belief A map from state indices to probabilities.
 * @param sim A POMDP simulator object
 * @return double
 */
double UpperBoundEvaluation(const Belief& belief, const PomdpInterface* sim);

#endif /* !_BOUND_H_ */
