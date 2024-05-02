/* This file has been written and/or modified by the following people:
 *
 * Alex Schutz
 *
 */

#pragma once

#include <deque>

#include "BeliefTree.h"

namespace MCVI {

// Build a greedy policy tree based on the value of each belief node
void BuildPolicyGraph(std::shared_ptr<BeliefTreeNode> initial_belief) {}

void RunAOStar(std::shared_ptr<BeliefTreeNode> initial_belief) {
  std::deque<std::shared_ptr<BeliefTreeNode>> fringe;
  fringe.emplace_back(initial_belief);

  while (!fringe.empty()) {
    const auto node = fringe.back();
    fringe.pop_back();

    for (int64_t a = 0; a < num_actions; ++a) {
      node->AddChild(a, heuristic, eval_depth, eval_epsilon, pomdp);
      for (const auto& c : node->GetChildren(a)) fringe.emplace_front(c);
    }

    // TODO: find a way of reconstructing the set of nodes that lead to node
    for (const std::shared_ptr<BeliefTreeNode>& n : policy_tree) {
      n->BackUpActions();
    }
  }

  return BuildPolicyGraph(initial_belief);
}

// TODO: write greedy simulator based on belief node upper bounds
// run alongside MCVI to evaluate performance

}  // namespace MCVI
