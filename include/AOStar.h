/* This file has been written and/or modified by the following people:
 *
 * Alex Schutz
 *
 */

#pragma once

#include <deque>

#include "BeliefTree.h"

namespace MCVI {

void RunAOStar(std::shared_ptr<BeliefTreeNode> initial_belief, int64_t max_iter,
               const PathToTerminal& heuristic, int64_t eval_depth,
               double eval_epsilon, SimInterface* pomdp) {
  for (int64_t a = 0; a < pomdp->GetSizeOfA(); ++a)
    initial_belief->GetOrAddChildren(a, heuristic, eval_depth, eval_epsilon,
                                     pomdp);
  initial_belief->UpdateBestAction();

  int64_t iter = 0;
  while (++iter <= max_iter) {
    std::vector<std::shared_ptr<BeliefTreeNode>> traversal_list = {
        initial_belief};
    std::vector<std::shared_ptr<BeliefTreeNode>> to_expand;

    size_t i = 0;
    while (i < traversal_list.size()) {
      const auto belief_node = traversal_list[i];
      for (const auto& [obs, obsNode] :
           belief_node->GetChildren(belief_node->GetBestActUBound())) {
        if (obsNode.GetBelief()->GetBestActUBound() == -1) {  // Leaf node
          to_expand.push_back(obsNode.GetBelief());
          continue;
        }
        traversal_list.push_back(obsNode.GetBelief());
      }
      ++i;
    }

    if (to_expand.empty()) return;

    std::vector<std::shared_ptr<BeliefTreeNode>> backup_list = traversal_list;
    for (const auto& node : to_expand) {
      for (int64_t a = 0; a < pomdp->GetSizeOfA(); ++a)
        node->GetOrAddChildren(a, heuristic, eval_depth, eval_epsilon, pomdp);
      backup_list.push_back(node);
    }

    for (auto it = backup_list.rbegin(); it < backup_list.rend(); ++it) {
      (*it)->BackUpBestActionUpperNoFSC();
      (*it)->UpdateBestAction();
    }
  }
}

}  // namespace MCVI
