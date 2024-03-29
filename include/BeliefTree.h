/* This file has been written and/or modified by the following people:
 *
 * Yang You
 * Alex Schutz
 *
 */

#pragma once

#include <memory>
#include <unordered_map>

#include "BeliefParticles.h"
#include "Bound.h"
#include "QLearning.h"

namespace MCVI {

class BeliefTreeNode {
 private:
  using BeliefEdgeMap = std::unordered_map<
      int64_t, std::unordered_map<int64_t, std::shared_ptr<BeliefTreeNode>>>;

  BeliefParticles _state_particles;
  BeliefEdgeMap _child_nodes;
  int64_t _best_action;
  double _upper_bound;
  double _lower_bound;
  int64_t _fsc_node_index;

 public:
  BeliefTreeNode(const BeliefParticles& belief, int64_t best_action,
                 double upper_bound, double lower_bound)
      : _state_particles(belief),
        _best_action(best_action),
        _upper_bound(upper_bound),
        _lower_bound(lower_bound),
        _fsc_node_index(-1) {}

  void AddChild(int64_t action, int64_t observation,
                std::shared_ptr<BeliefTreeNode> child);

  const BeliefParticles& GetParticles() const { return _state_particles; }

  int64_t GetFSCNodeIndex() const { return _fsc_node_index; }
  void SetFSCNodeIndex(int64_t idx) { _fsc_node_index = idx; }

  int64_t GetBestAction() const { return _best_action; }
  void SetBestAction(int64_t action, double lower_bound);

  double GetUpper() const { return _upper_bound; }
  double GetLower() const { return _lower_bound; }

  std::shared_ptr<BeliefTreeNode> GetChild(int64_t action, int64_t observation);
};

/// @brief Add a child belief to the parent given an action and observation edge
void CreateBeliefTreeNode(std::shared_ptr<BeliefTreeNode> parent,
                          int64_t action, int64_t observation,
                          const BeliefParticles& belief,
                          const std::vector<int64_t>& action_space,
                          QLearningPolicy policy, SimInterface* sim);

std::shared_ptr<BeliefTreeNode> CreateBeliefRootNode(
    const BeliefParticles& belief, const std::vector<int64_t>& action_space,
    QLearningPolicy policy, SimInterface* sim);

/// @brief Sample beliefs from a belief tree with heuristics
void SampleBeliefs(
    std::shared_ptr<BeliefTreeNode> node, int64_t state, int64_t depth,
    int64_t max_depth, int64_t nb_sim, const std::vector<int64_t>& action_space,
    SimInterface* pomdp, QLearningPolicy policy,
    std::vector<std::shared_ptr<BeliefTreeNode>>& traversal_list);

/// @brief Generate a set of next beliefs mapped by observation, obtained by
/// taking `action` in belief node `node`. Return the most probable observation
/// and the next beliefs
std::pair<int64_t, std::unordered_map<int64_t, BeliefParticles>> BeliefUpdate(
    std::shared_ptr<BeliefTreeNode> node, int64_t action, int64_t nb_sim,
    SimInterface* pomdp);

}  // namespace MCVI
