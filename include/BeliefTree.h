/* This file has been written and/or modified by the following people:
 *
 * Yang You
 * Alex Schutz
 *
 */

#pragma once

#include <memory>
#include <unordered_map>

#include "BeliefDistribution.h"
#include "Bound.h"

namespace MCVI {

class BeliefTreeNode;

class ActionNode {
 private:
  int64_t _action;              // Action of this node
  double _avgUpper, _avgLower;  // Bounds based on observation children
  std::unordered_map<int64_t, std::shared_ptr<BeliefTreeNode>>
      _observation_edges;
  std::unordered_map<int64_t, double>
      _o_weights;  // store the weights of different observations

 public:
  ActionNode(int64_t action, const BeliefDistribution& belief,
             int64_t max_belief_samples, const PathToTerminal& heuristic,
             int64_t eval_depth, double eval_epsilon, SimInterface* pomdp);

  int64_t GetAction() const { return _action; }

  double GetAvgUpper() const { return _avgUpper; }
  double GetAvgLower() const { return _avgLower; }

  std::shared_ptr<BeliefTreeNode> GetChild(int64_t observation) const;
  std::unordered_map<int64_t, std::shared_ptr<BeliefTreeNode>> GetChildren()
      const {
    return _observation_edges;
  }

  std::shared_ptr<BeliefTreeNode> ChooseObservation(double target) const;

 private:
  /// @brief Generate a set of next beliefs mapped by observation,
  /// obtained by taking `action` in belief.
  void BeliefUpdate(const BeliefDistribution& belief,
                    int64_t max_belief_samples, const PathToTerminal& heuristic,
                    int64_t eval_depth, double eval_epsilon,
                    SimInterface* pomdp);

  void CalculateBounds();
};

class BeliefTreeNode {
 private:
  BeliefDistribution _belief;
  std::unordered_map<int64_t, ActionNode> _action_edges;
  int64_t _best_action;

  double _upper_bound;
  double _lower_bound;
  int64_t _fsc_node_index;

 public:
  BeliefTreeNode(const BeliefDistribution& belief, int64_t best_action,
                 double upper_bound, double lower_bound)
      : _belief(belief),
        _best_action(best_action),
        _upper_bound(upper_bound),
        _lower_bound(lower_bound),
        _fsc_node_index(-1) {}

  void AddChild(int64_t action, int64_t max_belief_samples,
                const PathToTerminal& heuristic, int64_t eval_depth,
                double eval_epsilon, SimInterface* pomdp);

  const BeliefDistribution& GetBelief() const { return _belief; }

  int64_t GetFSCNodeIndex() const { return _fsc_node_index; }
  void SetFSCNodeIndex(int64_t idx) { _fsc_node_index = idx; }

  int64_t GetBestAction() const { return _best_action; }
  void SetBestAction(int64_t action, double lower_bound);

  double GetUpper() const { return _upper_bound; }
  void SetUpper(double upper_bound) { _upper_bound = upper_bound; }
  double GetLower() const { return _lower_bound; }

  std::shared_ptr<BeliefTreeNode> GetChild(int64_t action,
                                           int64_t observation) const;
  std::unordered_map<int64_t, std::shared_ptr<BeliefTreeNode>> GetChildren(
      int64_t action) const;

  std::shared_ptr<BeliefTreeNode> ChooseObservation(
      double target, int64_t max_belief_samples,
      const PathToTerminal& heuristic, int64_t eval_depth, double eval_epsilon,
      SimInterface* pomdp);
};

std::shared_ptr<BeliefTreeNode> CreateBeliefTreeNode(
    const BeliefDistribution& belief, const PathToTerminal& heuristic,
    int64_t eval_depth, double eval_epsilon, SimInterface* sim);

}  // namespace MCVI
