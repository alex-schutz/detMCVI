/* This file has been written and/or modified by the following people:
 *
 * Yang You
 * Alex Schutz
 *
 */

#pragma once

#include <memory>
#include <unordered_map>

#include "AlphaVectorFSC.h"
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
  double _immediate_reward;

 public:
  ActionNode(int64_t action, const BeliefDistribution& belief,
             int64_t max_belief_samples, const PathToTerminal& heuristic,
             int64_t eval_depth, double eval_epsilon, SimInterface* pomdp);

  int64_t GetAction() const { return _action; }

  double GetAvgUpper() const { return _avgUpper; }
  double GetAvgLower() const { return _avgLower; }

  double GetImmediateReward() const { return _immediate_reward; }

  double GetWeight(int64_t obs) const { return _o_weights.at(obs); }

  std::shared_ptr<BeliefTreeNode> GetChild(int64_t observation) const;
  std::unordered_map<int64_t, std::shared_ptr<BeliefTreeNode>> GetChildren()
      const {
    return _observation_edges;
  }

  std::shared_ptr<BeliefTreeNode> ChooseObservation(double target) const;

  void BackUp(AlphaVectorFSC& fsc, int64_t max_belief_samples, double R_lower,
              int64_t max_depth_sim, SimInterface* pomdp);

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
  int64_t _bestActUBound, _bestActLBound;

  double _upper_bound;
  double _lower_bound;
  int64_t _fsc_node_index;

  int64_t _best_policy_node;
  double _best_policy_val;

 public:
  BeliefTreeNode(const BeliefDistribution& belief, double upper_bound,
                 double lower_bound)
      : _belief(belief),
        _bestActUBound(-1),
        _bestActLBound(-1),
        _upper_bound(upper_bound),
        _lower_bound(lower_bound),
        _fsc_node_index(-1) {}

  void AddChild(int64_t action, int64_t max_belief_samples,
                const PathToTerminal& heuristic, int64_t eval_depth,
                double eval_epsilon, SimInterface* pomdp);
  const ActionNode& GetOrAddChildren(int64_t action, int64_t max_belief_samples,
                                     const PathToTerminal& heuristic,
                                     int64_t eval_depth, double eval_epsilon,
                                     SimInterface* pomdp);

  const BeliefDistribution& GetBelief() const { return _belief; }

  int64_t GetFSCNodeIndex() const { return _fsc_node_index; }
  void SetFSCNodeIndex(int64_t idx) { _fsc_node_index = idx; }

  void SetBestActionLBound(int64_t action, double lower_bound);

  int64_t GetBestPolicyNode() const { return _best_policy_node; }

  void UpdateBestAction();
  int64_t GetBestActLBound() const { return _bestActLBound; }
  int64_t GetBestActUBound() const { return _bestActUBound; }

  double GetUpper() const { return _upper_bound; }
  void SetUpper(double upper_bound) { _upper_bound = upper_bound; }
  double GetLower() const { return _lower_bound; }

  std::shared_ptr<BeliefTreeNode> GetChild(int64_t action,
                                           int64_t observation) const;
  std::unordered_map<int64_t, std::shared_ptr<BeliefTreeNode>> GetChildren(
      int64_t action) const;

  std::shared_ptr<BeliefTreeNode> ChooseObservation(double target);

  void BackUpActions(AlphaVectorFSC& fsc, int64_t max_belief_samples,
                     double R_lower, int64_t max_depth_sim,
                     SimInterface* pomdp);

  void BackUpFromPolicyGraph(AlphaVectorFSC& fsc, int64_t max_belief_samples,
                             double R_lower, int64_t max_depth_sim,
                             SimInterface* pomdp);
};

std::shared_ptr<BeliefTreeNode> CreateBeliefTreeNode(
    const BeliefDistribution& belief, const PathToTerminal& heuristic,
    int64_t eval_depth, double eval_epsilon, SimInterface* sim);

}  // namespace MCVI
