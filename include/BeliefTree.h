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

class ObservationNode {
 private:
  double _weight;
  double _sum_reward;
  double _upper_bound;
  double _lower_bound;
  int64_t _best_policy_node;
  double _best_policy_val;
  std::shared_ptr<BeliefTreeNode> _next_belief;

 public:
  ObservationNode(double weight, double sum_reward,
                  std::shared_ptr<BeliefTreeNode> next_belief,
                  double next_upper, double next_lower)
      : _weight(weight),
        _sum_reward(sum_reward),
        _upper_bound(sum_reward + next_upper),
        _lower_bound(sum_reward + next_lower),
        _best_policy_node(-1),
        _best_policy_val(-std::numeric_limits<double>::infinity()),
        _next_belief(next_belief) {}

  int64_t GetBestPolicyNode() const { return _best_policy_node; }

  void BackUp(AlphaVectorFSC& fsc, double R_lower, int64_t max_depth_sim,
              SimInterface* pomdp);

  void BackUpFromNextBelief();

  void BackUpFromPolicyGraph(AlphaVectorFSC& fsc, double R_lower,
                             int64_t max_depth_sim, SimInterface* pomdp);

  double GetWeight() const { return _weight; }
  double GetUpper() const { return _upper_bound; }
  double GetLower() const { return _lower_bound; }

  std::shared_ptr<BeliefTreeNode> GetBelief() const { return _next_belief; }
};

class ActionNode {
 private:
  int64_t _action;              // Action of this node
  double _avgUpper, _avgLower;  // Bounds based on observation children
  std::unordered_map<int64_t, ObservationNode> _observation_edges;

 public:
  ActionNode(int64_t action, const BeliefDistribution& belief,
             int64_t belief_depth, const OptimalPath& heuristic,
             int64_t eval_depth, const BoundFunction& lower_bound_func,
             SimInterface* pomdp);

  int64_t GetAction() const { return _action; }

  double GetAvgUpper() const { return _avgUpper; }
  double GetAvgLower() const { return _avgLower; }

  std::shared_ptr<BeliefTreeNode> GetChild(int64_t observation) const;
  const std::unordered_map<int64_t, ObservationNode>& GetChildren() const {
    return _observation_edges;
  }

  std::pair<std::shared_ptr<BeliefTreeNode>, double> ChooseObservation(
      double epsilon, double gamma, int64_t depth) const;

  void BackUp(AlphaVectorFSC& fsc, double R_lower, int64_t max_depth_sim,
              SimInterface* pomdp);

  void BackUpNoFSC();

 private:
  /// @brief Generate a set of next beliefs mapped by observation,
  /// obtained by taking `action` in belief.
  void BeliefUpdate(const BeliefDistribution& belief, int64_t belief_depth,
                    const OptimalPath& heuristic, int64_t eval_depth,
                    const BoundFunction& lower_bound_func, SimInterface* pomdp);

  void CalculateBounds();
};

static int64_t belief_tree_count = 0;

class BeliefTreeNode {
 private:
  BeliefDistribution _belief;
  std::unordered_map<int64_t, ActionNode> _action_edges;
  int64_t _belief_depth;

  int64_t _bestActUBound, _bestActLBound;
  int64_t _best_policy_node;

  double _upper_bound;
  double _lower_bound;

  int64_t _index;

 public:
  BeliefTreeNode(const BeliefDistribution& belief, double belief_depth,
                 double upper_bound, double lower_bound)
      : _belief(belief),
        _belief_depth(belief_depth),
        _bestActUBound(-1),
        _bestActLBound(-1),
        _best_policy_node(-1),
        _upper_bound(upper_bound),
        _lower_bound(lower_bound),
        _index(belief_tree_count++) {}

  void AddChild(int64_t action, const OptimalPath& heuristic,
                int64_t eval_depth, const BoundFunction& lower_bound_func,
                SimInterface* pomdp);
  const ActionNode& GetOrAddChildren(int64_t action,
                                     const OptimalPath& heuristic,
                                     int64_t eval_depth,
                                     const BoundFunction& lower_bound_func,
                                     SimInterface* pomdp);

  const BeliefDistribution& GetBelief() const { return _belief; }

  void SetBestPolicyNode(int64_t idx) { _best_policy_node = idx; }
  int64_t GetBestPolicyNode() const { return _best_policy_node; }

  void UpdateBestAction();
  int64_t GetBestActLBound() const { return _bestActLBound; }
  int64_t GetBestActUBound() const { return _bestActUBound; }

  double GetUpper() const { return _upper_bound; }
  double GetLower() const { return _lower_bound; }
  int64_t GetDepth() const { return _belief_depth; }

  std::shared_ptr<BeliefTreeNode> GetChild(int64_t action,
                                           int64_t observation) const;
  const std::unordered_map<int64_t, ObservationNode>& GetChildren(
      int64_t action) const;

  std::pair<std::shared_ptr<BeliefTreeNode>, double> ChooseObservation(
      double epsilon, double gamma);

  void BackUpActions(AlphaVectorFSC& fsc, double R_lower, int64_t max_depth_sim,
                     SimInterface* pomdp);

  void BackUpBestActionUpperNoFSC();

  int64_t GetId() const { return _index; }

  void DrawBeliefTree(std::ostream& ofs) const;

  int64_t DrawPolicyTree(std::ostream& ofs) const;

 private:
  void GenerateGraphviz(std::ostream& out) const;

  void DrawPolicyBranch(std::ostream& ofs, int64_t& i) const;
};

std::shared_ptr<BeliefTreeNode> CreateBeliefTreeNode(
    const BeliefDistribution& belief, int64_t belief_depth, double upper_bound,
    double lower_bound);

}  // namespace MCVI
