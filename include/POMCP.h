#pragma once

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>

#include "BeliefDistribution.h"
#include "Bound.h"
#include "Sample.h"
#include "SimInterface.h"

namespace POMCP {

using State = MCVI::State;
using SimInterface = MCVI::SimInterface;
class TreeNode;
using TreeNodePtr = std::shared_ptr<TreeNode>;

class BeliefParticles {
 private:
  std::vector<State> particles;  // a vector of sI

 public:
  BeliefParticles() {};
  ~BeliefParticles() {};
  BeliefParticles(std::vector<State> &particles);
  State SampleOneState() const;
  size_t GetParticleSize() const { return this->particles.size(); }
  const std::vector<State> &getParticles() const { return particles; }
};

class TreeNode {
 private:
  TreeNodePtr ParentNode_ = nullptr;
  std::map<int64_t, std::map<int64_t, TreeNodePtr>>
      ChildNodes_;  // aI, oI -> n_new

  std::map<int64_t, int64_t> action_counts;
  std::map<int64_t, double> all_action_Q;
  double value_ = 0;
  int64_t visits_ = 0;
  int64_t depth = 0;

 public:
  TreeNode() {};
  TreeNode(int64_t depth) { this->depth = depth; }

  void AddParentNode(TreeNodePtr ParentNode) { this->ParentNode_ = ParentNode; }
  void AddChildNode(int64_t aI, int64_t oI, TreeNodePtr ChildNode) {
    this->ChildNodes_[aI][oI] = ChildNode;
  }

  void SetValue(double value) { this->value_ = value; }
  void AddVisit() { this->visits_ += 1; }
  void AddActionCount(int64_t aI) { this->action_counts[aI] += 1; }
  int64_t GetVisitNumber() { return this->visits_; }

  TreeNodePtr GetChildNode(int64_t aI, int64_t oI) {
    return this->ChildNodes_[aI][oI];
  }
  const std::map<int64_t, std::map<int64_t, TreeNodePtr>> &GetChildNodes()
      const {
    return this->ChildNodes_;
  }

  int64_t GetActionCount(int64_t aI) const;

  void SetActionQ(int64_t aI, double Q) { this->all_action_Q[aI] = Q; }
  double GetActionQ(int64_t aI) const;
  const std::map<int64_t, double> &GetAllActionQ() const {
    return this->all_action_Q;
  }

  int64_t GetDepth() { return this->depth; }
};

int64_t BestAction(const TreeNodePtr node);

class PomcpPlanner {
 private:
  int64_t max_depth = 100;
  TreeNodePtr rootnode = nullptr;
  SimInterface *simulator;  // should change it to const
  int64_t size_A;
  int64_t nb_restarts_simulation = 1;  // default
  double epsilon = 0.01;
  double discount;
  std::vector<TreeNode> all_nodes;
  std::chrono::microseconds timeout;
  double c;

 public:
  PomcpPlanner() {};
  PomcpPlanner(SimInterface *sim, double discount);
  ~PomcpPlanner() {};
  void Init(double c, int64_t pomcp_nb_rollout,
            std::chrono::microseconds timeout, double threshold,
            int64_t max_depth);
  int64_t Search(const BeliefParticles &b);
  double Rollout(const State &sampled_sI,
                 int64_t node_depth);  // random policy simulation
  double Simulate(const State &sampled_sI, TreeNodePtr node, int64_t depth);
  TreeNodePtr CreateNewNode(TreeNodePtr parent_node, int64_t aI, int64_t oI);
  int64_t UcbActionSelection(TreeNodePtr node) const;
  void SearchOffline(const BeliefParticles &b, TreeNodePtr rootnode);
};

void RunPOMCPAndEvaluate(const BeliefParticles &init_belief, double pomcp_c,
                         int64_t pomcp_nb_rollout, double pomcp_epsilon,
                         int64_t pomcp_depth, int64_t max_computation_ms,
                         int64_t max_eval_steps, int64_t n_eval_trials,
                         int64_t nb_particles_b0, int64_t eval_interval_ms,
                         int64_t completion_threshold, int64_t completion_reps,
                         int64_t node_limit, std::mt19937_64 &rng,
                         const MCVI::OptimalPath &solver,
                         std::optional<MCVI::StateValueFunction> valFunc,
                         SimInterface *pomdp);

size_t EvaluationWithGreedyTreePolicy(
    TreeNodePtr root, int64_t max_steps, int64_t num_sims,
    int64_t init_belief_samples, SimInterface *pomdp, std::mt19937_64 &rng,
    const MCVI::OptimalPath &solver,
    std::optional<MCVI::StateValueFunction> valFunc,
    const std::string &alg_name);

// count the number of nodes in a tree
size_t CountNodes(const TreeNodePtr &root);

}  // namespace POMCP
