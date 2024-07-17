#pragma once

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>

#include "SimInterface.h"
#include "time.h"

namespace POMCP {

using State = MCVI::State;
using SimInterface = MCVI::SimInterface;

class BeliefParticles {
 private:
  std::vector<State> particles;  // a vector of sI

 public:
  BeliefParticles(){};
  ~BeliefParticles(){};
  BeliefParticles(std::vector<State> &particles);
  State SampleOneState() const;
  size_t GetParticleSize() const { return this->particles.size(); }
  const std::vector<State> &getParticles() const { return particles; }
};

class TreeNode {
 private:
  TreeNode *ParentNode_ = nullptr;
  std::map<int64_t, std::map<int64_t, TreeNode *>>
      ChildNodes_;  // aI, oI -> n_new

  std::map<int64_t, int64_t> action_counts;
  std::map<int64_t, double> all_action_Q;
  double value_ = 0;
  int64_t visits_ = 0;
  int64_t depth = 0;

 public:
  TreeNode(){};
  TreeNode(int64_t depth) { this->depth = depth; }

  void AddParentNode(TreeNode *ParentNode) { this->ParentNode_ = ParentNode; }
  void AddChildNode(int64_t aI, int64_t oI, TreeNode *ChildNode) {
    this->ChildNodes_[aI][oI] = ChildNode;
  }

  void SetValue(double value) { this->value_ = value; }
  void AddVisit() { this->visits_ += 1; }
  void AddActionCount(int64_t aI) { this->action_counts[aI] += 1; }
  int64_t GetVisitNumber() { return this->visits_; }

  TreeNode *GetChildNode(int64_t aI, int64_t oI) {
    return this->ChildNodes_[aI][oI];
  }
  bool CheckChildNodeExist(int64_t aI, int64_t oI) {
    return this->ChildNodes_[aI].count(oI);
  }

  int64_t GetActionCount(int64_t aI) const;

  void SetActionQ(int64_t aI, double Q) { this->all_action_Q[aI] = Q; }
  double GetActionQ(int64_t aI) const;
  const std::map<int64_t, double> &GetAllActionQ() const {
    return this->all_action_Q;
  }

  int64_t GetDepth() { return this->depth; }

  ~TreeNode();
};

int64_t BestAction(const TreeNode *node);

class PomcpPlanner {
 private:
  int64_t max_depth = 100;
  TreeNode *rootnode = nullptr;
  SimInterface *simulator;  // should change it to const
  int64_t size_A;
  int64_t nb_restarts_simulation = 1;  // default
  double epsilon = 0.01;
  double discount;
  std::vector<TreeNode> all_nodes;
  std::chrono::microseconds timeout;
  double c;

  std::map<int64_t, bool> Root_best_action_possible_obs;

 public:
  PomcpPlanner(){};
  PomcpPlanner(SimInterface *sim, double discount);
  ~PomcpPlanner(){};
  void Init(double c, int64_t pomcp_nb_rollout,
            std::chrono::microseconds timeout, double threshold,
            int64_t max_depth);
  int64_t Search(const BeliefParticles &b);
  double Rollout(const State &sampled_sI,
                 int64_t node_depth);  // random policy simulation
  double Simulate(const State &sampled_sI, TreeNode *node, int64_t depth);
  TreeNode *CreateNewNode(TreeNode *parent_node, int64_t aI, int64_t oI);
  int64_t UcbActionSelection(TreeNode *node) const;
  POMCP::TreeNode *SearchOffline(const BeliefParticles &b);
};

}  // namespace POMCP
