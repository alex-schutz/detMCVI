#include "POMCP.h"

#include <algorithm>
#include <chrono>
#include <random>

namespace POMCP {

static size_t SampleOneNumber(size_t min, size_t max) {
  static std::mt19937_64 rng(std::random_device{}());
  auto dist = std::uniform_int_distribution<size_t>(min, max - 1);
  return dist(rng);
}

BeliefParticles::BeliefParticles(std::vector<State> &particles) {
  this->particles = particles;
}

State BeliefParticles::SampleOneState() const {
  const size_t random_i_particle = SampleOneNumber(0, this->GetParticleSize());
  return this->particles[random_i_particle];
}

int64_t TreeNode::GetActionCount(int64_t aI) const {
  if (this->action_counts.count(aI)) {
    return this->action_counts.find(aI)->second;
  } else {
    return 0;
  }
}

// delete all child
TreeNode::~TreeNode() {
  std::map<int64_t, std::map<int64_t, TreeNode *>>::iterator
      it;  // aI, oI -> n_new
  for (it = ChildNodes_.begin(); it != ChildNodes_.end(); it++) {
    std::map<int64_t, TreeNode *>::iterator it_node;
    for (it_node = it->second.begin(); it_node != it->second.end(); it_node++) {
      delete it_node->second;
    }
  }
};

double TreeNode::GetActionQ(int64_t aI) const {
  if (this->all_action_Q.count(aI)) {
    return this->all_action_Q.find(aI)->second;
  } else {
    return 0.0;
  }
}

PomcpPlanner::PomcpPlanner(SimInterface *sim, double discount) {
  this->simulator = sim;
  this->size_A = sim->GetSizeOfA();
  this->discount = discount;
}

double PomcpPlanner::Rollout(const State &sampled_sI, int64_t node_depth) {
  double sum_accumlated_rewards = 0.0;
  for (int64_t i = 0; i < this->nb_restarts_simulation; i++) {
    State sI = sampled_sI;
    double total_discount = pow(this->discount, node_depth);
    double temp_res = 0;
    while (total_discount > epsilon && node_depth < max_depth) {
      int64_t aI = SampleOneNumber(0, this->size_A);
      const auto [s_next, oI, reward, done] = this->simulator->Step(sI, aI);
      temp_res += reward * total_discount;
      total_discount *= this->discount;
      sI = s_next;
      node_depth += 1;
    }
    sum_accumlated_rewards += temp_res;
  }
  double res = sum_accumlated_rewards / nb_restarts_simulation;
  return res;
}

static bool CmpPair(const std::pair<int64_t, double> &p1,
                    const std::pair<int64_t, double> &p2) {
  return p1.second < p2.second;
}

int64_t PomcpPlanner::Search(const BeliefParticles &b) {
  const auto PlanStartTime = std::chrono::steady_clock::now();
  auto PlanEndTime = std::chrono::steady_clock::now();
  std::chrono::microseconds PlanSpentTime =
      std::chrono::duration_cast<std::chrono::microseconds>(PlanEndTime -
                                                            PlanStartTime);
  TreeNode *new_node = new TreeNode(0);

  this->rootnode = new_node;
  while (PlanSpentTime < this->timeout) {
    const State sampled_sI = b.SampleOneState();
    this->Simulate(sampled_sI, this->rootnode, 0);
    PlanEndTime = std::chrono::steady_clock::now();
    PlanSpentTime = std::chrono::duration_cast<std::chrono::microseconds>(
        PlanEndTime - PlanStartTime);
  }
  int64_t best_aI = BestAction(this->rootnode);

  this->Root_best_action_possible_obs.clear();
  int64_t size_obs = this->simulator->GetSizeOfObs();
  for (int64_t oI = 0; oI < size_obs; oI++) {
    this->Root_best_action_possible_obs[oI] =
        this->rootnode->CheckChildNodeExist(best_aI, oI);
  }
  delete this->rootnode;

  return best_aI;
}

double PomcpPlanner::Simulate(const State &sampled_sI, TreeNode *node,
                              int64_t depth) {
  double esti_V = 0;
  node->AddVisit();
  double total_discount = pow(this->discount, depth);
  if (total_discount < epsilon || depth == max_depth) {
    return 0;
  }

  int64_t aI = UcbActionSelection(node);
  if (aI < 0 || aI >= this->size_A) {
    std::cout << "simulate" << std::endl;
    std::cout << "aI: " << aI << std::endl;
    throw "";
  }

  const auto [next_sI, oI, reward, done] =
      this->simulator->Step(sampled_sI, aI);

  // check if have child node with this new history "hao"
  if (node->CheckChildNodeExist(aI, oI)) {
    TreeNode *child_node = node->GetChildNode(aI, oI);
    esti_V = reward + this->discount * Simulate(next_sI, child_node, depth + 1);
  } else {
    CreateNewNode(node, aI, oI);
    esti_V = reward + this->discount * Rollout(next_sI, depth + 1);
  }

  node->AddActionCount(aI);
  int64_t aI_count = node->GetActionCount(aI);
  double current_Qba = node->GetActionQ(aI);
  double updated_Qba = current_Qba + (esti_V - current_Qba) / aI_count;
  node->SetActionQ(aI, updated_Qba);
  node->SetValue(esti_V);
  return esti_V;
}

void PomcpPlanner::Init(double c, int64_t pomcp_nb_rollout,
                        std::chrono::microseconds timeout, double threshold,
                        int64_t max_depth) {
  this->c = c;
  this->timeout = timeout;
  this->epsilon = threshold;
  this->max_depth = max_depth;
  this->nb_restarts_simulation = pomcp_nb_rollout;
}

TreeNode *PomcpPlanner::CreateNewNode(TreeNode *parent_node, int64_t aI,
                                      int64_t oI) {
  int64_t parent_depth = parent_node->GetDepth();
  TreeNode *child_node = new TreeNode(parent_depth + 1);
  child_node->AddParentNode(parent_node);
  parent_node->AddChildNode(aI, oI, child_node);
  return child_node;
}

int64_t PomcpPlanner::UcbActionSelection(TreeNode *node) const {
  int64_t num_node_visit = node->GetVisitNumber();
  double max_value = -std::numeric_limits<double>::infinity();
  int64_t selected_aI = -1;
  for (int64_t aI = 0; aI < size_A; aI++) {
    double ratio_visit = 0;
    int64_t nb_aI_visit = node->GetActionCount(aI);
    if (nb_aI_visit == 0) {
      ratio_visit = std::numeric_limits<double>::infinity();
    } else {
      ratio_visit = num_node_visit / nb_aI_visit;
    }

    double value = node->GetActionQ(aI) + this->c * std::sqrt(ratio_visit);

    if (value > max_value) {
      max_value = value;
      selected_aI = aI;
    }
  }

  return selected_aI;
}

TreeNode *PomcpPlanner::SearchOffline(const BeliefParticles &b) {
  const auto PlanStartTime = std::chrono::steady_clock::now();
  auto PlanEndTime = std::chrono::steady_clock::now();
  std::chrono::microseconds PlanSpentTime =
      std::chrono::duration_cast<std::chrono::microseconds>(PlanEndTime -
                                                            PlanStartTime);
  TreeNode *new_node = new TreeNode(0);

  this->rootnode = new_node;
  while (PlanSpentTime.count() < this->timeout.count()) {
    const State sampled_sI = b.SampleOneState();
    this->Simulate(sampled_sI, this->rootnode, 0);
    PlanEndTime = std::chrono::steady_clock::now();
    PlanSpentTime = std::chrono::duration_cast<std::chrono::microseconds>(
        PlanEndTime - PlanStartTime);
  }

  return this->rootnode;
}

int64_t BestAction(const TreeNode *node) {
  const auto actionQ = node->GetAllActionQ();
  if (actionQ.empty()) return -1;
  return std::max_element(actionQ.begin(), actionQ.end(), CmpPair)->first;
}

}  // namespace POMCP
