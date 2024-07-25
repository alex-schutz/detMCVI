#include "POMCP.h"

#include <algorithm>

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
      if (done) break;
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
  TreeNodePtr new_node = std::make_shared<TreeNode>(0);

  this->rootnode = new_node;
  while (PlanSpentTime < this->timeout) {
    const State sampled_sI = b.SampleOneState();
    this->Simulate(sampled_sI, this->rootnode, 0);
    PlanEndTime = std::chrono::steady_clock::now();
    PlanSpentTime = std::chrono::duration_cast<std::chrono::microseconds>(
        PlanEndTime - PlanStartTime);
  }
  return BestAction(this->rootnode);
}

double PomcpPlanner::Simulate(const State &sampled_sI, TreeNodePtr node,
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
    TreeNodePtr child_node = node->GetChildNode(aI, oI);
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

TreeNodePtr PomcpPlanner::CreateNewNode(TreeNodePtr parent_node, int64_t aI,
                                        int64_t oI) {
  int64_t parent_depth = parent_node->GetDepth();
  TreeNodePtr child_node = std::make_shared<TreeNode>(parent_depth + 1);
  child_node->AddParentNode(parent_node);
  parent_node->AddChildNode(aI, oI, child_node);
  return child_node;
}

int64_t PomcpPlanner::UcbActionSelection(TreeNodePtr node) const {
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

void PomcpPlanner::SearchOffline(const BeliefParticles &b,
                                 TreeNodePtr rootnode) {
  const auto PlanStartTime = std::chrono::steady_clock::now();
  auto PlanEndTime = std::chrono::steady_clock::now();
  std::chrono::microseconds PlanSpentTime =
      std::chrono::duration_cast<std::chrono::microseconds>(PlanEndTime -
                                                            PlanStartTime);

  this->rootnode = rootnode;
  while (PlanSpentTime.count() < this->timeout.count()) {
    const State sampled_sI = b.SampleOneState();
    this->Simulate(sampled_sI, this->rootnode, 0);
    PlanEndTime = std::chrono::steady_clock::now();
    PlanSpentTime = std::chrono::duration_cast<std::chrono::microseconds>(
        PlanEndTime - PlanStartTime);
  }
}

int64_t BestAction(const TreeNodePtr node) {
  const auto actionQ = node->GetAllActionQ();
  if (actionQ.empty()) return -1;
  return std::max_element(actionQ.begin(), actionQ.end(), CmpPair)->first;
}

static std::pair<double, bool> OracleReward(const MCVI::State &state,
                                            const MCVI::OptimalPath &solver,
                                            int64_t max_depth) {
  const auto [sum_reward, path] = solver.getMaxReward(state, max_depth);
  const bool can_reach_terminal = true;
  return {sum_reward, can_reach_terminal};
}

size_t EvaluationWithGreedyTreePolicy(
    TreeNodePtr root, int64_t max_steps, int64_t num_sims,
    int64_t init_belief_samples, SimInterface *pomdp, std::mt19937_64 &rng,
    const MCVI::OptimalPath &solver,
    std::optional<MCVI::StateValueFunction> valFunc,
    const std::string &alg_name) {
  const double gamma = pomdp->GetDiscount();
  MCVI::EvaluationStats eval_stats;
  const MCVI::BeliefDistribution init_belief_eval =
      MCVI::SampleInitialBelief(init_belief_samples, pomdp);
  for (int64_t sim = 0; sim < num_sims; ++sim) {
    State state = MCVI::SampleOneState(init_belief_eval, rng);
    const auto [optimal, has_soln] =
        (valFunc.has_value()) ? valFunc.value()(state, max_steps)
                              : OracleReward(state, solver, max_steps);
    double sum_r = 0.0;
    auto node = root;
    int64_t i = 0;
    for (; i < max_steps; ++i) {
      const int64_t action = (node) ? BestAction(node) : -1;
      if (!node || action == -1) {
        if (!has_soln) {
          eval_stats.no_solution_off_policy.update(sum_r - optimal);
        } else {
          eval_stats.off_policy.update(sum_r);
        }
        break;
      }
      const auto [sNext, obs, reward, done] = pomdp->Step(state, action);
      sum_r += std::pow(gamma, i) * reward;

      if (done) {
        if (!has_soln) {
          eval_stats.no_solution_on_policy.update(sum_r - optimal);
        } else {
          eval_stats.complete.update(sum_r - optimal);
        }
        break;
      }

      state = sNext;
      node = node->GetChildNode(action, obs);
    }
    if (i == max_steps) {
      if (!has_soln) {
        eval_stats.no_solution_on_policy.update(sum_r - optimal);
      } else {
        eval_stats.max_depth.update(sum_r - optimal);
      }
    }
  }
  PrintStats(eval_stats.complete, alg_name + " completed problem");
  PrintStats(eval_stats.off_policy, alg_name + " exited policy");
  PrintStats(eval_stats.max_depth, alg_name + " max depth");
  PrintStats(eval_stats.no_solution_on_policy,
             alg_name + " no solution (on policy)");
  PrintStats(eval_stats.no_solution_off_policy,
             alg_name + " no solution (exited policy)");
  return eval_stats.complete.getCount();
}

void RunPOMCPAndEvaluate(const BeliefParticles &init_belief, double pomcp_c,
                         int64_t pomcp_nb_rollout, double pomcp_epsilon,
                         int64_t pomcp_depth, int64_t max_computation_ms,
                         int64_t max_eval_steps, int64_t n_eval_trials,
                         int64_t nb_particles_b0, int64_t eval_interval_ms,
                         int64_t completion_threshold, int64_t completion_reps,
                         std::mt19937_64 &rng, const MCVI::OptimalPath &solver,
                         std::optional<MCVI::StateValueFunction> valFunc,
                         SimInterface *pomdp) {
  const double gamma = pomdp->GetDiscount();
  auto pomcp = PomcpPlanner(pomdp, gamma);
  pomcp.Init(pomcp_c, pomcp_nb_rollout,
             std::chrono::milliseconds(eval_interval_ms), pomcp_epsilon,
             pomcp_depth);
  TreeNodePtr root_node = std::make_shared<TreeNode>(0);

  int64_t time_sum = 0;
  int64_t completed_times = 0;
  while (time_sum < max_computation_ms) {
    pomcp.SearchOffline(init_belief, root_node);
    time_sum += eval_interval_ms;

    std::cout << "Evaluation of POMCP policy (" << max_eval_steps << " steps, "
              << n_eval_trials << " trials) at time " << time_sum / 1e3 << ":"
              << std::endl;
    const int64_t completed_count = EvaluationWithGreedyTreePolicy(
        root_node, max_eval_steps, n_eval_trials, nb_particles_b0, pomdp, rng,
        solver, valFunc, "POMCP");
    std::cout << "POMCP offline policy tree contains " << CountNodes(root_node)
              << " nodes." << std::endl;

    if (completed_count >= completion_threshold)
      completed_times++;
    else
      completed_times = 0;
    if (completed_times >= completion_reps) {
      std::cout << "POMCP planning complete, completed " << completion_reps
                << " times." << std::endl;
      return;
    }
  }
  std::cout << "POMCP planning complete, reached computation time."
            << std::endl;
  return;
}

size_t CountNodes(const TreeNodePtr &root) {
  if (!root) return 0;
  size_t count = 1;
  for (const auto &action_pair : root->GetChildNodes()) {
    for (const auto &observation_pair : action_pair.second) {
      count += CountNodes(observation_pair.second);
    }
  }
  return count;
}

}  // namespace POMCP
