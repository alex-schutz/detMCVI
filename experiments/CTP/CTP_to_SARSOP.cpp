#include <iostream>

#include "CTP.h"
#include "Params.h"

#define RANDOM_SEED (42)

int main(int argc, char* argv[]) {
  const EvalParams params = parseArgs(argc, argv);
  std::mt19937_64 rng(RANDOM_SEED);

  std::vector<int64_t> nodes;
  std::unordered_map<std::pair<int64_t, int64_t>, double, pairhash> edges;
  std::unordered_map<std::pair<int64_t, int64_t>, double, pairhash> stoch_edges;
  int64_t origin;
  int64_t goal;
  ctpGraphFromFile(params.datafile, nodes, edges, stoch_edges, origin, goal);

  // Initialise the POMDP
  std::cout << "Initialising CTP" << std::endl;
  auto pomdp = CTP(rng, nodes, edges, stoch_edges, origin, goal);

  std::cout << "Writing CTP SARSOP to " << params.datafile + ".pomdp"
            << std::endl;
  std::fstream f(params.datafile + ".pomdp", std::fstream::out);
  pomdp.toSARSOP(f, params.nb_particles_b0);
  f.close();
  return 0;
}
