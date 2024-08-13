#include <iostream>

#include "Params.h"
#include "wumpus.h"

#define RANDOM_SEED (42)

int main(int argc, char* argv[]) {
  const EvalParams params = parseArgs(argc, argv);
  std::mt19937_64 rng(RANDOM_SEED);

  // Initialise the POMDP
  std::cout << "Initialising Wumpus" << std::endl;
  int64_t grid_size = 4;
  ReadWumpusParams(params.datafile, grid_size);
  const size_t heuristic_samples = params.max_belief_samples;
  auto pomdp = Wumpus(grid_size, heuristic_samples, rng);

  std::cout << "Writing CTP Wumpus to " << params.datafile + ".pomdp"
            << std::endl;
  std::fstream f(params.datafile + ".pomdp", std::fstream::out);
  pomdp.toSARSOP(f, params.nb_particles_b0);
  f.close();
  return 0;
}
