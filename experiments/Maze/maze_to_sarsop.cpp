#include <iostream>

#include "Params.h"
#include "maze.h"

#define RANDOM_SEED (42)

int main(int argc, char* argv[]) {
  const EvalParams params = parseArgs(argc, argv);
  std::mt19937_64 rng(RANDOM_SEED);

  // Initialise the POMDP
  std::cout << "Initialising Maze" << std::endl;
  std::vector<std::string> maze;
  ReadMazeParams(params.datafile, maze);
  auto pomdp = Maze(maze, rng);

  std::cout << "Writing Maze SARSOP to " << params.datafile + ".pomdp"
            << std::endl;
  std::fstream f(params.datafile + ".pomdp", std::fstream::out);
  pomdp.toSARSOP(f, params.nb_particles_b0);
  f.close();
  return 0;
}
