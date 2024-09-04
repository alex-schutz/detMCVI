#include <iostream>

#include "Params.h"
#include "mastermind.h"

#define RANDOM_SEED (42)

int main(int argc, char* argv[]) {
  const EvalParams params = parseArgs(argc, argv);
  std::mt19937_64 rng(RANDOM_SEED);

  int colour_count = 6;
  int peg_count = 4;
  ReadMastermindParams(params.datafile, colour_count, peg_count);

  // Initialise the POMDP
  std::cout << "Initialising Mastermind" << std::endl;
  auto pomdp = Mastermind(colour_count, peg_count, rng);

  std::cout << "Writing Mastermind SARSOP to " << params.datafile + ".pomdp"
            << std::endl;
  std::fstream f(params.datafile + ".pomdp", std::fstream::out);
  pomdp.toSARSOP(f);
  f.close();
  return 0;
}
