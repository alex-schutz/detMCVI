# detMCVI, A Version of MCVI for Deterministic POMDPs

## Compilation
```sh
mkdir build && cd build
cmake ..
make
```

To run the Canadian Traveller Problem example, run `build/experiments/CTP_experiment`.
The file `experiments/CTP_generator.py` is provided to generate graphs for the CTP experiment, but manual graphs can also be specified.
Modify `experiments/CTP_graph.h` to update the problem instance.

## Issues
- Upper/lower bound updates should be done by averaging child bounds, not recalculating
- For large belief spaces, the pdf should be sampled instead of explicitly iterated through
- Simulation depth paramater might not need to exist, see how this was done originally

## Optimisations
- Initialise FSCs using shortest paths
- Add multi-threading
- Add pruning based on reachability
- Add `AvailableActions` functionality
- Optimise path storage in shortest path calculator (only need to store action + next node instead of entire path)
