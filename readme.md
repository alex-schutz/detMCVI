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
- Path finding for bounds crashes if there are no terminal states

## Optimisations
- Initialise FSCs using shortest paths
- Optimise path storage in shortest path calculator (only need to store action + next node instead of entire path)
- Add multi-threading
- Add pruning based on reachability
- Add `AvailableActions` functionality
- Simulation depth paramater does not need to exist
