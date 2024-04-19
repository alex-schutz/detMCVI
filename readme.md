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
	- This is not working. Need to do the following:
		- Add observation child nodes which track upper/lower bounds and values instead of in belief. Can still point to child belief. That way when backing up we include the immediate reward.
		- Check the belief expansion procedure
	- Actually I think the main problem might have been that we weren't discounting the initial upper bound estimate. Could we integrate this into the previous (pre-branch) version?
- Simulation depth paramater might not need to exist, see how this was done originally
	- That's the excess uncertainty parameter in the belief expansion

## Optimisations
- Initialise FSCs using shortest paths
- Add multi-threading
- Add pruning based on reachability
- Add `AvailableActions` functionality
- Optimise path storage in shortest path calculator (only need to store action + next node instead of entire path)
