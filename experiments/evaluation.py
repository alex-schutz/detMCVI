#!/usr/bin/python3

import numpy as np
from CTP_generator import generate_graph
import subprocess
import re
import pandas as pd
import time

TIMEOUT = 60 * 60 * 5
FL_REGEX = "-?[\d]+[.,\d]+|-?[\d]*[.][\d]+|-?[\d]+"
INT_REGEX = "-?\d+"

seed = np.random.randint(0, 9999999)
print("seed: ", seed)

timestr = time.strftime("%Y-%m-%d_%H-%M")
RESULTS_FOLDER = f"eval_results_random_{timestr}_{seed}"

results = []

# subprocess.run(
#     "rm -rf build; mkdir build",
#     check=True,
#     shell=True,
# )
subprocess.run(
    "cd build && cmake ..",
    check=True,
    shell=True,
)
subprocess.run(
    f"mkdir {RESULTS_FOLDER}",
    check=True,
    shell=True,
)

headers_written = False

for N in range(5, 51):
    for i in range(5):
        # Generate CTP instance
        graph_file = "experiments/auto_generated_graph.h"
        with open(graph_file, "w") as f:
            while True:
                solvable = generate_graph(N, seed, f, False, 0.4, False)
                seed += 1
                if solvable:
                    break

        outfile = f"{RESULTS_FOLDER}/CtpInstance_{N}_{i}.txt"
        with open(outfile, "w") as f:
            # Copy instance to file
            with open(graph_file, "r") as f1:
                f.write(f1.read())
            f.write("\n")

        # Build files
        cmd = "cd build && make"
        p = subprocess.run(
            cmd,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            shell=True,
        )
        if p.returncode != 0:
            print(p.stdout)
            print(p.stderr)
            raise RuntimeError(f"Build failed with returncode {p.returncode}")

        with open(outfile, "w+") as f:
            # Run solver
            cmd = "time build/experiments/ctp_experiment"
            p = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.PIPE,
                timeout=TIMEOUT,
                shell=True,
            )
            if p.returncode != 0:
                print(f"INSTANCE {N}_{i} FAILED")
                print(p.stderr)
                continue
            else:
                print(p.stderr, file=f)

        # Summarise results
        instance_result: dict[str, int | float | str] = {"Nodes": N, "Trial": i}
        with open(outfile, "r") as f:
            for line in f:
                if "AO* complete " in line:
                    instance_result["AO* runtime (s)"] = float(
                        re.findall(FL_REGEX, line)[0]
                    )
                elif "MCVI complete " in line:
                    instance_result["MCVI runtime (s)"] = float(
                        re.findall(FL_REGEX, line)[0]
                    )
                elif "--- Iter " in line:
                    instance_result["MCVI iterations"] = (
                        int(re.findall(INT_REGEX, line)[0]) + 1
                    )
                elif "MCVI policy FSC contains" in line:
                    instance_result["MCVI policy nodes"] = int(
                        re.findall(INT_REGEX, line)[0]
                    )
                elif "AO* greedy policy tree contains" in line:
                    instance_result["AO* policy nodes"] = int(
                        re.findall(INT_REGEX, line)[0]
                    )
                elif "Evaluation of alternative (AO* greedy) policy" in line:
                    line = next(f)
                    instance_result["AO* avg reward"] = float(
                        re.findall(FL_REGEX, line)[0]
                    )
                    line = next(f)
                    instance_result["AO* max reward"] = float(
                        re.findall(FL_REGEX, line)[0]
                    )
                    line = next(f)
                    instance_result["AO* min reward"] = float(
                        re.findall(FL_REGEX, line)[0]
                    )
                elif "Evaluation of policy" in line:
                    line = next(f)
                    instance_result["MCVI avg reward"] = float(
                        re.findall(FL_REGEX, line)[0]
                    )
                    line = next(f)
                    instance_result["MCVI max reward"] = float(
                        re.findall(FL_REGEX, line)[0]
                    )
                    line = next(f)
                    instance_result["MCVI min reward"] = float(
                        re.findall(FL_REGEX, line)[0]
                    )
        instance_result["avg_reward_difference"] = (
            instance_result["MCVI avg reward"] - instance_result["AO* avg reward"]
        )
        instance_result["policy_size_ratio"] = (
            instance_result["MCVI policy nodes"] / instance_result["AO* policy nodes"]
        )
        instance_result["runtime_ratio"] = (
            instance_result["MCVI runtime (s)"] / instance_result["AO* runtime (s)"]
        )
        results.append(instance_result)

        # save as we go
        df = pd.DataFrame([instance_result])
        if not headers_written:
            df.to_csv(f"{RESULTS_FOLDER}/ctp_results_random.csv", index=False)
            headers_written = True
        else:
            df.to_csv(
                f"{RESULTS_FOLDER}/ctp_results_random.csv",
                mode="a",
                index=False,
                header=False,
            )

df = pd.DataFrame(results)
df.to_csv(f"{RESULTS_FOLDER}/ctp_results_random_all.csv", index=False)
