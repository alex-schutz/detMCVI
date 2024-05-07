#!/usr/bin/python3

import numpy as np
from CTP_generator import generate_graph
import subprocess
import re
import pandas as pd
import time

TIMEOUT = 60 * 60 * 50

seed = np.random.randint(0, 9999999)
print("seed: ", seed)

timestr = time.strftime("%Y-%m-%d_%H-%M")
RESULTS_FOLDER = f"eval_results_random_{timestr}"

problem_sizes = list(range(5, 21)) + list(range(20, 51, 5))
n_repetitions = 5


def extract_int(line):
    match = re.findall("-?\d+", line)
    return int(match[0])


def extract_float(line):
    match = re.findall("-?[\d]+[.,\d]+|-?[\d]*[.][\d]+|-?[\d]+", line)
    return float(match[0])


def process_output_file(f, N, i, seed) -> dict[str, int | float | str]:
    instance_result: dict[str, int | float | str] = {
        "Nodes": N,
        "Trial": i,
        "Seed": seed,
    }
    patterns = [
        ("AO* complete ", "AO* runtime (s)", extract_float),
        ("MCVI complete ", "MCVI runtime (s)", extract_float),
        ("State space size:", "State space size", extract_int),
        ("Observation space size:", "Observation space size", extract_int),
        ("Initial belief size:", "Initial belief size", extract_int),
        ("--- Iter ", "MCVI iterations", lambda x: extract_int(x) + 1),
        ("MCVI policy FSC contains", "MCVI policy nodes", extract_int),
        ("AO* greedy policy tree contains", "AO* policy nodes", extract_int),
        ("Evaluation of alternative (AO* greedy) policy", "AO*", None),
        ("Evaluation of policy", "MCVI", None),
    ]
    with open(f, "r") as f:
        for line in f:
            for pattern, key, extractor in patterns:
                if pattern not in line:
                    continue
                if extractor:
                    instance_result[key] = extractor(line)
                else:
                    line = next(f)
                    instance_result[f"{key} avg reward"] = extract_float(line)
                    line = next(f)
                    instance_result[f"{key} max reward"] = extract_float(line)
                    line = next(f)
                    instance_result[f"{key} min reward"] = extract_float(line)
    instance_result["avg_reward_difference"] = (
        instance_result["MCVI avg reward"] - instance_result["AO* avg reward"]
    )
    instance_result["policy_size_ratio"] = (
        instance_result["MCVI policy nodes"] / instance_result["AO* policy nodes"]
    )
    instance_result["runtime_ratio"] = (
        instance_result["MCVI runtime (s)"] / instance_result["AO* runtime (s)"]
    )
    return instance_result


if __name__ == "__main__":
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

    for N in problem_sizes:
        for i in range(n_repetitions):
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
            instance_result = process_output_file(outfile, N, i, seed)
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
