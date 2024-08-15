#!/usr/bin/python3

import numpy as np
from CTP_generator import generate_graph, ctp_to_file
import subprocess
import re
import pandas as pd
import time

TIMEOUT = 60 * 60 * 50

timestr = time.strftime("%Y-%m-%d_%H-%M")

mode = "random"  # fan or random
problem_sizes = list(range(5, 21)) + list(range(20, 51, 5))
n_repetitions = 5

RESULTS_FOLDER = f"eval_results_{mode}_{timestr}"


def extract_int(line):
    match = re.findall("-?\d+", line)
    return int(match[0])


def extract_float(line):
    match = re.findall("-?[\d]+[.,\d]+|-?[\d]*[.][\d]+|-?[\d]+|-?inf", line)
    return float(match[0])


def percentage_by_type(results, result_types, type_index, alg) -> float:
    n_trials = sum([results[f"{alg} {t} Count"] for t in result_types])
    n_type = results[f"{alg} {result_types[type_index]} Count"]
    return n_type / n_trials * 100


def process_output_file(f, N, i, seed) -> dict[str, int | float | str]:
    instance_result: dict[str, int | float | str] = {
        "Nodes": N,
        "Trial": i,
        "Seed": seed,
    }
    patterns = [
        ("POMCP complete ", "POMCP runtime (s)", extract_float),
        ("AO* complete ", "AO* runtime (s)", extract_float),
        ("detMCVI complete ", "detMCVI runtime (s)", extract_float),
        ("OrigMCVI complete ", "OrigMCVI runtime (s)", extract_float),
        ("State space size:", "State space size", extract_int),
        ("Observation space size:", "Observation space size", extract_int),
        ("Initial belief size:", "Initial belief size", extract_int),
        ("--- Iter ", "detMCVI iterations", lambda x: extract_int(x) + 1),
        ("detMCVI policy FSC contains", "detMCVI policy nodes", extract_int),
        ("AO* greedy policy tree contains", "AO* policy nodes", extract_int),
        ("POMCP policy tree contains", "POMCP policy nodes", extract_int),
        ("OrigMCVI policy FSC contains", "OrigMCVI policy nodes", extract_int),
        ("Orig--- Iter ", "OrigMCVI iterations", lambda x: extract_int(x) + 1),
    ]
    algs = ["detMCVI", "AO*", "POMCP", "OrigMCVI"]
    result_types = [
        "completed problem",
        "exited policy",
        "max depth",
        "no solution (on policy)",
        "no solution (exited policy)",
    ]
    data_types = {
        "Count": extract_int,
        "Average regret": extract_float,
        "Highest regret": extract_float,
        "Lowest regret": extract_float,
        "Regret variance": extract_float,
    }
    for alg in algs:
        for result_type in result_types:
            for d, t in data_types.items():
                key = " ".join([alg, result_type, d])
                patterns += [(key, key, t)]

    with open(f, "r") as f:
        for line in f:
            for pattern, key, extractor in patterns:
                if pattern not in line:
                    continue
                instance_result[key] = extractor(line)

    for alg in algs:
        for i, t in enumerate(result_types):
            instance_result[f"{alg} {t} Percentage"] = percentage_by_type(
                instance_result, result_types, i, alg
            )

    return instance_result


# def generate_fan_instance(f, N):
#     print(PREAMBLE, file=f)
#     n_edges = f"#define NUM_FAN_EDGES_CTP {N}"
#     print(n_edges, file=f)
#     c_code = """
# const int64_t CTPOrigin = 0;
# const int64_t CTPGoal = 2;
# std::vector<int64_t> CTPNodes = {0, 1, 2};
# std::unordered_map<std::pair<int64_t, int64_t>, double, pairhash> CTPEdges = {
#     {{0, 1}, 1}};
# std::unordered_map<std::pair<int64_t, int64_t>, double, pairhash> CTPStochEdges;
# struct CTPDataInitializer {
#   CTPDataInitializer() {
#     double prob_prod = 1.0;
#     const double k = 2.0 / (NUM_FAN_EDGES_CTP + 1);
#     for (int64_t i = 1; i < NUM_FAN_EDGES_CTP; ++i) {
#       const int64_t node = 2 + i;
#       CTPNodes.push_back(node);
#       const double p = 1 - 1.0 / (NUM_FAN_EDGES_CTP * prob_prod);
#       prob_prod *= p;
#       CTPEdges[{1, node}] = k * i;
#       CTPEdges[{2, node}] = 1.0;
#       CTPStochEdges[{1, node}] = p;
#     }
#     CTPEdges[{1, 2 + NUM_FAN_EDGES_CTP}] = k * NUM_FAN_EDGES_CTP;
#     CTPEdges[{2, 2 + NUM_FAN_EDGES_CTP}] = 1.0;
#     CTPNodes.push_back(2 + NUM_FAN_EDGES_CTP);
#   }
# };
# static CTPDataInitializer ctpDataInitializer;
# """
#     print(c_code, file=f)


def generate_ctp_instance(N, seed, graph_file):
    with open(graph_file, "w") as f:
        if mode == "random":
            while True:
                G, origin, goal, solvable = generate_graph(N, seed, True, 0.4, False)
                if solvable:
                    ctp_to_file(G, origin, goal, f)
                    break
                else:
                    seed += 1
        # elif mode == "fan":
        #     generate_fan_instance(f, N)
    return seed


def run_ctp_instance(N, i):
    outfile = f"{RESULTS_FOLDER}/CTPInstance_{N}_{i}.txt"
    problem_runtime = (N**2 - 20) * 1000

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

    with open(outfile, "w") as f:
        # Run solver
        cmd = (
            f"time build/experiments/CTP/ctp_experiment --max_time_ms {problem_runtime}"
        )
        p = subprocess.run(
            cmd,
            stdout=f,
            stderr=subprocess.PIPE,
            shell=True,
        )
        if p.returncode != 0:
            print(f"INSTANCE {N}_{i} FAILED")
            print(p.stderr)
            return outfile, 1
        else:
            print(p.stderr, file=f)

        # Copy problem instance to outfile
        f.write("\n")
        with open(GRAPH_FILE, "r") as f1:
            f.write(f1.read())
    return outfile, 0


def initialise_folder(results_folder):
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
        f"mkdir -p {results_folder}",
        check=True,
        shell=True,
    )


if __name__ == "__main__":
    seed = np.random.randint(0, 9999999)

    initialise_folder(RESULTS_FOLDER)

    results = []
    headers_written = False

    for N in problem_sizes:
        for i in range(n_repetitions):
            graph_file = f"{RESULTS_FOLDER}/ctp_graph_{N}_{i}.txt"
            seed = generate_ctp_instance(N, seed, graph_file)

            outfile, error = run_ctp_instance(N, i)
            if error:
                continue

            # Summarise results
            instance_result = process_output_file(outfile, N, i, seed)
            results.append(instance_result)
            seed += 1

            # Save as we go
            df = pd.DataFrame([instance_result])
            if not headers_written:
                df.to_csv(f"{RESULTS_FOLDER}/ctp_results_{mode}.csv", index=False)
                headers_written = True
            else:
                df.to_csv(
                    f"{RESULTS_FOLDER}/ctp_results_{mode}.csv",
                    mode="a",
                    index=False,
                    header=False,
                )

    df = pd.DataFrame(results)
    df.to_csv(f"{RESULTS_FOLDER}/ctp_results_{mode}_all.csv", index=False)
