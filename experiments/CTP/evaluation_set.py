#!/usr/bin/python3

import numpy as np
from CTP_generator import generate_delaunay_graph_set, graph_to_cpp
from evaluation import initialise_folder
from time_series import parse_file
import subprocess
import pandas as pd
import time
import pickle
from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count
import sys

TIMEOUT = 60 * 60 * 20
SET_SIZE = 10
GRAPH_FILE = "experiments/CTP/auto_generated_graph.h"

max_time = {
    5: 30 * 1000,
    10: 2 * 60 * 1000,
    15: 10 * 60 * 1000,
    20: 60 * 60 * 1000,
    30: 20 * 60 * 60 * 1000,
    40: 60 * 60 * 60 * 1000,
    50: 5 * 60 * 60 * 1000,
    100: 5 * 60 * 60 * 1000,
}
eval_ms = {
    5: 1,
    10: 2,
    15: 1000,
    20: 10 * 1000,
    30: 30 * 1000,
    40: 60 * 1000,
    50: 2 * 60 * 1000,
    100: 20 * 60 * 1000,
}


def instantiate_ctp():
    # Build files
    cmd = "cd build && make"
    p = subprocess.run(
        cmd,
        stderr=subprocess.PIPE,
        stdout=sys.stdout,
        shell=True,
    )
    if p.returncode != 0:
        print(p.stdout)
        print(p.stderr)
        raise RuntimeError(f"Build failed with returncode {p.returncode}")


def run_ctp_instance(N, i, results_folder):
    outfile = f"{results_folder}/CTPInstance_{N}_{i}.txt"

    with open(outfile, "w") as f:
        # Run solver
        cmd = f"time build/experiments/CTP/ctp_timeseries --max_sim_depth {2*N} --max_time_ms {max_time[N]} --eval_interval_ms {eval_ms[N]}"
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
            return outfile, 1
        else:
            print(p.stderr, file=f)

        # Copy problem instance to outfile
        f.write("\n")
        with open(GRAPH_FILE, "r") as f1:
            f.write(f1.read())
    return outfile, 0


def work(params):
    problem_size, i, results_folder, seed = params
    print(f"Starting problem {i}")
    outfile, error = run_ctp_instance(problem_size, i, results_folder)
    if error:
        return

    # Summarise results
    instance_result = parse_file(outfile)
    instance_result["Seed"] = seed
    instance_result["Set number"] = i

    # Save as we go
    df = instance_result
    df.to_csv(f"{results_folder}/ctp_results_{i}.csv", index=False)


if __name__ == "__main__":

    for problem_size in max_time.keys():
        timestr = time.strftime("%Y-%m-%d_%H-%M")
        results_folder = f"eval_results_{problem_size}x{SET_SIZE}_{timestr}"

        seed = np.random.randint(0, 9999999)

        initialise_folder(results_folder)

        results = []
        headers_written = False

        problem_graphs = generate_delaunay_graph_set(problem_size, SET_SIZE, seed)
        with open(f"{results_folder}/problem_graphs.pickle", "wb") as f:
            pickle.dump(problem_graphs, f)

        tp = ThreadPool(min(cpu_count() - 1, 5))
        for i, (G, origin, goal, seed) in enumerate(problem_graphs):
            with open(GRAPH_FILE, "w") as f:
                graph_to_cpp(G, origin, goal, f)
            instantiate_ctp()
            params = problem_size, i, results_folder, seed
            tp.apply_async(work, (params,))
            time.sleep(1)

        tp.close()
        tp.join()

        results = []
        for i in range(SET_SIZE):
            try:
                with open(f"{results_folder}/ctp_results_{i}.csv", "r") as f:
                    results.append(pd.read_csv(f))
            except:
                pass

        df = pd.concat(results, ignore_index=True, sort=False)
        df.to_csv(f"{results_folder}/ctp_results_all.csv", index=False)
