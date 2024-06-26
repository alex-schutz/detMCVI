#!/usr/bin/python3

import numpy as np
from CTP_generator import generate_delaunay_graph_set, ctp_to_file
from evaluation import initialise_folder
from time_series import parse_file
import subprocess
import pandas as pd
import time
import pickle
import concurrent.futures
from multiprocessing import cpu_count
import sys

TIMEOUT = 60 * 60 * 20
SET_SIZE = 10

max_time = {
    # 5: 30 * 1000,
    # 10: 2 * 60 * 1000,
    # 15: 10 * 60 * 1000,
    # 20: 60 * 60 * 1000,
    30: 20 * 60 * 60 * 1000,
    # 40: 60 * 60 * 60 * 1000,
    # 50: 5 * 60 * 60 * 1000,
    # 100: 5 * 60 * 60 * 1000,
}
eval_ms = {
    # 5: 0,
    # 10: 2,
    # 15: 1000,
    # 20: 5 * 1000,
    30: 10 * 1000,
    # 40: 60 * 1000,
    # 50: 2 * 60 * 1000,
    # 100: 20 * 60 * 1000,
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


def run_ctp_instance(N, i, problem_file, results_folder):
    outfile = f"{results_folder}/CTPInstance_{N}_{i}.txt"

    with open(outfile, "w") as f:
        # Run solver
        cmd = f"time build/experiments/CTP/ctp_timeseries {problem_file} --max_sim_depth {2*N} --max_time_ms {max_time[N]} --eval_interval_ms {eval_ms[N]}"
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
    return outfile, 0


def work(params):
    problem_size, i, problem_file, results_folder, seed = params
    print(f"Starting problem {problem_size} {i}")
    outfile, error = run_ctp_instance(problem_size, i, problem_file, results_folder)
    if error:
        return

    # Summarise results
    instance_result = parse_file(outfile)
    instance_result["Seed"] = seed
    instance_result["Set number"] = i

    # Save as we go
    df = instance_result
    df.to_csv(f"{results_folder}/ctp_results_{i}.csv", index=False)


def generate_problem_set(problem_size):
    timestr = time.strftime("%Y-%m-%d_%H-%M")
    results_folder = f"eval_results_{problem_size}x{SET_SIZE}_{timestr}"

    seed = np.random.randint(0, 9999999)

    initialise_folder(results_folder)

    problem_graphs = generate_delaunay_graph_set(problem_size, SET_SIZE, seed)
    with open(f"{results_folder}/problem_graphs.pickle", "wb") as f:
        pickle.dump(problem_graphs, f)

    problem_files = []
    seeds = []
    for i, (G, origin, goal, s) in enumerate(problem_graphs):
        problem_file = f"{results_folder}/ctp_graph_{problem_size}_{i}.txt"
        with open(problem_file, "w") as f:
            ctp_to_file(G, origin, goal, f)
        problem_files.append(problem_file)
        seeds.append(s)

    return results_folder, problem_files, seeds


def summarise_results(results_folder):
    results = []
    for i in range(SET_SIZE):
        try:
            with open(f"{results_folder}/ctp_results_{i}.csv", "r") as f:
                results.append(pd.read_csv(f))
        except:
            pass

    df = pd.concat(results, ignore_index=True, sort=False)
    df.to_csv(f"{results_folder}/ctp_results_all.csv", index=False)


def run_problem_set(problem_size):
    results_folder, problem_files, seeds = generate_problem_set(problem_size)
    futures = []
    for i, (problem_file, seed) in enumerate(zip(problem_files, seeds)):
        params = (problem_size, i, problem_file, results_folder, seed)
        future = tp.submit(work, params)
        futures.append((future, results_folder))

    return futures


if __name__ == "__main__":
    instantiate_ctp()

    max_workers = min(cpu_count() - 2, 5)
    tp = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    problem_sets = max_time.keys()
    all_futures = []

    for problem_size in problem_sets:
        futures = run_problem_set(problem_size)
        all_futures.extend(futures)

    # Check if futures are done and summarise results
    while all_futures:
        for future, results_folder in all_futures[:]:
            if future.done():
                all_futures.remove((future, results_folder))
                summarise_results(results_folder)

    tp.shutdown()
