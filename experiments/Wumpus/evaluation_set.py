#!/usr/bin/python3

import numpy as np
from evaluation import initialise_folder
from time_series import parse_file
import subprocess
import pandas as pd
import time
import concurrent.futures
from multiprocessing import cpu_count
import sys

TIMEOUT = 60 * 60 * 20

max_time = {
    2: 30 * 1000,
    3: 20 * 60 * 1000,
    4: 100 * 60 * 1000,
}
eval_ms = {
    2: 100,
    3: 1000,
    4: 2 * 60 * 1000,
}


def instantiate():
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


def run_instance(N, problem_file, results_folder):
    outfile = f"{results_folder}/WumpusInstance_{N}.txt"

    with open(outfile, "w") as f:
        # Run solver
        cmd = f"time build/experiments/Wumpus/wumpus_timeseries {problem_file} --max_sim_depth {25*N} --max_time_ms {max_time[N]} --eval_interval_ms {eval_ms[N]}"
        p = subprocess.run(
            cmd,
            stdout=f,
            stderr=subprocess.PIPE,
            timeout=TIMEOUT,
            shell=True,
        )
        if p.returncode != 0:
            print(f"INSTANCE {N} FAILED")
            print(p.stderr)
            return outfile, 1
        else:
            print(p.stderr, file=f)
    return outfile, 0


def work(params):
    problem_size, problem_file, results_folder, seed = params
    print(f"Starting Wumpus problem {problem_size}")
    outfile, error = run_instance(problem_size, problem_file, results_folder)
    if error:
        return

    # Summarise results
    instance_result = parse_file(outfile)

    # Save as we go
    df = instance_result
    df.to_csv(f"{results_folder}/wumpus_results_{problem_size}.csv", index=False)


def generate_problem_set(problem_size):
    timestr = time.strftime("%Y-%m-%d_%H-%M")
    results_folder = f"Wumpus_results_{timestr}"

    seed = np.random.randint(0, 9999999)

    initialise_folder(results_folder)

    problem_files = []
    seeds = []
    for n in range(2, 5):
        problem_file = f"{results_folder}/wumpus_world_{problem_size}.txt"
        with open(problem_file, "w") as f:
            f.write(f"{n}" + "\n")
        problem_files.append(problem_file)

    return results_folder, problem_files, seeds


def summarise_results(results_folder):
    results = []
    for i in range(2, 5):
        try:
            with open(f"{results_folder}/wumpus_results_{i}.csv", "r") as f:
                results.append(pd.read_csv(f))
        except:
            pass

    df = pd.concat(results, ignore_index=True, sort=False)
    df.to_csv(f"{results_folder}/wumpus_results_all.csv", index=False)


def run_problem_set(problem_size):
    results_folder, problem_files, seeds = generate_problem_set(problem_size)
    futures = []
    for i, (problem_file, seed) in enumerate(zip(problem_files, seeds)):
        params = (problem_size, i, problem_file, results_folder, seed)
        future = tp.submit(work, params)
        futures.append((future, results_folder))

    return futures


if __name__ == "__main__":
    instantiate()

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
