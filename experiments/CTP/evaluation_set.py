#!/usr/bin/python3

import numpy as np
from CTP_generator import generate_delaunay_graph_set, ctp_to_file
from time_series import parse_file
import subprocess
import pandas as pd
import time
import pickle
import concurrent.futures
from multiprocessing import cpu_count
import sys
import shutil

SET_SIZE = 10

max_time = {
    5: 30 * 1000,
    10: 2 * 60 * 1000,
    15: 10 * 60 * 1000,
    20: 60 * 60 * 1000,
    30: 20 * 60 * 60 * 1000,
    40: 60 * 60 * 60 * 1000,
    50: 5 * 60 * 60 * 1000,
    75: 10 * 60 * 60 * 1000,
    100: 20 * 60 * 60 * 1000,
}
eval_ms = {
    5: 1,
    10: 1,
    15: 100,
    20: 5 * 1000,
    30: 10 * 1000,
    40: 60 * 1000,
    50: 2 * 60 * 1000,
    75: 10 * 60 * 1000,
    100: 20 * 60 * 1000,
}


def initialise_folder(results_folder):
    subprocess.run(
        f"mkdir -p {results_folder}",
        check=True,
        shell=True,
    )


def instantiate_ctp(executable):
    # subprocess.run(
    #     "rm -rf build; mkdir build",
    #     check=True,
    #     shell=True,
    # )

    # Build files
    cmd = "cd build && cmake .. && make"
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

    shutil.copy("build/experiments/CTP/ctp_timeseries", executable)


def run_ctp_instance(N, i, problem_file, results_folder, executable):
    outfile = f"{results_folder}/CTPInstance_{N}_{i}.txt"

    with open(outfile, "w") as f:
        # Run solver
        cmd = [
            executable,
            problem_file,
            "--max_sim_depth",
            str(2 * N),
            "--max_time_ms",
            str(max_time[N]),
            "--eval_interval_ms",
            str(eval_ms[N]),
        ]
        p = subprocess.run(
            cmd,
            stdout=f,
            stderr=subprocess.PIPE,
            # timeout=(max_time[N] / 1000 + max_time[N] / eval_ms[N] * N) * 3,
        )
        if p.returncode != 0:
            print(f"INSTANCE {N}_{i} FAILED")
            print(p.stderr)
            return outfile, 1
        else:
            print(p.stderr, file=f)
    return outfile, 0


def work(params):
    problem_size, i, problem_file, results_folder, seed, executable = params
    print(f"Starting problem {problem_size} {i}")
    try:
        outfile, error = run_ctp_instance(
            problem_size, i, problem_file, results_folder, executable
        )
        print("run_ctp_instance completed")
    except Exception as e:
        print(f"Error in run_ctp_instance: {e}")
        # return

    if error:
        print(f"Error occurred in problem {problem_size} {i}")
        # return

    # Summarise results
    try:
        instance_result = parse_file(outfile)
        instance_result["Set number"] = i

        print(f"Saving {problem_size} {i}")
        # Save as we go
        df = instance_result
        df.to_csv(f"{results_folder}/ctp_results_{problem_size}_{i}.csv", index=False)
    except Exception as e:
        print(f"Error in processing or saving results: {e}")


def generate_problem_set(problem_size, timestr):
    results_folder = (
        f"experiments/CTP/evaluation/ctp_results_{problem_size}x{SET_SIZE}_{timestr}"
    )

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
    for n in max_time.keys():
        for i in range(SET_SIZE):
            try:
                with open(f"{results_folder}/ctp_results_{n}_{i}.csv", "r") as f:
                    results.append(pd.read_csv(f))
            except:
                pass
    if results:
        df = pd.concat(results, ignore_index=True, sort=False)
        df.to_csv(f"{results_folder}/ctp_results_all.csv", index=False)


def run_problem_set(problem_size, timestr, executable):
    results_folder, problem_files, seeds = generate_problem_set(problem_size, timestr)
    folder_exec = shutil.copy(executable, results_folder)
    futures = []
    for i, (problem_file, seed) in enumerate(zip(problem_files, seeds)):
        params = (problem_size, i, problem_file, results_folder, seed, folder_exec)
        future = tp.submit(work, params)
        futures.append((future, results_folder))

    return futures


if __name__ == "__main__":
    timestr = time.strftime("%Y-%m-%d_%H-%M")
    executable = "ctp_timeseries" + timestr
    instantiate_ctp(executable=executable)

    max_workers = min(cpu_count() - 2, 4)
    tp = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    problem_sets = max_time.keys()
    all_futures = []

    for problem_size in problem_sets:
        futures = run_problem_set(problem_size, timestr, executable)
        all_futures.extend(futures)

    # Check if futures are done and summarise results
    while all_futures:
        for future, results_folder in all_futures[:]:
            if future.done():
                all_futures.remove((future, results_folder))
                summarise_results(results_folder)

    tp.shutdown()
