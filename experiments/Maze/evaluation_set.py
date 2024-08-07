#!/usr/bin/python3

from experiments.CTP.time_series import parse_file
import subprocess
import pandas as pd
import time
import concurrent.futures
from multiprocessing import cpu_count
import sys
import shutil

SET_SIZE = 10
PROBLEM_SRC_FOLDER = "experiments/Maze/evaluation"

max_time = {
    5: 10 * 1000,
    10: 2 * 60 * 1000,
    15: 10 * 60 * 1000,
    20: 60 * 60 * 1000,
    25: 5 * 60 * 60 * 1000,
    30: 20 * 60 * 60 * 1000,
}
eval_ms = {
    5: 10,
    10: 1000,
    15: 5 * 1000,
    20: 60 * 1000,
    25: 5 * 60 * 1000,
    30: 10 * 60 * 1000,
}


def initialise_folder(results_folder):
    subprocess.run(
        f"mkdir -p {results_folder}",
        check=True,
        shell=True,
    )


def instantiate_maze(executable):
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

    shutil.copy("build/experiments/Maze/maze_timeseries", executable)


def run_maze_instance(N, i, problem_file, results_folder, executable):
    outfile = f"{results_folder}/MazeInstance_{N}_{i}.txt"

    with open(outfile, "w") as f:
        # Run solver
        cmd = f"{executable} {problem_file} --max_sim_depth {4*N*N} --max_time_ms {max_time[N]} --eval_interval_ms {eval_ms[N]}"
        p = subprocess.run(
            cmd,
            stdout=f,
            stderr=subprocess.PIPE,
            timeout=max_time[N] * 15 / 1000,
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
    problem_size, i, problem_file, results_folder, executable = params
    print(f"Starting problem {problem_size} {i}")
    outfile, error = run_maze_instance(
        problem_size, i, problem_file, results_folder, executable
    )
    if error:
        return

    # Summarise results
    instance_result = parse_file(outfile)
    instance_result["Set number"] = i

    # Save as we go
    df = instance_result
    df.to_csv(f"{results_folder}/maze_results_{problem_size}_{i}.csv", index=False)


def generate_problem_set(problem_size, timestr):
    results_folder = (
        f"experiments/Maze/evaluation/maze_results_{problem_size}x{SET_SIZE}_{timestr}"
    )

    initialise_folder(results_folder)

    problem_files = []
    x = f"{problem_size}x{problem_size}"
    for i in range(10):
        src_file = f"{PROBLEM_SRC_FOLDER}/{x}/{x}_{i}.txt"
        problem_file = f"{results_folder}/{x}_{i}.txt"
        shutil.copyfile(src_file, problem_file)
        problem_files.append(problem_file)

    return results_folder, problem_files


def summarise_results(results_folder):
    results = []
    for n in max_time.keys():
        for i in range(SET_SIZE):
            try:
                with open(f"{results_folder}/maze_results_{n}_{i}.csv", "r") as f:
                    results.append(pd.read_csv(f))
            except:
                pass

    df = pd.concat(results, ignore_index=True, sort=False)
    df.to_csv(f"{results_folder}/maze_results_all.csv", index=False)


def run_problem_set(problem_size, timestr, executable):
    results_folder, problem_files = generate_problem_set(problem_size, timestr)
    folder_exec = shutil.copy(executable, results_folder)
    futures = []
    for i, problem_file in enumerate(problem_files):
        params = (problem_size, i, problem_file, results_folder, folder_exec)
        future = tp.submit(work, params)
        futures.append((future, results_folder))

    return futures


if __name__ == "__main__":
    timestr = time.strftime("%Y-%m-%d_%H-%M")
    executable = "maze_timeseries" + timestr
    instantiate_maze(executable=executable)

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
