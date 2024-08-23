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
import networkx as nx
import random

SET_SIZE = 10

max_time = {
    5: 30 * 1000,
    10: 1 * 60 * 1000,
    15: 10 * 60 * 1000,
    20: 60 * 60 * 1000,
    30: 15 * 60 * 1000,
    40: 30 * 60 * 1000,
    50: 45 * 60 * 1000,
    75: 3 * 60 * 60 * 1000,
    100: 10 * 60 * 60 * 1000,
}
eval_ms = {
    # 5: 1,
    # 10: 10,
    # 15: 100,
    # 20: 1000,
    30: 2 * 1000,
    # 40: 10 * 1000,
    # 50: 2 * 60 * 1000,
    # 75: 10 * 60 * 1000,
    # 100: 20 * 60 * 1000,
}
max_time_1 = {
    # 5: 30 * 1000,
    # 10: 1 * 60 * 1000,
    # 15: 10 * 60 * 1000,
    # 20: 60 * 60 * 1000,
    # 30: 15 * 60 * 1000,
}
max_time_2 = {
    30: 60 * 60 * 1000,
    # 40: 30 * 60 * 1000,
    # 50: 45 * 60 * 1000,
    # 75: 3 * 60 * 60 * 1000,
    # 100: 10 * 60 * 60 * 1000,
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
            "--max_belief_samples",
            str(10000),
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

    with open(outfile, "a") as f:
        # Run solver
        cmd = [
            "../MCVI/build/experiments/CTP/ctp_timeseries",
            problem_file,
            "--max_sim_depth",
            str(2 * N),
            "--max_time_ms",
            str(max_time[N]),
            "--eval_interval_ms",
            str(eval_ms[N]),
            "--max_belief_samples",
            str(min(5 * int(0.00007299857 * N**5.712286 + 5), 10000)),
        ]
        p = subprocess.run(
            cmd,
            stdout=f,
            stderr=subprocess.PIPE,
            timeout=(max_time[N] / 1000 + max_time[N] / eval_ms[N] * 10),
        )
        if p.returncode != 0:
            print(f"INSTANCE {N}_{i} Orig MCVI FAILED")
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


def load_problem_set(problem_size, timestr, load_timestr):
    problem_folder = f"experiments/CTP/evaluation/ctp_results_{problem_size}x{SET_SIZE}_{load_timestr}"
    results_folder = (
        f"experiments/CTP/evaluation/ctp_results_{problem_size}x{SET_SIZE}_{timestr}"
    )

    initialise_folder(results_folder)

    with open(f"{problem_folder}/problem_graphs.pickle", "rb") as f:
        problem_graphs = pickle.load(f)

    problem_files = []
    seeds = []
    for i, (G, origin, goal, s) in enumerate(problem_graphs):
        problem_file = f"{results_folder}/ctp_graph_{problem_size}_{i}.txt"
        with open(problem_file, "w") as f:
            ctp_to_file(G, origin, goal, f)
        problem_files.append(problem_file)
        seeds.append(s)

    return results_folder, problem_files, seeds


def load_graph(
    nodes,
    edge_start_ids,
    edge_end_ids,
    edge_costs,
    stoch_edge_start_ids,
    stoch_edge_end_ids,
):
    stoch_edge_scores = [random.random() for _ in stoch_edge_start_ids]
    # Create an empty directed graph
    G = nx.Graph()

    # Add all the nodes
    G.add_nodes_from(nodes)

    # Add the edges with the 'weight' attribute
    for start, end, cost in zip(edge_start_ids, edge_end_ids, edge_costs):
        G.add_edge(start, end, weight=cost)

    # Add the 'blocked_prob' attribute to the relevant edges
    for start, end, score in zip(
        stoch_edge_start_ids, stoch_edge_end_ids, stoch_edge_scores
    ):
        if G.has_edge(start, end):
            G[start][end]["blocked_prob"] = score
        else:
            raise RuntimeError(f"Expected edge {start}_{end}")

    spl = dict(nx.all_pairs_dijkstra_path_length(G))
    while True:
        goal = random.choice(nodes)
        origin = random.choice(nodes)
        if spl[goal][origin] >= 25:
            break

    return G, origin, goal, True


def generate_field_map_set(set_size: int, seed: int):
    # fmt: off
    nodes = [1, 2, 6, 9, 11, 14, 18, 22, 24, 27, 30, 32, 36, 38, 39, 40, 41, 43, 47, 50, 54, 57, 58, 59, 61, 62, 66, 69, 73, 75, 76, 80, 84, 87, 92, 94, 97, 98, 99, 101, 102, 108, 111, 115, 118, 120, 123, 127, 130, 134, 135, 137, 221, 223, 226, 234]
    edge_start_ids = [1, 38, 39, 40, 39, 57, 58, 36, 54, 61, 75, 97, 98, 101, 99, 97, 134, 39, 38, 58, 30, 2, 115, 32, 27, 87, 36, 2, 94, 94, 39, 41, 84, 2, 6, 94, 92, 24, 62, 27, 69, 118, 115, 76, 32, 76, 221, 135, 9, 11, 18, 14, 43, 80, 76, 123, 127, 108, 24, 36, 30, 99, 22, 130, 111, 73, 50, 66, 47, 47, 22, 75, 102, 102, 108, 118, 62, 97, 18, 69, 66, 1, 36, 9, 50, 11, 101, 59, 59, 120, 98, 50, 118, 6, 80, 92, 75]
    edge_end_ids = [2, 39, 40, 41, 57, 58, 59, 59, 61, 62, 76, 98, 99, 102, 120, 99, 135, 41, 57, 221, 87, 127, 137, 223, 87, 226, 54, 234, 97, 123, 54, 43, 87, 6, 9, 226, 94, 92, 66, 30, 73, 120, 118, 80, 36, 102, 223, 137, 11, 14, 22, 18, 47, 84, 108, 127, 130, 111, 27, 38, 32, 101, 24, 134, 115, 75, 54, 69, 50, 66, 94, 108, 108, 135, 137, 127, 221, 130, 92, 80, 84, 130, 223, 123, 66, 123, 118, 223, 221, 130, 127, 62, 134, 127, 97, 226, 120]
    edge_costs = [1.0366843938827515, 1.0963903665542603, 1.095942497253418, 1.0091201066970825, 1.5665544271469116, 1.1034995317459106, 1.019964575767517, 1.6042250394821167, 1.3916740417480469, 1.1529104709625244, 1.1529231071472168, 1.0500808954238892, 1.2487624883651733, 1.3050310611724854, 1.2817814350128174, 2.298818826675415, 1.0238898992538452, 1.9316502809524536, 1.0282256603240967, 1.478731393814087, 2.651732921600342, 3.9989302158355713, 3.1567797660827637, 3.1029212474823, 3.005357265472412, 2.9063923358917236, 2.7895383834838867, 3.798736572265625, 3.57950758934021, 2.53360652923584, 3.62361216545105, 2.4340240955352783, 3.565783739089966, 4.64850378036499, 3.5128726959228516, 4.931519985198975, 2.283743143081665, 4.4898881912231445, 4.342657089233398, 3.2582554817199707, 4.638794898986816, 2.246900796890259, 3.549166679382324, 4.301112651824951, 4.261964321136475, 4.118381977081299, 2.1892285346984863, 2.1872687339782715, 2.1790215969085693, 3.437047243118286, 4.458247661590576, 4.261058330535889, 4.355869293212891, 4.231439113616943, 3.9524178504943848, 4.310975551605225, 3.1624772548675537, 3.1409451961517334, 3.255725383758545, 2.147822856903076, 2.1346757411956787, 2.1075141429901123, 2.1067116260528564, 4.142826080322266, 4.025764465332031, 1.9849423170089722, 4.453464031219482, 3.144089937210083, 3.0531070232391357, 6.352810859680176, 6.48881721496582, 3.1369378566741943, 5.224290370941162, 8.027682304382324, 8.065381050109863, 5.982461929321289, 7.109212398529053, 6.914524078369141, 7.734538555145264, 3.520467758178711, 5.677493095397949, 2.4061033725738525, 2.4888052940368652, 5.786336421966553, 3.5026092529296875, 6.749170303344727, 1.8322887420654297, 1.939146637916565, 1.1526044607162476, 5.0492119789123535, 5.280936241149902, 3.103715658187866, 4.850697994232178, 3.9849438667297363, 6.984862327575684, 3.330843925476074, 7.944096088409424]
    stoch_edge_start_ids = [47, 22, 75, 102, 102, 108, 118, 62, 97, 18, 69, 66, 1, 36, 9, 50, 11, 101, 59, 59, 120, 98, 50, 118, 6, 80, 92, 75]
    stoch_edge_end_ids = [66, 94, 108, 108, 135, 137, 127, 221, 130, 92, 80, 84, 130, 223, 123, 66, 123, 118, 223, 221, 130, 127, 62, 134, 127, 97, 226, 120]
    # fmt: on
    problem_set = []
    for n in range(set_size):
        while True:
            G, origin, goal, solvable = load_graph(
                nodes,
                edge_start_ids,
                edge_end_ids,
                edge_costs,
                stoch_edge_start_ids,
                stoch_edge_end_ids,
            )
            if solvable:
                problem_set.append((G, origin, goal, seed))
                break
            else:
                seed += 1
        seed += 1
    return problem_set


def generate_problem_set(problem_size, timestr):
    results_folder = (
        f"experiments/CTP/evaluation/ctp_results_{problem_size}x{SET_SIZE}_{timestr}"
    )

    seed = np.random.randint(0, 9999999)

    initialise_folder(results_folder)

    problem_graphs = generate_field_map_set(SET_SIZE, seed)
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


def run_problem_set(problem_size, timestr, executable, tp):
    # results_folder, problem_files, seeds = load_problem_set(
    #     problem_size, timestr, "2024-08-15_18-00"
    # )
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

    max_workers = min(cpu_count() - 2, 1)
    tp = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    problem_sets = max_time_2.keys()
    all_futures = []

    for problem_size in problem_sets:
        futures = run_problem_set(problem_size, timestr, executable, tp)
        all_futures.extend(futures)

    # Check if futures are done and summarise results
    while all_futures:
        for future, results_folder in all_futures[:]:
            if future.done():
                all_futures.remove((future, results_folder))
                summarise_results(results_folder)

    tp.shutdown()

    # max_workers = min(cpu_count() - 2, 1)
    # tp2 = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    # problem_sets = max_time_2.keys()
    # all_futures = []

    # for problem_size in problem_sets:
    #     futures = run_problem_set(problem_size, timestr, executable, tp2)
    #     all_futures.extend(futures)

    # # Check if futures are done and summarise results
    # while all_futures:
    #     for future, results_folder in all_futures[:]:
    #         if future.done():
    #             all_futures.remove((future, results_folder))
    #             summarise_results(results_folder)

    # tp2.shutdown()
