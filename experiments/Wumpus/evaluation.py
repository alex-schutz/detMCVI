#!/usr/bin/python3

import numpy as np
import subprocess
import re
import pandas as pd


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
        ("MCVI complete ", "MCVI runtime (s)", extract_float),
        ("State space size:", "State space size", extract_int),
        ("Observation space size:", "Observation space size", extract_int),
        ("Initial belief size:", "Initial belief size", extract_int),
        ("--- Iter ", "MCVI iterations", lambda x: extract_int(x) + 1),
        ("MCVI policy FSC contains", "MCVI policy nodes", extract_int),
        ("AO* greedy policy tree contains", "AO* policy nodes", extract_int),
        ("POMCP policy tree contains", "POMCP policy nodes", extract_int),
    ]
    algs = ["MCVI", "AO*", "POMCP"]
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
        f"mkdir {results_folder}",
        check=True,
        shell=True,
    )
