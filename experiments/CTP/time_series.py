import plotly.graph_objects as go
import numpy as np
import pandas as pd
import re
import plotly.io as pio

pio.kaleido.scope.mathjax = None


def extract_int(line) -> int:
    match = re.findall("-?\d+", line)
    return int(match[0])


def extract_float(line) -> float:
    match = re.findall("-?[\d]+[.,\d]+|-?[\d]*[.][\d]+|-?[\d]+|-?inf", line)
    return float(match[0])


def percentage_by_type(results, result_types, type_index) -> float:
    n_trials = sum([results[f"{t} Count"] for t in result_types])
    n_type = results[f"{result_types[type_index]} Count"]
    return n_type / n_trials * 100 if n_trials > 0 else np.nan


def parse_evaluation(lines: list[str]) -> dict[str, float | int]:
    result_types = [
        "completed problem",
        "exited policy",
        "max iterations",
        "no solution (on policy)",
        "no solution (exited policy)",
    ]
    data_types = {
        "Count": extract_int,
        "Average reward": extract_float,
        "Highest reward": extract_float,
        "Lowest reward": extract_float,
        "Reward variance": extract_float,
    }
    patterns = []
    for result_type in result_types:
        for d, t in data_types.items():
            key = " ".join([result_type, d])
            patterns += [(key, t)]

    result = {}
    for pattern, extractor in patterns:
        for line in lines:
            if pattern in line:
                result[pattern] = extractor(line)
            elif "nodes." in line:
                result["policy nodes"] = extract_int(line)

    for i, res in enumerate(result_types):
        if result[f"{res} Count"] == 0:
            result[f"{res} Average reward"] = np.nan
        result[f"{res} Percentage"] = percentage_by_type(result, result_types, i)

    return result


def parse_file(filename) -> pd.DataFrame:
    mcvi_stats = {}
    ao_stats = {}

    with open(filename, "r") as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            if lines[i].startswith("Evaluation of policy"):
                time = extract_float(lines[i].split("at time ")[1])
                if i + 27 > len(lines):
                    break
                info_lines = lines[i + 1 : i + 27]
                mcvi_stats[time] = parse_evaluation(info_lines)
                i += 26
            elif lines[i].startswith("Evaluation of alternative (AO* greedy) policy"):
                time = extract_float(lines[i].split("at time ")[1])
                if i + 27 > len(lines):
                    break
                info_lines = lines[i + 1 : i + 27]
                ao_stats[time] = parse_evaluation(info_lines)
                i += 26
            else:
                i += 1

    mcvi_data = [
        {"Algorithm": "MCVI", "Timestamp": timestamp, **stats}
        for timestamp, stats in mcvi_stats.items()
    ]
    ai_data = [
        {"Algorithm": "AO*", "Timestamp": timestamp, **stats}
        for timestamp, stats in ao_stats.items()
    ]
    combined_data = mcvi_data + ai_data
    return pd.DataFrame(combined_data)


def lighten_colour(colour_str, factor=0.2):
    r, g, b = tuple(int(colour_str.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{factor})"


def plot_timeseries(df: pd.DataFrame, title, figname, output="show"):
    algs = ["MCVI", "AO*"]
    fig = go.Figure()
    colours = ["#cb6ce6", "#0097b2"]

    for i, alg in enumerate(algs):
        data = df[df["Algorithm"] == alg].sort_values("Timestamp")
        data_info = data[
            [
                "completed problem Highest reward",
                "completed problem Lowest reward",
                "completed problem Percentage",
            ]
        ]
        fig.add_trace(
            go.Scatter(
                x=data["Timestamp"],
                y=data["completed problem Average reward"],
                mode="lines+markers",
                line_shape="hv",
                line=dict(color=colours[i]),
                name=alg,
                customdata=data_info,
                hovertemplate="Computation time: %{x:.2f}s<br>Average reward: %{y:.2f}<br>Highest reward: %{customdata[0]:.2f}<br>Lowest reward: %{customdata[1]:.2f}<br>Percentage completed: %{customdata[2]:.2f}%",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=data["Timestamp"],
                y=data["completed problem Highest reward"],
                fill="tonexty",
                fillcolor=lighten_colour(colours[i], 0.2),
                mode="lines",
                line=dict(color=lighten_colour(colours[i], 0.2)),
                line_shape="hv",
                name="Highest reward",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=data["Timestamp"],
                y=data["completed problem Lowest reward"],
                fill="tonexty",
                fillcolor=lighten_colour(colours[i], 0.2),
                mode="lines",
                line=dict(color=lighten_colour(colours[i], 0.2)),
                line_shape="hv",
                name="Lowest reward",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Execution time (s)",
        yaxis_title="Reward",
        xaxis_range=[0, df["Timestamp"].max()],
    )

    if output == "show":
        fig.show()
    elif output in ["png", "eps", "pdf"]:
        fig.write_image(f"{figname}.{output}", scale=2)
    elif output == "html":
        fig.write_html(f"{figname}.html")


def plot_data(df: pd.DataFrame, dataname, ylabel, title, figname, output="show"):
    algs = ["MCVI", "AO*"]
    fig = go.Figure()
    colours = ["#cb6ce6", "#0097b2"]

    for i, alg in enumerate(algs):
        data = df[df["Algorithm"] == alg].sort_values("Timestamp")
        fig.add_trace(
            go.Scatter(
                x=data["Timestamp"],
                line=dict(color=colours[i]),
                y=data[dataname],
                mode="lines",
                name=alg,
                line_shape="hv",
            )
        )

    fig.update_layout(
        # title=title,
        xaxis_title="Execution time (s)",
        yaxis_title=ylabel,
        xaxis_range=[0, df["Timestamp"].max()],
        font=dict(size=32),
    )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

    if output == "show":
        fig.show()
    elif output in ["png", "eps", "pdf"]:
        fig.write_image(f"{figname}.{output}")
    elif output == "html":
        fig.write_html(f"{figname}.html")


if __name__ == "__main__":
    output = "show"
    # output="png"
    # output="html"

    # series_file1 = "timeseries_10_516847623.txt"
    # series_file = "timeseries_14_516847623_2.txt"
    series_file1 = "timeseries_40_9117029.txt"
    series_file2 = "timeseries_40_9117029_AO3.txt"
    # generate_graph(40, 9117029, True, 0.4, True)

    df = parse_file(series_file1)
    df = df[df["Algorithm"] == "MCVI"]
    df2 = parse_file(series_file2)
    df = pd.concat([df, df2], ignore_index=True, sort=False)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    plot_timeseries(df, "Policy value", "timeseries_14_value", output)
    plot_data(
        df,
        "completed problem Percentage",
        "Goal reached %",
        "Policy completion",
        "timeseries_14_comp",
        output,
    )
    plot_data(
        df,
        "policy nodes",
        "Policy nodes",
        "Policy size",
        "timeseries_14_nodes",
        output,
    )
