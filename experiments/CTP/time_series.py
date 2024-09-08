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
            result[f"{res} Average regret"] = np.nan
        result[f"{res} Percentage"] = percentage_by_type(result, result_types, i)

    return result


def parse_file(filename) -> pd.DataFrame:
    detmcvi_stats = {}
    ao_stats = {}
    pomcp_stats = {}
    origmcvi_stats = {}
    sarsop_stats = {}
    qmdp_stats = {}

    with open(filename, "r") as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            if lines[i].startswith("Evaluation of policy"):
                tline = lines[i].split("at time ")
                time = extract_float(tline[1]) if len(tline) > 1 else np.nan
                if i + 27 > len(lines):
                    break
                info_lines = lines[i + 1 : i + 27]
                detmcvi_stats[time] = parse_evaluation(info_lines)
                i += 26
            elif lines[i].startswith("Evaluation of") and "AO*" in lines[i]:
                tline = lines[i].split("at time ")
                time = extract_float(tline[1]) if len(tline) > 1 else np.nan
                if i + 27 > len(lines):
                    break
                info_lines = lines[i + 1 : i + 27]
                ao_stats[time] = parse_evaluation(info_lines)
                i += 26
            elif lines[i].startswith("Evaluation of POMCP policy"):
                tline = lines[i].split("at time ")
                time = extract_float(tline[1]) if len(tline) > 1 else np.nan
                if i + 27 > len(lines):
                    break
                info_lines = lines[i + 1 : i + 27]
                pomcp_stats[time] = parse_evaluation(info_lines)
                i += 26
            elif lines[i].startswith("Evaluation of OrigMCVI policy"):
                tline = lines[i].split("at time ")
                time = extract_float(tline[1]) if len(tline) > 1 else np.nan
                if i + 27 > len(lines):
                    break
                info_lines = lines[i + 1 : i + 27]
                origmcvi_stats[time] = parse_evaluation(info_lines)
                i += 26
            elif lines[i].startswith("Evaluation of SARSOP policy"):
                tline = lines[i].split("at time ")
                time = extract_float(tline[1]) if len(tline) > 1 else np.nan
                if i + 27 > len(lines):
                    break
                info_lines = lines[i + 1 : i + 27]
                sarsop_stats[time] = parse_evaluation(info_lines)
                i += 26
            elif lines[i].startswith("Evaluation of QMDP policy"):
                tline = lines[i].split("at time ")
                time = extract_float(tline[1]) if len(tline) > 1 else np.nan
                if i + 27 > len(lines):
                    break
                info_lines = lines[i + 1 : i + 27]
                qmdp_stats[time] = parse_evaluation(info_lines)
                i += 26
            else:
                i += 1

    mcvi_data = [
        {"Algorithm": "detMCVI", "Timestamp": timestamp, **stats}
        for timestamp, stats in detmcvi_stats.items()
    ]
    ao_data = [
        {"Algorithm": "AO*", "Timestamp": timestamp, **stats}
        for timestamp, stats in ao_stats.items()
    ]
    pomcp_data = [
        {"Algorithm": "POMCP", "Timestamp": timestamp, **stats}
        for timestamp, stats in pomcp_stats.items()
    ]
    orig_data = [
        {"Algorithm": "OrigMCVI", "Timestamp": timestamp, **stats}
        for timestamp, stats in origmcvi_stats.items()
    ]
    sarsop_data = [
        {"Algorithm": "SARSOP", "Timestamp": timestamp, **stats}
        for timestamp, stats in sarsop_stats.items()
    ]
    qmdp_data = [
        {"Algorithm": "QMDP", "Timestamp": timestamp, **stats}
        for timestamp, stats in qmdp_stats.items()
    ]
    combined_data = (
        mcvi_data + ao_data + pomcp_data + orig_data + sarsop_data + qmdp_data
    )
    return pd.DataFrame(combined_data)


def lighten_colour(colour_str, factor=0.2):
    r, g, b = tuple(int(colour_str.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{factor})"


algs = ["detMCVI", "AO*", "POMCP", "OrigMCVI", "QMDP"]
colours = ["#cb6ce6", "#0097b2", "#90e079", "#a0e079", "#f5bd45"]


def plot_timeseries(df: pd.DataFrame, title, figname, output="show"):
    fig = go.Figure()

    for i, alg in enumerate(algs):
        data = df[df["Algorithm"] == alg].sort_values("Timestamp")
        data_info = data[
            [
                "completed problem Highest regret",
                "completed problem Lowest regret",
                "completed problem Percentage",
            ]
        ]
        if len(data) == 1:
            fig.add_trace(
                go.Scatter(
                    x=data["Timestamp"],
                    y=-data["completed problem Average regret"],
                    mode="markers" if len(data) == 1 else "lines",
                    # line_shape="hv",
                    marker_symbol="cross",
                    line=dict(color=colours[i]),
                    name=alg,
                    customdata=data_info,
                    hovertemplate="Computation time: %{x:.2f}s<br>Average regret: %{y:.2f}<br>Highest regret: %{customdata[0]:.2f}<br>Lowest regret: %{customdata[1]:.2f}<br>Percentage completed: %{customdata[2]:.2f}%",
                    error_y=dict(
                        type="data",
                        array=1.96
                        * np.sqrt(data["completed problem Regret variance"])
                        / np.sqrt(data["completed problem Count"]),
                        visible=True,
                    ),
                )
            )

        else:
            fig.add_trace(
                go.Scatter(
                    x=data["Timestamp"],
                    y=-data["completed problem Average regret"],
                    mode="markers" if len(data) == 1 else "lines",
                    # line_shape="hv",
                    marker_symbol="cross",
                    line=dict(color=colours[i]),
                    name=alg,
                    customdata=data_info,
                    hovertemplate="Computation time: %{x:.2f}s<br>Average regret: %{y:.2f}<br>Highest regret: %{customdata[0]:.2f}<br>Lowest regret: %{customdata[1]:.2f}<br>Percentage completed: %{customdata[2]:.2f}%",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=data["Timestamp"],
                    y=-data["completed problem Average regret"]
                    + 1.96
                    * np.sqrt(data["completed problem Regret variance"])
                    / np.sqrt(data["completed problem Count"]),
                    fill=None,
                    fillcolor=lighten_colour(colours[i], 0.2),
                    mode="lines",
                    line=dict(color=lighten_colour(colours[i], 0.0)),
                    # line_shape="hv",
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=data["Timestamp"],
                    y=-data["completed problem Average regret"]
                    - 1.96
                    * np.sqrt(data["completed problem Regret variance"])
                    / np.sqrt(data["completed problem Count"]),
                    fill="tonexty",
                    fillcolor=lighten_colour(colours[i], 0.2),
                    mode="lines",
                    line=dict(color=lighten_colour(colours[i], 0.0)),
                    # line_shape="hv",
                    name="95% CI",
                    showlegend=False,
                )
            )

    fig.update_layout(
        # title=title,
        xaxis_title="Planning time (s)",
        yaxis_title="Regret",
        xaxis_range=[0, df["Timestamp"].max()],
        font=dict(size=16),
        width=250,
        height=250,
        showlegend=False,
    )
    fig.update_xaxes(
        domain=(0.15, 1),
        tickfont=dict(size=14),
    )
    fig.update_yaxes(
        domain=(0.15, 1),
        tickfont=dict(size=14),
    )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

    if output == "show":
        fig.show()
    elif output in ["png", "eps", "pdf"]:
        fig.write_image(f"{figname}.{output}")
    elif output == "html":
        fig.write_html(f"{figname}.html")


def plot_data(df: pd.DataFrame, dataname, ylabel, title, figname, output="show"):

    fig = go.Figure()

    for i, alg in enumerate(algs):
        data = df[df["Algorithm"] == alg].sort_values("Timestamp")
        fig.add_trace(
            go.Scatter(
                x=data["Timestamp"],
                line=dict(color=colours[i]),
                y=data[dataname],
                mode="markers" if len(data) == 1 else "lines",
                marker_symbol="cross",
                name=alg,
                # line_shape="hv",
            )
        )

    fig.update_layout(
        # title=title,
        xaxis_title="Planning time (s)",
        yaxis_title=ylabel,
        xaxis_range=[0, df["Timestamp"].max()],
        font=dict(size=16),
        showlegend=False,
        width=250,
        height=250,
    )
    fig.update_xaxes(
        domain=(0.15, 1),
        tickfont=dict(size=14),
    )
    fig.update_yaxes(
        domain=(0.15, 1),
        tickfont=dict(size=14),
    )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    if dataname == "policy nodes":
        fig.update_layout(
            yaxis_range=[0, df[df["Algorithm"] != "POMCP"][dataname].max() * 1.05]
        )

    if output == "show":
        fig.show()
    elif output in ["png", "eps", "pdf"]:
        fig.write_image(f"{figname}.{output}")
    elif output == "html":
        fig.write_html(f"{figname}.html")


def plot_legend(df, figname):
    fig = go.Figure()
    for i, alg in enumerate(algs):
        data = df[df["Algorithm"] == alg].sort_values("Timestamp")
        if len(data) < 1:
            continue
        fig.add_trace(
            go.Scatter(
                x=[1],
                y=[4],
                mode="markers" if len(data) == 1 else "lines",
                marker_symbol="cross",
                line=dict(color=colours[i]),
                name=alg,
                # visible="legendonly",
            )
        )

    # Update layout to remove the plot area and axes
    fig.update_layout(
        xaxis=dict(
            showline=False,
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            domain=(0.999, 1),
        ),
        yaxis=dict(
            showline=False,
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            domain=(0.999, 1),
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(255,255,255,1)",
        showlegend=True,
        legend=dict(
            x=0,  # Adjust legend position as needed
            y=1,  # Adjust legend position as needed
        ),
        font=dict(size=16),
        width=120,
        height=250,
    )

    # Remove margins
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))

    if output == "show":
        fig.show()
    elif output in ["png", "eps", "pdf"]:
        fig.write_image(f"{figname}.{output}")
    elif output == "html":
        fig.write_html(f"{figname}.html")


if __name__ == "__main__":
    output = "show"
    output = "png"
    output = "pdf"
    # output="html"

    series_file = "CTPInstance_plot.txt"
    # series_file2 = "experiments/Wumpus/evaluation/wumpus_results_2_2024-08-10_17-08/WumpusInstance_2.txt"

    df = parse_file(series_file)
    # df = df[df["Algorithm"] != "detMCVI"]
    # df2 = parse_file(series_file2)
    # df2 = df2[df2["Algorithm"] == "detMCVI"]
    # df = pd.concat([df, df2])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    plot_timeseries(df, "Policy value", "timeseries_regret", output)
    plot_data(
        df,
        "completed problem Percentage",
        "Success rate %",
        "Policy completion",
        "timeseries_success",
        output,
    )
    plot_data(
        df,
        "policy nodes",
        "Policy nodes",
        "Policy size",
        "timeseries_nodes",
        output,
    )
    plot_legend(df, "legend")
