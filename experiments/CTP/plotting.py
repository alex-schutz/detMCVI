import plotly.express as px

output = "show"
# output="png"
# output="html"

files = [
    "eval_results_random_2024-05-16_13-30/ctp_results_random.csv",
]


def plot_comparison(df, xlabel, ylabel, title, figname, output="show"):
    xmax = df[xlabel].max()
    ymax = df[ylabel].max()
    xmin = df[xlabel].min()
    ymin = df[ylabel].min()
    range_xy = [
        min(xmin, ymin) - 0.05 * abs(min(xmin, ymin)),
        max(xmax, ymax) + 0.05 * abs(max(xmax, ymax)),
    ]
    fig = px.scatter(
        df,
        x=xlabel,
        y=ylabel,
        range_x=range_xy,
        range_y=range_xy,
        color="percentage_complete_difference",
        trendline="ols",
        title=title,
        color_continuous_midpoint=0,
        color_continuous_scale="Spectral",
        hover_data=[
            "CTP problem nodes",
            "State space size",
            "Initial belief size",
            "Stochastic edge count",
            "MCVI runtime (s)",
            "AO* runtime (s)",
            "AO* completed problem Average reward",
            "MCVI completed problem Average reward",
            "MCVI completed problem Percentage",
            "AO* completed problem Percentage",
        ],
    )
    fig.update_xaxes(constrain="domain")
    fig.update_yaxes(scaleanchor="x")
    if output == "show":
        fig.show()
    elif output == "png":
        fig.write_image(f"{figname}.png", scale=2)
    elif output == "html":
        fig.write_html(f"{figname}.html")


if __name__ == "__main__":
    import pandas as pd
    import math
    import numpy as np

    df_from_each_file = (pd.read_csv(f) for f in files)
    df = pd.concat(df_from_each_file, ignore_index=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df["Stochastic edge count"] = (df["State space size"] / df["Nodes"]).apply(
        math.log2
    )
    df = df.rename(
        columns={
            "avg_reward_difference": "Difference in average reward from AO*",
            "Nodes": "CTP problem nodes",
        }
    )

    plot_comparison(
        df,
        "AO* policy nodes",
        "MCVI policy nodes",
        "Policy size",
        "policy_size",
        output,
    )
    plot_comparison(
        df,
        "AO* completed problem Average reward",
        "MCVI completed problem Average reward",
        "Average reward",
        "reward",
        output,
    )
