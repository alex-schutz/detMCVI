import plotly.express as px

output = "show"
# output="png"
# output="html"

files = [
    "eval_results_random_2024-05-08_10-16/ctp_results_random.csv",
]


def plot_comparison(xlabel, ylabel, title, figname, output="show"):
    xmax = df[xlabel].max()
    ymax = df[ylabel].max()
    range_xy = [0, 1.05 * max(xmax, ymax)]
    fig = px.scatter(
        df,
        x=xlabel,
        y=ylabel,
        range_x=range_xy,
        range_y=range_xy,
        color="Difference in average reward from AO*",
        trendline="ols",
        title=title,
        color_continuous_midpoint=0,
        color_continuous_scale="Spectral",
        hover_data=[
            "CTP problem nodes",
            "State space size",
            "Initial belief size",
            "Stochastic edge count",
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

    df_from_each_file = (pd.read_csv(f) for f in files)
    df = pd.concat(df_from_each_file, ignore_index=True)
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
        "AO* policy nodes", "MCVI policy nodes", "Policy size", "policy_size", output
    )
    plot_comparison(
        "AO* runtime (s)", "MCVI runtime (s)", "Computation time", "runtime", output
    )
