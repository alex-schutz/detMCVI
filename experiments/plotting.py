import plotly.express as px
import pandas as pd
import math

output = "show"
# output="png"
# output="html"

files = [
    "eval_results_random_2024-05-03_14-09_9936525/ctp_results_random_all.csv",
    "eval_results_random_2024-05-04_08-33_1170464/ctp_results_random_all.csv",
    "eval_results_random_2024-05-06_09-13_2996797/ctp_results_random_all.csv",
]

df_from_each_file = (pd.read_csv(f) for f in files)
df = pd.concat(df_from_each_file, ignore_index=True)
df["Stochastic edge count"] = (df["State space size"] / df["Nodes"]).apply(math.log2)
df = df.rename(
    columns={
        "avg_reward_difference": "Difference in average reward",
        "Nodes": "CTP problem nodes",
    }
)

xlabel = "AO* policy nodes"
ylabel = "MCVI policy nodes"
xmax = df[xlabel].max()
ymax = df[xlabel].max()
range_xy = [0, max(xmax, ymax)]
fig = px.scatter(
    df,
    x=xlabel,
    y=ylabel,
    range_x=range_xy,
    range_y=range_xy,
    color="Difference in average reward",
    trendline="ols",
    title="Policy size",
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
    fig.write_image("policy_size.png", scale=2)
elif output == "html":
    fig.write_html("policy_size.html")

xlabel = "AO* runtime (s)"
ylabel = "MCVI runtime (s)"
xmax = df[xlabel].max()
ymax = df[xlabel].max()
range_xy = [0, max(xmax, ymax)]
fig = px.scatter(
    df,
    x=xlabel,
    y=ylabel,
    range_x=range_xy,
    range_y=range_xy,
    color="Difference in average reward",
    trendline="ols",
    title="Computation time",
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
    fig.write_image("runtime.png", scale=2)
elif output == "html":
    fig.write_html("runtime.html")
