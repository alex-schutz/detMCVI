import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from time_series import lighten_colour
import numpy as np
import os

folder = "/home/alex/ori/detMCVI/experiments/CTP/evaluation/ctp_downsample_9"
prefixed = [
    filename for filename in os.listdir(folder) if filename.startswith("ctp_results_9_")
]

dfs = [pd.read_csv(folder + "/" + p) for p in prefixed]
df = pd.concat(dfs)


df["Success rate (%)"] = df["completed problem Count"] / 100
df["Initial belief sample size (%)"] = df["Set number"]
result = (
    df.groupby(["Set number", "Algorithm"])
    .apply(lambda group: (group.iloc[-1]))
    .reset_index(drop=True)
)
# result.drop(result[result["Timestamp"] >= 3600].index, inplace=True)


fig = px.scatter(
    result,
    "Initial belief sample size (%)",
    "Success rate (%)",
    "Algorithm",
    log_x=False,
)
for i in range(len(fig.data)):
    fig.data[i].update(mode="markers+lines")
fig.show()
