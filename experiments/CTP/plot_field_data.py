import plotly.graph_objects as go
import pandas as pd
from time_series import lighten_colour
import numpy as np

folder = "/home/alex/ori/detMCVI/experiments/CTP/evaluation/field_trial_3"

full_df = pd.read_csv(f"{folder}/ctp_results_all.csv")

result = (
    full_df.groupby(["Set number", "Algorithm"])
    .apply(
        lambda group: (
            # group[group["completed problem Percentage"] >= 99].iloc[0]
            # if any(group["completed problem Percentage"] >= 99)
            # else
            group.iloc[-1]
        )
    )
    .reset_index(drop=True)
)


def do_plot(df, y, ytitle, mul=1, offset=0, output="show"):
    colours = ["#cb6ce6", "#90e079", "#f5bd45", "#0097b2"]
    x = "Algorithm"
    alg_mapping = {"detMCVI": 0, "AO*": 1, "QMDP": 2}
    # alg_mapping = {a: i for i, a in enumerate(df[x].unique())}
    df["alg num"] = df[x].map(alg_mapping)

    df["jitter"] = df["alg num"] + np.linspace(-0.2, 0.2, num=len(df))

    fig = go.Figure()
    fig.add_trace(
        go.Box(
            y=offset + mul * df[y],
            x=df["alg num"],
            boxpoints=False,
            showlegend=False,
            fillcolor="rgba(0,0,0,0)",
            line=dict(color=colours[-1]),
            zorder=1,
        )
    )

    fig.add_trace(
        go.Scatter(
            y=offset + mul * df[y],
            x=df["jitter"],
            mode="markers",
            marker=dict(
                color=[
                    lighten_colour(colours[(i % (len(colours) - 1))], 0.7)
                    for i in result["alg num"]
                ],
                # size=10,
            ),
            showlegend=False,
            text=result["Set number"],
            zorder=0,
        )
    )

    fig.update_layout(
        xaxis=dict(
            tickmode="array",
            tickvals=list(alg_mapping.values()),
            ticktext=list(alg_mapping.keys()),
            title="Algorithm",
        ),
        yaxis=(
            dict(title=ytitle, tickformat="~s")
            if df[y].max() > 1000
            else dict(title=ytitle)
        ),
        width=250,
        height=250,
    )

    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        font=dict(size=16),
    )

    fig.update_yaxes(minor=dict(tickmode="auto", nticks=5, showgrid=True))

    if output == "show":
        fig.show()
    elif output in ["png", "eps", "pdf"]:
        fig.write_image(f"{ytitle}.{output}")
    elif output == "html":
        fig.write_html(f"{ytitle}.html")


output = "show"
do_plot(
    result,
    "completed problem Percentage",
    "Failure rate (%)",
    -1,
    offset=100,
    output=output,
)
do_plot(result, "completed problem Average regret", "Average regret", -1, output=output)
do_plot(result, "policy nodes", "Policy nodes", output=output)
