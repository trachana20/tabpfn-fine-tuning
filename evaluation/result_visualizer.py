import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t
import pandas as pd


def visualize_accuracy_across_seeds(df):
    # Group by fold_i and calculate mean and confidence interval
    grouped = (
        df.groupby("fold")
        .agg(
            {
                "pre_eval": "mean",
                "post_eval": "mean",
            },
        )
        .reset_index()
    )
    # Calculate confidence intervals
    confidence_interval_pre_eval = [
        stats.norm.interval(0.95, loc=mean, scale=stats.sem(df["pre_eval"]))
        for mean in grouped["pre_eval"]
    ]
    confidence_interval_post_eval = [
        stats.norm.interval(0.95, loc=mean, scale=stats.sem(df["post_eval"]))
        for mean in grouped["post_eval"]
    ]

    # Plot the bar chart
    fig, ax = plt.subplots()
    index = np.arange(len(grouped))
    bar_width = 0.05
    opacity = 0.8

    rects1 = ax.bar(
        index,
        grouped["pre_eval"],
        bar_width,
        alpha=opacity,
        color="b",
        label="Pre Evaluation",
    )

    rects2 = ax.bar(
        index + bar_width,
        grouped["post_eval"],
        bar_width,
        alpha=opacity,
        color="r",
        label="Post Evaluation",
    )

    ax.set_xlabel("Fold")
    ax.set_ylabel("Mean Evaluation")
    ax.set_title("Mean Evaluation Across Folds (averaged over seeds)")
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(grouped["fold"])
    ax.legend()
