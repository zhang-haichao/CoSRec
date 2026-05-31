import json
import os

import matplotlib.pyplot as plt
import numpy as np


RESULT_5 = os.path.join(
    "results",
    "paper",
    "rq2",
    "rq2_budget5_summary.json",
)
RESULT_10 = os.path.join(
    "results",
    "paper",
    "rq2",
    "rq2_budget10_summary.json",
)
OUT_DIR = os.path.join("results", "paper", "rq2")

METHOD_ORDER = ["Random", "High-Degree", "Attention-proxy", "FIA", "CoSRec"]
COLOR_5 = ["#D9D9D9", "#C8CDD2", "#B7D4F6", "#B8E6E2", "#F6B8A8"]
COLOR_10 = ["#A6A6A6", "#8F979E", "#5B8FF9", "#36CFC9", "#D4380D"]


def load_success_map(path):
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    return {row["method"]: row["success_at_1"] for row in rows}


def main():
    success_5 = load_success_map(RESULT_5)
    success_10 = load_success_map(RESULT_10)

    labels = METHOD_ORDER
    values_5 = [success_5[label] for label in labels]
    values_10 = [success_10[label] for label in labels]

    x = np.arange(len(labels))
    width = 0.34

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
        }
    )

    fig, ax = plt.subplots(figsize=(9.2, 4.8))

    bars_5 = ax.bar(
        x - width / 2,
        values_5,
        width,
        label="5% budget",
        color=COLOR_5,
        edgecolor="black",
        linewidth=0.6,
    )
    bars_10 = ax.bar(
        x + width / 2,
        values_10,
        width,
        label="10% budget",
        color=COLOR_10,
        edgecolor="black",
        linewidth=0.6,
    )

    ax.set_ylabel("Success@1 (%)")
    ax.set_xlabel("Method")
    ax.set_title("RQ2 on MovieLens-100K (all 943 users)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(values_10) * 1.22)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc="upper left")

    for bars in (bars_5, bars_10):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.28,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUT_DIR, "rq2_success_at1_grouped_bar.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUT_DIR, "rq2_success_at1_grouped_bar.pdf"), bbox_inches="tight")


if __name__ == "__main__":
    main()
