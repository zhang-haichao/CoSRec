import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


IN_PATH = Path("results/paper/rq3/rq3_case_studies.json")
OUT_PATH = Path("results/paper/rq3/RQ3.png")


def short_title(title: str, limit: int = 28) -> str:
    if len(title) <= limit:
        return title
    return title[: limit - 3].rstrip() + "..."


def draw_rank_list(ax, x, y_values, titles, changed_index, color, align):
    for idx, (y, title) in enumerate(zip(y_values, titles), start=1):
        is_changed = idx - 1 == changed_index
        bbox = None
        weight = "normal"
        if is_changed:
            bbox = {
                "boxstyle": "round,pad=0.22",
                "facecolor": color,
                "edgecolor": "none",
                "alpha": 0.9,
            }
            weight = "bold"
        ax.text(
            x,
            y,
            f"{idx}. {short_title(title)}",
            ha=align,
            va="center",
            fontsize=8.5,
            fontweight=weight,
            bbox=bbox,
        )


def draw_case(ax, case):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    y_values = [0.72, 0.50, 0.28]
    user_x, user_y = 0.55, 0.50
    support_x = 0.42
    support_y_values = [0.75, 0.50, 0.25]

    target_index = 2
    ax.text(0.02, 0.92, f"User #{case['user_id']}", ha="left", va="top", fontsize=10, fontweight="bold")
    ax.text(0.13, 0.86, "Original window", ha="center", va="center", fontsize=8, color="#555555")
    ax.text(0.87, 0.86, "After removal", ha="center", va="center", fontsize=8, color="#555555")

    draw_rank_list(
        ax,
        0.08,
        y_values,
        case["original_recommendation_list"],
        target_index,
        "#F2A65A",
        "left",
    )
    draw_rank_list(
        ax,
        0.70,
        y_values,
        case["recommendation_list_after_removal"],
        target_index,
        "#77BFA3",
        "left",
    )

    ax.scatter([user_x], [user_y], s=460, color="#2F5597", edgecolor="white", linewidth=1.4, zorder=3)
    ax.text(user_x, user_y, "user", ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")

    support_items = case["counterfactual_support_set"]
    max_weight = max(item["contribution"] for item in support_items)
    for item, y in zip(support_items, support_y_values):
        width = 1.2 + 3.0 * item["contribution"] / max_weight
        ax.scatter([support_x], [y], s=120, color="#F7F7F7", edgecolor="#B22222", linewidth=1.1, zorder=3)
        ax.text(
            support_x - 0.02,
            y,
            short_title(item["title"], 24),
            ha="right",
            va="center",
            fontsize=7.8,
            color="#333333",
        )
        ax.add_line(
            Line2D(
                [user_x, support_x],
                [user_y, y],
                color="#B22222",
                linewidth=width,
                linestyle=(0, (4, 3)),
                alpha=0.75,
                zorder=1,
            )
        )
        ax.text(
            support_x + 0.18,
            y,
            f"w={item['contribution']:.4f}",
            ha="left",
            va="center",
            fontsize=7.5,
            color="#B22222",
        )
        ax.text(support_x + 0.025, y + 0.035, "x", ha="center", va="center", fontsize=10, color="#B22222")


def main() -> None:
    with IN_PATH.open("r", encoding="utf-8") as f:
        cases = json.load(f)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(len(cases), 1, figsize=(11, 7.2), layout="constrained")
    for ax, case in zip(axes, cases):
        draw_case(ax, case)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
