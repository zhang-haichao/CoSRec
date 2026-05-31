#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strict RQ1 validity/specificity trade-off evaluation for CoSRec.

This script keeps the RQ1 table shape from the paper while using one consistent
evaluation protocol:
  - frozen LightGCN backbone,
  - discrete edge removal,
  - full-graph re-inference,
  - Success = target rank item changes,
  - Stability = Top-10 overlap after intervention.

For runtime, the default evaluates a seed-controlled user sample. Set
`--max_users 0` to evaluate all eligible users.
"""

import argparse
import csv
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from experiments.rq2.baseline_comparison import (
    build_edge_values,
    mask_seen,
    topk_for_user,
    user_edge_positions,
)
from experiments.rq3.case_study_ml100k import (
    lightgcn_forward_native,
    load_edge_csv_from_df,
    load_node_csv_from_df,
    split_by_user,
)
from recsys.LightGCN import LightGCN
from utils.utils import get_user_positive_items


DATASETS = ["ML-100K", "ML-1M"]
BUDGETS = [0.05, 0.10, 0.15, 0.20]
TARGET_RANKS = [0, 1, 2]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_dataset(dataset: str) -> pd.DataFrame:
    if dataset == "ML-100K":
        path = Path("data/ml-100k/u.data")
        return pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=["user_id", "item_id", "score", "timestamp"],
        )[["user_id", "item_id", "score"]]
    if dataset == "ML-1M":
        path = Path("data/ml1m/ratings.csv")
        df = pd.read_csv(path)
        return df.rename(
            columns={"movie_id": "item_id", "user_rating": "score"}
        )[["user_id", "item_id", "score"]]
    raise ValueError(f"Unsupported dataset: {dataset}")


def rating_threshold_for(dataset: str, args: argparse.Namespace) -> float:
    if dataset == "ML-100K":
        return args.rating_threshold_ml100k
    if dataset == "ML-1M":
        return args.rating_threshold_ml1m
    raise ValueError(dataset)


def build_positive_mask(edge_index: torch.Tensor, num_users: int, num_items: int) -> torch.Tensor:
    mask = torch.zeros((num_users, num_items), dtype=torch.bool)
    mask[edge_index[0].long(), edge_index[1].long()] = True
    return mask


def sample_negative_items(
    user_idx: torch.Tensor,
    positive_mask: torch.Tensor,
    num_items: int,
) -> torch.Tensor:
    neg_items = torch.randint(0, num_items, (user_idx.numel(),), device=user_idx.device)
    invalid = positive_mask[user_idx.cpu(), neg_items.cpu()].to(user_idx.device)
    while bool(invalid.any()):
        neg_items[invalid] = torch.randint(0, num_items, (int(invalid.sum().item()),), device=user_idx.device)
        invalid = positive_mask[user_idx.cpu(), neg_items.cpu()].to(user_idx.device)
    return neg_items


def bpr_loss(users_emb, items_emb, user_idx, pos_item_idx, neg_item_idx) -> torch.Tensor:
    pos_scores = (users_emb[user_idx] * items_emb[pos_item_idx]).sum(dim=1)
    neg_scores = (users_emb[user_idx] * items_emb[neg_item_idx]).sum(dim=1)
    return -F.logsigmoid(pos_scores - neg_scores).mean()


def train_lightgcn_if_needed(
    dataset: str,
    edge_index: torch.Tensor,
    full_edge_index: torch.Tensor,
    num_users: int,
    num_items: int,
    model_path: Path,
    args: argparse.Namespace,
) -> None:
    if model_path.exists():
        return

    print(f"[train] {dataset}: checkpoint missing, training {model_path}")
    model = LightGCN(num_users, num_items, K=args.layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.train_lr)
    positive_mask = build_positive_mask(full_edge_index, num_users, num_items)
    user_idx = edge_index[0].long()
    pos_item_idx = edge_index[1].long()
    include_initial = args.embedding_readout == "all_layers"

    for epoch in range(1, args.train_epochs + 1):
        model.train()
        optimizer.zero_grad()
        users_emb, items_emb = lightgcn_forward_native(
            model,
            edge_index,
            num_users + num_items,
            include_initial=include_initial,
        )
        neg_item_idx = sample_negative_items(user_idx, positive_mask, num_items)
        loss = bpr_loss(users_emb, items_emb, user_idx, pos_item_idx, neg_item_idx)
        reg = (
            model.users_emb(user_idx).pow(2).sum(dim=1).mean()
            + model.items_emb(pos_item_idx).pow(2).sum(dim=1).mean()
            + model.items_emb(neg_item_idx).pow(2).sum(dim=1).mean()
        )
        total = loss + args.weight_decay * reg
        total.backward()
        optimizer.step()
        if epoch == 1 or epoch % max(1, args.train_epochs // 5) == 0 or epoch == args.train_epochs:
            print(f"  epoch={epoch:03d} loss={loss.item():.4f} reg={reg.item():.4f}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"[train] saved {model_path}")


def load_model(dataset: str, num_users: int, num_items: int, args: argparse.Namespace) -> LightGCN:
    model_path = Path(args.ml100k_model if dataset == "ML-100K" else args.ml1m_model)
    model = LightGCN(num_users, num_items, K=args.layers)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def get_rating(model: LightGCN, train_edge_index: torch.Tensor, include_initial: bool) -> torch.Tensor:
    users_emb, items_emb = lightgcn_forward_native(
        model,
        train_edge_index,
        model.num_users + model.num_items,
        include_initial=include_initial,
    )
    return torch.matmul(users_emb, items_emb.T)


def optimize_mask(
    model: LightGCN,
    train_edge_index: torch.Tensor,
    user_idx: int,
    history: List[int],
    target_item: int,
    boundary_item: int,
    args: argparse.Namespace,
    include_initial: bool,
) -> torch.Tensor:
    positions = user_edge_positions(train_edge_index, user_idx)
    p_vec = torch.nn.Parameter(torch.ones(len(history)) * 0.89)
    optimizer = torch.optim.SGD([p_vec], lr=args.cosrec_lr)

    for _ in range(args.cosrec_epochs):
        optimizer.zero_grad()
        values = torch.ones(train_edge_index.shape[1])
        values[positions] = p_vec
        users_emb, items_emb = lightgcn_forward_native(
            model,
            train_edge_index,
            model.num_users + model.num_items,
            values,
            include_initial=include_initial,
        )
        target_score = torch.matmul(users_emb[user_idx], items_emb[int(target_item)])
        boundary_score = torch.matmul(users_emb[user_idx], items_emb[int(boundary_item)])
        cf_loss = target_score - boundary_score
        graph_loss = torch.mean((p_vec - 1.0) ** 2)
        l1_loss = torch.linalg.norm(p_vec, ord=1)
        loss = args.lambda_graph * graph_loss + args.beta * cf_loss + 0.0002 * l1_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p_vec], 2.0)
        optimizer.step()
        p_vec.data = torch.clamp(p_vec.data, min=0.01, max=1.0)

    return p_vec.detach()


def select_users(user_history: Dict[int, List[int]], args: argparse.Namespace) -> List[int]:
    eligible = [
        user_idx
        for user_idx, history in user_history.items()
        if len(history) >= max(args.min_history, math.floor(len(history) * max(BUDGETS)))
    ]
    if args.max_users <= 0 or args.max_users >= len(eligible):
        return eligible
    return sorted(random.Random(args.seed).sample(eligible, args.max_users))


def evaluate_dataset(dataset: str, args: argparse.Namespace) -> Tuple[List[Dict], Dict]:
    print(f"[dataset] {dataset}")
    df = load_dataset(dataset)
    user_mapping = load_node_csv_from_df(df, "user_id")
    item_mapping = load_node_csv_from_df(df, "item_id")
    edge_index = load_edge_csv_from_df(
        df,
        "user_id",
        user_mapping,
        "item_id",
        item_mapping,
        "score",
        rating_threshold=rating_threshold_for(dataset, args),
    )
    train_edge_index, _ = split_by_user(edge_index, len(user_mapping), args.seed)

    model_path = Path(args.ml100k_model if dataset == "ML-100K" else args.ml1m_model)
    train_lightgcn_if_needed(
        dataset,
        train_edge_index,
        edge_index,
        len(user_mapping),
        len(item_mapping),
        model_path,
        args,
    )
    model = load_model(dataset, len(user_mapping), len(item_mapping), args)
    include_initial = args.embedding_readout == "all_layers"

    user_history = {
        user_idx: train_edge_index[1, train_edge_index[0] == user_idx].tolist()
        for user_idx in range(len(user_mapping))
    }
    users = select_users(user_history, args)
    rating = mask_seen(get_rating(model, train_edge_index, include_initial), train_edge_index)
    original_top10 = {
        user_idx: torch.topk(rating[user_idx], k=10).indices.tolist()
        for user_idx in users
    }

    rows = []
    detailed = {"dataset": dataset, "num_users": len(users), "records": []}
    for target_rank in TARGET_RANKS:
        print(f"  target=Top-{target_rank + 1}, users={len(users)}")
        budget_success = {budget: [] for budget in BUDGETS}
        budget_stability = {budget: [] for budget in BUDGETS}
        budget_removed = {budget: [] for budget in BUDGETS}

        for n, user_idx in enumerate(users, start=1):
            history = user_history[user_idx]
            target_item = original_top10[user_idx][target_rank]
            boundary_item = original_top10[user_idx][min(target_rank + 1, 9)]
            p_vec = optimize_mask(
                model,
                train_edge_index,
                user_idx,
                history,
                target_item,
                boundary_item,
                args,
                include_initial,
            )
            sorted_local = torch.topk(p_vec, k=len(history), largest=False).indices.tolist()
            sorted_history = [history[i] for i in sorted_local]

            for budget in BUDGETS:
                k_edges = max(1, int(math.floor(len(history) * budget)))
                remove_items = sorted_history[: min(k_edges, len(sorted_history))]
                values = build_edge_values(train_edge_index, user_idx, remove_items)
                perturbed_top10 = topk_for_user(
                    model,
                    train_edge_index,
                    user_idx,
                    10,
                    values,
                    history,
                    include_initial,
                )
                success = int(perturbed_top10[target_rank] != target_item)
                stability = len(set(original_top10[user_idx]).intersection(perturbed_top10)) / 10.0
                budget_success[budget].append(success)
                budget_stability[budget].append(stability)
                budget_removed[budget].append(len(remove_items))
                detailed["records"].append(
                    {
                        "target_rank": target_rank + 1,
                        "budget": budget,
                        "mapped_user_id": int(user_idx),
                        "removed_edges": int(len(remove_items)),
                        "success": int(success),
                        "stability_at_10": float(stability),
                    }
                )

            if n % 25 == 0:
                print(f"    {n}/{len(users)} users")

        for budget in BUDGETS:
            rows.append(
                {
                    "dataset": dataset,
                    "target": f"Top-{target_rank + 1}",
                    "budget": budget,
                    "success": 100.0 * sum(budget_success[budget]) / len(budget_success[budget]),
                    "stability": 100.0 * sum(budget_stability[budget]) / len(budget_stability[budget]),
                    "avg_removed_edges": sum(budget_removed[budget]) / len(budget_removed[budget]),
                    "num_users": len(users),
                }
            )
    return rows, detailed


def write_outputs(rows: List[Dict], detailed: List[Dict], args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "rq1_strict_tradeoff_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "target",
                "budget",
                "success",
                "stability",
                "avg_removed_edges",
                "num_users",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "rq1_strict_tradeoff_detailed.json").open("w", encoding="utf-8") as f:
        json.dump(detailed, f, indent=2)

    paper_csv_path = output_dir / "rq1_main_results.csv"
    with paper_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "target",
                "budget_percent",
                "target_replaced_percent",
                "stability_percent",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "dataset": row["dataset"],
                    "target": row["target"],
                    "budget_percent": int(round(float(row["budget"]) * 100)),
                    "target_replaced_percent": f"{float(row['success']):.2f}",
                    "stability_percent": f"{float(row['stability']):.2f}",
                }
            )

    df = pd.DataFrame(rows)
    budgets = BUDGETS
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{threeparttable}",
        r"\caption{Validity-specificity trade-off of CoSRec under different sparsity budgets on sampled target cases. Target Replaced measures the percentage of sampled cases where the original target Top-$K$ recommendation is replaced after removing the learned support set. Stability measures how much of the remaining recommendation list is preserved after the intervention.}",
        r"\label{tab:main_results}",
        r"\begin{tabular}{clcccccccc}",
        r"\toprule",
        r"\multirow{2}{*}{Dataset} & \multirow{2}{*}{Target} & \multicolumn{4}{c}{Target Replaced (\%)} & \multicolumn{4}{c}{Stability (\%)} \\",
        r"\cmidrule(lr){3-6} \cmidrule(lr){7-10}",
        r"& & 5\% & 10\% & 15\% & 20\% & 5\% & 10\% & 15\% & 20\% \\",
        r"\midrule",
    ]
    for dataset_idx, dataset in enumerate(DATASETS):
        for target_idx, target in enumerate(["Top-1", "Top-2", "Top-3"]):
            subset = df[(df["dataset"] == dataset) & (df["target"] == target)].sort_values("budget")
            success = [float(subset[subset["budget"] == b]["success"].iloc[0]) for b in budgets]
            stability = [float(subset[subset["budget"] == b]["stability"].iloc[0]) for b in budgets]
            prefix = rf"\multirow{{3}}{{*}}{{{dataset}}}" if target_idx == 0 else ""
            lines.append(
                f"{prefix} & {target} & "
                + " & ".join(f"{v:.2f}" for v in success)
                + " & "
                + " & ".join(f"{v:.2f}" for v in stability)
                + r" \\"
            )
        if dataset_idx == 0:
            lines.append(r"\midrule")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{threeparttable}",
            r"\end{table*}",
        ]
    )
    (output_dir / "rq1_strict_tradeoff_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    plot_rq1(df, output_dir)
    print(f"[saved] {csv_path}")
    print(f"[saved] {output_dir / 'rq1_strict_tradeoff_table.tex'}")


def plot_rq1(df: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    top1 = df[df["target"] == "Top-1"].copy()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True, layout="constrained")
    for ax, dataset in zip(axes, DATASETS):
        sub = top1[top1["dataset"] == dataset].sort_values("budget")
        x = sub["budget"].to_numpy() * 100
        ax.plot(x, sub["success"], marker="o", linewidth=2.2, markersize=5.5, label="Target Replaced", color="#0072B2")
        ax.plot(x, sub["stability"], marker="s", linewidth=2.2, markersize=5.5, label="Stability", color="#D55E00")
        ax.set_title(dataset)
        ax.set_xlabel("Budget (%)")
        ax.set_xticks([5, 10, 15, 20])
        ax.grid(True, linestyle="--", linewidth=0.55, alpha=0.35)
        ax.legend(frameon=False, loc="best")
    axes[0].set_ylabel("Rate (%)")
    axes[0].set_ylim(0, 105)
    fig.savefig(output_dir / "rq1_top1_tradeoff.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output_dir / "rq1_top1_tradeoff.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strict RQ1 trade-off evaluation.")
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--max_users", type=int, default=100, help="0 means all eligible users.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rating_threshold_ml100k", type=float, default=3.0)
    parser.add_argument("--rating_threshold_ml1m", type=float, default=4.0)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--embedding_readout", choices=["all_layers", "propagation_only"], default="all_layers")
    parser.add_argument("--cosrec_epochs", type=int, default=8)
    parser.add_argument("--cosrec_lr", type=float, default=0.2)
    parser.add_argument("--lambda_graph", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--min_history", type=int, default=3)
    parser.add_argument("--train_epochs", type=int, default=30)
    parser.add_argument("--train_lr", type=float, default=0.005)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--ml100k_model", default="model/light_GCN_corrected_negfix.pt")
    parser.add_argument("--ml1m_model", default="model/light_GCN_ml1m_strict_t4.pt")
    parser.add_argument("--output_dir", default="results/paper/rq1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)
    set_seed(args.seed)

    all_rows = []
    all_detailed = []
    for dataset in args.datasets:
        rows, detailed = evaluate_dataset(dataset, args)
        all_rows.extend(rows)
        all_detailed.append(detailed)
    write_outputs(all_rows, all_detailed, args)


if __name__ == "__main__":
    main()
