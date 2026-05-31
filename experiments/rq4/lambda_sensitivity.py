#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run CoSRec sparsity-weight sensitivity analysis.

This script intentionally reuses the active RQ2 evaluator instead of
duplicating metric code. The evaluator computes Success@1, Success@5,
Stability@10, and average removed edges in:

    experiments.rq2.baseline_comparison

Each lambda value is executed in an isolated subprocess so one failed setting
does not poison the remaining runs.
"""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


LAMBDA_GRID = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
RESULT_COLUMNS = [
    "lambda",
    "status",
    "Success@1",
    "Success@5",
    "Stability@10",
    "Avg_Removed_Edges",
    "num_users",
    "budget_percent",
    "cosrec_epochs",
    "cosrec_lr",
    "beta",
    "objective",
    "graph_scope",
    "embedding_readout",
    "selection_mode",
    "output_dir",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sensitivity analysis for the CoSRec sparsity/graph-loss weight."
    )
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=LAMBDA_GRID,
        help="Sparsity weights to evaluate.",
    )
    parser.add_argument(
        "--budget_percent",
        type=float,
        default=0.05,
        help="Fixed percentage of each user's history to remove. Use 0.10 for the 10%% budget.",
    )
    parser.add_argument("--k_edges", type=int, default=3, help="Fallback fixed-k budget if budget_percent is 0.")
    parser.add_argument("--max_users", type=int, default=943, help="Maximum users evaluated by the RQ2 script.")
    parser.add_argument(
        "--ordered_users",
        action="store_true",
        help="Use the first max_users eligible users instead of a seed-controlled random subset.",
    )
    parser.add_argument("--min_history", type=int, default=3)
    parser.add_argument("--target_rank", type=int, default=0, help="0-based recommendation rank to explain.")
    parser.add_argument(
        "--success_metric",
        choices=["top1_changed", "target_rank_changed", "target_not_in_top10"],
        default="top1_changed",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cosrec_epochs", type=int, default=20)
    parser.add_argument("--cosrec_lr", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--cosrec_objective", choices=["score", "boundary_margin"], default="boundary_margin")
    parser.add_argument("--hessian_candidates", type=int, default=8)
    parser.add_argument("--model_path", default=os.path.join("model", "light_GCN_corrected_negfix.pt"))
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--graph_scope", choices=["full", "user_local"], default="full")
    parser.add_argument("--embedding_readout", choices=["all_layers", "propagation_only"], default="all_layers")
    parser.add_argument("--target_pairs_file", default="")
    parser.add_argument("--target_pairs", default="")
    parser.add_argument("--user_ids", default="")
    parser.add_argument("--results_csv", default=os.path.join("results", "paper", "rq4", "rq4_lambda_sensitivity.csv"))
    parser.add_argument("--plot_path", default=os.path.join("results", "paper", "rq4", "rq4_lambda_sensitivity.pdf"))
    parser.add_argument(
        "--work_dir",
        default=os.path.join("results", "paper", "rq4", "lambda_runs"),
        help="Directory for per-lambda RQ2 evaluator outputs and logs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip lambda values already marked ok in the CSV.",
    )
    parser.add_argument(
        "--plot_metric",
        choices=["Avg_Removed_Edges", "Stability@10"],
        default="Avg_Removed_Edges",
        help="Right-axis metric for the generated PDF.",
    )
    parser.add_argument(
        "--plot_max_lambda",
        type=float,
        default=0.0,
        help="If positive, only plot successful rows with lambda <= this value.",
    )
    parser.add_argument(
        "--selection_mode",
        choices=["adaptive_expected", "fixed_budget_evaluator"],
        default="adaptive_expected",
        help=(
            "adaptive_expected converts the learned retention mask to a variable number of "
            "removed edges via round(sum(1-p)), capped by budget_percent. "
            "fixed_budget_evaluator delegates to the RQ2 baseline-comparison script."
        ),
    )
    return parser.parse_args()


def lambda_label(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def existing_successful_lambdas(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    try:
        rows = pd.read_csv(csv_path)
    except Exception:
        return set()
    if "lambda" not in rows or "status" not in rows:
        return set()
    ok_rows = rows[rows["status"] == "ok"]
    return {float(value) for value in ok_rows["lambda"].dropna().tolist()}


def ensure_csv_header(csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists() and csv_path.stat().st_size > 0:
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            existing_header = next(reader, [])
            has_rows = any(True for _ in reader)
        if existing_header == RESULT_COLUMNS:
            return
        if has_rows:
            raise RuntimeError(
                f"{csv_path} has an incompatible header and existing rows. "
                "Move it aside or choose --results_csv before rerunning."
            )
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()


def append_row(csv_path: Path, row: Dict[str, object]) -> None:
    ensure_csv_header(csv_path)
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writerow({key: row.get(key, "") for key in RESULT_COLUMNS})
        f.flush()
        os.fsync(f.fileno())


def evaluator_command(lambda_value: float, output_dir: Path, args: argparse.Namespace) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "experiments.rq2.baseline_comparison",
        "--k_edges",
        str(args.k_edges),
        "--budget_percent",
        str(args.budget_percent),
        "--max_users",
        str(args.max_users),
        "--min_history",
        str(args.min_history),
        "--target_rank",
        str(args.target_rank),
        "--success_metric",
        args.success_metric,
        "--seed",
        str(args.seed),
        "--methods",
        "CoSRec",
        "--cosrec_epochs",
        str(args.cosrec_epochs),
        "--cosrec_lr",
        str(args.cosrec_lr),
        "--lambda_graph",
        str(lambda_value),
        "--beta",
        str(args.beta),
        "--cosrec_objective",
        args.cosrec_objective,
        "--hessian_candidates",
        str(args.hessian_candidates),
        "--model_path",
        args.model_path,
        "--layers",
        str(args.layers),
        "--graph_scope",
        args.graph_scope,
        "--embedding_readout",
        args.embedding_readout,
        "--output_dir",
        str(output_dir),
    ]
    if args.target_pairs_file:
        cmd.extend(["--target_pairs_file", args.target_pairs_file])
    if args.target_pairs:
        cmd.extend(["--target_pairs", args.target_pairs])
    if args.user_ids:
        cmd.extend(["--user_ids", args.user_ids])
    return cmd


def load_cosrec_summary(output_dir: Path) -> Dict[str, object]:
    candidates = sorted(output_dir.glob("rq2_*_summary.json"))
    if not candidates:
        raise FileNotFoundError(f"No RQ2 summary JSON found in {output_dir}")
    summary_path = candidates[0]
    with summary_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    for row in rows:
        if row.get("method") == "CoSRec":
            return row
    raise RuntimeError(f"No CoSRec row found in {summary_path}")


def run_lambda(lambda_value: float, args: argparse.Namespace) -> Dict[str, object]:
    output_dir = Path(args.work_dir) / f"lambda_{lambda_label(lambda_value)}"
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = evaluator_command(lambda_value, output_dir, args)
    log_path = output_dir / "run.log"
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("KMP_INIT_AT_FORK", "FALSE")

    with log_path.open("w", encoding="utf-8") as log:
        log.write("Command: " + " ".join(cmd) + "\n\n")
        log.flush()
        completed = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            env=env,
        )

    if completed.returncode != 0:
        raise RuntimeError(f"Evaluator exited with code {completed.returncode}. See {log_path}")

    summary = load_cosrec_summary(output_dir)
    return {
        "lambda": lambda_value,
        "status": "ok",
        "Success@1": float(summary["success_at_1"]),
        "Success@5": float(summary["success_at_5"]),
        "Stability@10": float(summary["stability_at_10"]),
        "Avg_Removed_Edges": float(summary["avg_removed_edges"]),
        "num_users": int(summary["num_users"]),
        "budget_percent": float(summary.get("budget_percent", args.budget_percent)),
        "cosrec_epochs": int(args.cosrec_epochs),
        "cosrec_lr": float(args.cosrec_lr),
        "beta": float(args.beta),
        "objective": args.cosrec_objective,
        "graph_scope": args.graph_scope,
        "embedding_readout": args.embedding_readout,
        "selection_mode": args.selection_mode,
        "output_dir": str(output_dir),
        "error": "",
    }


def optimize_cosrec_mask(
    model,
    train_edge_index,
    user_idx: int,
    history: List[int],
    target_item: int,
    boundary_item: Optional[int],
    lambda_value: float,
    args: argparse.Namespace,
    include_initial: bool,
):
    import torch

    from experiments.rq2.baseline_comparison import user_edge_positions
    from experiments.rq3.case_study_ml100k import lightgcn_forward_native

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
        if args.cosrec_objective == "boundary_margin" and boundary_item is not None:
            boundary_score = torch.matmul(users_emb[user_idx], items_emb[int(boundary_item)])
            cf_loss = target_score - boundary_score
        else:
            cf_loss = target_score
        graph_loss = torch.mean((p_vec - 1.0) ** 2)
        l1_loss = torch.linalg.norm(p_vec, ord=1)
        loss = lambda_value * graph_loss + args.beta * cf_loss + 0.0002 * l1_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p_vec], 2.0)
        optimizer.step()
        p_vec.data = torch.clamp(p_vec.data, min=0.01, max=1.0)

    return p_vec.detach().cpu()


def adaptive_removed_items(p_vec, history: List[int], max_edges: int) -> List[int]:
    import torch

    expected_removed = int(round(float(torch.clamp(1.0 - p_vec, min=0.0).sum().item())))
    remove_count = min(max_edges, max(0, expected_removed), len(history))
    if remove_count <= 0:
        return []
    selected = torch.topk(p_vec, k=remove_count, largest=False).indices.tolist()
    return [history[i] for i in selected]


def run_adaptive_sensitivity(args: argparse.Namespace) -> None:
    import random
    import time

    import torch

    from data.ml_100k import ml_100k_DataLoader
    from experiments.rq2.baseline_comparison import (
        build_edge_values,
        get_rating,
        inference_edges_for_user,
        mask_seen,
        removal_budget,
        topk_for_user,
    )
    from experiments.rq3.case_study_ml100k import load_edge_csv_from_df, load_node_csv_from_df, split_by_user
    from recsys.LightGCN import LightGCN

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)

    csv_path = Path(args.results_csv)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    data_loader = ml_100k_DataLoader(dataset_name="ml-100k")
    origin_data = data_loader.load_data()
    user_mapping = load_node_csv_from_df(origin_data, "user_id")
    movie_mapping = load_node_csv_from_df(origin_data, "item_id")
    edge_index = load_edge_csv_from_df(
        origin_data,
        "user_id",
        user_mapping,
        "item_id",
        movie_mapping,
        "score",
        rating_threshold=3,
    )
    train_edge_index, _ = split_by_user(edge_index, len(user_mapping), args.seed)

    model = LightGCN(len(user_mapping), len(movie_mapping), K=args.layers)
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model.eval()

    user_history = {
        user_idx: train_edge_index[1, train_edge_index[0] == user_idx].tolist()
        for user_idx in range(len(user_mapping))
    }
    eligible_users = [
        user_idx
        for user_idx, history in user_history.items()
        if len(history) >= max(args.min_history, removal_budget(len(history), args))
    ]
    if args.ordered_users or args.max_users >= len(eligible_users):
        candidate_users = eligible_users[: args.max_users]
    else:
        candidate_users = sorted(random.Random(args.seed).sample(eligible_users, args.max_users))
    target_pairs = [(user_idx, args.target_rank) for user_idx in candidate_users]
    include_initial = args.embedding_readout == "all_layers"

    if args.graph_scope == "full":
        original_rating = mask_seen(get_rating(model, train_edge_index, include_initial=include_initial), train_edge_index)
        original_top10 = {
            user_idx: torch.topk(original_rating[user_idx], k=10).indices.tolist()
            for user_idx in candidate_users
        }
    else:
        original_top10 = {}
        for user_idx in candidate_users:
            history = user_history[user_idx]
            local_edge_index = inference_edges_for_user(train_edge_index, user_idx, args.graph_scope)
            original_top10[user_idx] = topk_for_user(
                model,
                local_edge_index,
                user_idx,
                10,
                None,
                history,
                include_initial,
            )

    print(
        f"Adaptive sensitivity on {len(target_pairs)} target instances, "
        f"budget cap={'{:.0%}'.format(args.budget_percent) if args.budget_percent > 0 else 'fixed k=' + str(args.k_edges)}, "
        f"epochs={args.cosrec_epochs}, beta={args.beta}, objective={args.cosrec_objective}"
    )

    completed = existing_successful_lambdas(csv_path) if args.resume else set()
    for lambda_value in args.lambdas:
        if args.resume and float(lambda_value) in completed:
            print(f"[skip] lambda={lambda_value:g} already has a successful CSV row")
            continue

        print(f"[run] lambda={lambda_value:g}")
        start_time = time.time()
        output_dir = work_dir / f"lambda_{lambda_label(float(lambda_value))}"
        output_dir.mkdir(parents=True, exist_ok=True)
        detailed = []

        try:
            success_at_1_values = []
            success_at_5_values = []
            stabilities = []
            removed_counts = []
            for n, (user_idx, target_rank0) in enumerate(target_pairs, start=1):
                history = user_history[user_idx]
                eval_edge_index = inference_edges_for_user(train_edge_index, user_idx, args.graph_scope)
                target_item = original_top10[user_idx][target_rank0]
                boundary_item = original_top10[user_idx][min(target_rank0 + 1, 9)]
                max_edges = removal_budget(len(history), args)

                p_vec = optimize_cosrec_mask(
                    model,
                    eval_edge_index,
                    user_idx,
                    history,
                    target_item,
                    boundary_item,
                    float(lambda_value),
                    args,
                    include_initial,
                )
                remove_items = adaptive_removed_items(p_vec, history, max_edges)
                values = build_edge_values(eval_edge_index, user_idx, remove_items)
                perturbed_top10 = topk_for_user(
                    model,
                    eval_edge_index,
                    user_idx,
                    10,
                    values,
                    history,
                    include_initial,
                )

                original_top1 = original_top10[user_idx][0]
                success_at_1 = int(perturbed_top10[0] != original_top1)
                success_at_5 = int(original_top1 not in perturbed_top10[:5])
                stability = len(set(original_top10[user_idx]).intersection(perturbed_top10)) / 10.0

                success_at_1_values.append(success_at_1)
                success_at_5_values.append(success_at_5)
                stabilities.append(stability)
                removed_counts.append(len(remove_items))
                detailed.append(
                    {
                        "lambda": float(lambda_value),
                        "mapped_user_id": int(user_idx),
                        "target_rank": int(target_rank0 + 1),
                        "removed_edges": int(len(remove_items)),
                        "expected_removed_edges": float(torch.clamp(1.0 - p_vec, min=0.0).sum().item()),
                        "success_at_1": int(success_at_1),
                        "success_at_5": int(success_at_5),
                        "stability_at_10": float(stability),
                    }
                )

                if n % 50 == 0:
                    print(f"  lambda={lambda_value:g}: {n}/{len(target_pairs)} target instances")

            row = {
                "lambda": float(lambda_value),
                "status": "ok",
                "Success@1": 100.0 * sum(success_at_1_values) / len(success_at_1_values),
                "Success@5": 100.0 * sum(success_at_5_values) / len(success_at_5_values),
                "Stability@10": sum(stabilities) / len(stabilities),
                "Avg_Removed_Edges": sum(removed_counts) / len(removed_counts),
                "num_users": len(target_pairs),
                "budget_percent": float(args.budget_percent),
                "cosrec_epochs": int(args.cosrec_epochs),
                "cosrec_lr": float(args.cosrec_lr),
                "beta": float(args.beta),
                "objective": args.cosrec_objective,
                "graph_scope": args.graph_scope,
                "embedding_readout": args.embedding_readout,
                "selection_mode": args.selection_mode,
                "output_dir": str(output_dir),
                "error": "",
            }
            append_row(csv_path, row)
            with (output_dir / "lambda_sensitivity_detailed.json").open("w", encoding="utf-8") as f:
                json.dump(detailed, f, indent=2)
            print(
                "[ok] lambda={lambda_value:g} Success@1={s1:.2f}% Success@5={s5:.2f}% "
                "Stability@10={stab:.4f} Avg_Removed_Edges={removed:.2f} elapsed={elapsed:.2f} min".format(
                    lambda_value=float(lambda_value),
                    s1=row["Success@1"],
                    s5=row["Success@5"],
                    stab=row["Stability@10"],
                    removed=row["Avg_Removed_Edges"],
                    elapsed=(time.time() - start_time) / 60.0,
                )
            )
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            append_row(
                csv_path,
                {
                    "lambda": float(lambda_value),
                    "status": "failed",
                    "budget_percent": args.budget_percent,
                    "cosrec_epochs": args.cosrec_epochs,
                    "cosrec_lr": args.cosrec_lr,
                    "beta": args.beta,
                    "objective": args.cosrec_objective,
                    "graph_scope": args.graph_scope,
                    "embedding_readout": args.embedding_readout,
                    "selection_mode": args.selection_mode,
                    "output_dir": str(output_dir),
                    "error": error_text,
                },
            )
            print(f"[failed] lambda={lambda_value:g}: {error_text}", file=sys.stderr)
            traceback.print_exc()
            continue

    plotted = plot_results(
        csv_path,
        Path(args.plot_path),
        args.plot_metric,
        args.plot_max_lambda if args.plot_max_lambda > 0 else None,
    )
    if plotted:
        print(f"[plot] saved {plotted}")
    print(f"[csv] saved {csv_path}")


def plot_results(
    csv_path: Path,
    plot_path: Path,
    right_metric: str,
    max_lambda: Optional[float] = None,
) -> Optional[Path]:
    if not csv_path.exists():
        print(f"CSV not found, skipping plot: {csv_path}")
        return None

    data = pd.read_csv(csv_path)
    if data.empty:
        print(f"CSV has no rows, skipping plot: {csv_path}")
        return None

    data = data[data["status"] == "ok"].copy()
    if data.empty:
        print("No successful lambda runs found, skipping plot.")
        return None

    data["lambda"] = pd.to_numeric(data["lambda"], errors="coerce")
    data["Success@1"] = pd.to_numeric(data["Success@1"], errors="coerce")
    data[right_metric] = pd.to_numeric(data[right_metric], errors="coerce")
    data = data.dropna(subset=["lambda", "Success@1", right_metric]).sort_values("lambda")
    if max_lambda is not None and max_lambda > 0:
        data = data[data["lambda"] <= max_lambda]
    if data.empty:
        print("No plottable successful rows found, skipping plot.")
        return None

    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax_left = plt.subplots(figsize=(6.2, 3.8))
    blue = "#1f77b4"
    red = "#d62728"

    ax_left.plot(
        data["lambda"],
        data["Success@1"],
        color=blue,
        marker="o",
        linewidth=2.2,
        markersize=5.5,
        label="Success@1",
    )
    ax_left.set_xlabel(r"Sparsity weight $\lambda$")
    ax_left.set_ylabel("Success@1 (%)", color=blue)
    ax_left.tick_params(axis="y", labelcolor=blue)
    ax_left.grid(True, which="major", axis="both", linestyle="--", linewidth=0.6, alpha=0.35)
    if len(data["lambda"].unique()) > 1 and data["lambda"].min() > 0:
        ax_left.set_xscale("log")

    ax_right = ax_left.twinx()
    ax_right.plot(
        data["lambda"],
        data[right_metric],
        color=red,
        marker="s",
        linewidth=2.2,
        markersize=5.5,
        label=right_metric.replace("_", " "),
    )
    ylabel = "Avg. removed edges" if right_metric == "Avg_Removed_Edges" else "Stability@10"
    ax_right.set_ylabel(ylabel, color=red)
    ax_right.tick_params(axis="y", labelcolor=red)
    if right_metric == "Stability@10":
        ymin = max(0.0, float(data[right_metric].min()) - 0.02)
        ymax = min(1.0, float(data[right_metric].max()) + 0.02)
        if math.isclose(ymin, ymax):
            ymin, ymax = max(0.0, ymin - 0.01), min(1.0, ymax + 0.01)
        ax_right.set_ylim(ymin, ymax)

    lines = ax_left.get_lines() + ax_right.get_lines()
    labels = [line.get_label() for line in lines]
    ax_left.legend(lines, labels, loc="best", frameon=False)

    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def plot_results_with_stability(
    csv_path: Path,
    plot_path: Path,
    max_lambda: Optional[float] = None,
) -> Optional[Path]:
    if not csv_path.exists():
        print(f"CSV not found, skipping plot: {csv_path}")
        return None

    data = pd.read_csv(csv_path)
    data = data[data["status"] == "ok"].copy()
    for column in ["lambda", "Success@1", "Avg_Removed_Edges", "Stability@10"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["lambda", "Success@1", "Avg_Removed_Edges", "Stability@10"]).sort_values("lambda")
    if max_lambda is not None and max_lambda > 0:
        data = data[data["lambda"] <= max_lambda]
    if data.empty:
        print("No plottable successful rows found, skipping plot.")
        return None

    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    blue = "#1f77b4"
    red = "#d62728"
    green = "#2ca02c"

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(5.2, 3.35),
        sharex=True,
        layout="constrained",
        gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.01},
    )

    ax_top.plot(
        data["lambda"],
        data["Success@1"],
        color=blue,
        marker="o",
        linewidth=2.35,
        markersize=5.8,
        label="Success@1",
    )
    ax_top.set_ylabel("Success@1 (%)", color=blue, labelpad=4)
    ax_top.tick_params(axis="y", labelcolor=blue)
    ax_top.grid(True, which="major", axis="both", linestyle="--", linewidth=0.6, alpha=0.35)
    success_min = float(data["Success@1"].min())
    success_max = float(data["Success@1"].max())
    ax_top.set_ylim(max(0.0, success_min - 2.0), success_max + 2.0)

    ax_top_right = ax_top.twinx()
    ax_top_right.plot(
        data["lambda"],
        data["Avg_Removed_Edges"],
        color=red,
        marker="s",
        linewidth=2.35,
        markersize=5.8,
        label="Avg. removed edges",
    )
    ax_top_right.set_ylabel("Removed edges", color=red, labelpad=2)
    ax_top_right.tick_params(axis="y", labelcolor=red)
    removed_min = float(data["Avg_Removed_Edges"].min())
    removed_max = float(data["Avg_Removed_Edges"].max())
    ax_top_right.set_ylim(max(0.0, removed_min - 0.35), removed_max + 0.35)

    top_lines = ax_top.get_lines() + ax_top_right.get_lines()
    ax_top.legend(
        top_lines,
        [line.get_label() for line in top_lines],
        loc="lower left",
        frameon=False,
        handlelength=1.8,
        borderaxespad=0.2,
        labelspacing=0.25,
    )

    ax_bottom.plot(
        data["lambda"],
        data["Stability@10"],
        color=green,
        marker="^",
        linewidth=2.35,
        markersize=5.8,
        label="Stability@10",
    )
    ax_bottom.set_ylabel("Stability@10", labelpad=4)
    ax_bottom.set_xlabel(r"Sparsity weight $\lambda$", labelpad=4)
    ax_bottom.grid(True, which="major", axis="both", linestyle="--", linewidth=0.6, alpha=0.35)
    ax_bottom.legend(
        loc="upper left",
        frameon=False,
        handlelength=1.8,
        borderaxespad=0.2,
        labelspacing=0.25,
    )
    ymin = max(0.0, float(data["Stability@10"].min()) - 0.015)
    ymax = min(1.0, float(data["Stability@10"].max()) + 0.005)
    ax_bottom.set_ylim(ymin, ymax)

    if len(data["lambda"].unique()) > 1 and data["lambda"].min() > 0:
        ax_top.set_xscale("log")
        ax_bottom.set_xscale("log")

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def plot_results_three_panel(
    csv_path: Path,
    plot_path: Path,
    max_lambda: Optional[float] = None,
) -> Optional[Path]:
    if not csv_path.exists():
        print(f"CSV not found, skipping plot: {csv_path}")
        return None

    data = pd.read_csv(csv_path)
    data = data[data["status"] == "ok"].copy()
    for column in ["lambda", "Success@1", "Avg_Removed_Edges", "Stability@10"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["lambda", "Success@1", "Avg_Removed_Edges", "Stability@10"]).sort_values("lambda")
    if max_lambda is not None and max_lambda > 0:
        data = data[data["lambda"] <= max_lambda]
    if data.empty:
        print("No plottable successful rows found, skipping plot.")
        return None

    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    specs = [
        ("Success@1", "Success@1 (%)", "#1f77b4", "o", (0, max(30.0, float(data["Success@1"].max()) + 2.0))),
        (
            "Removed edges",
            "Avg. edges",
            "#d62728",
            "s",
            (0, max(7.0, float(data["Avg_Removed_Edges"].max()) + 0.4)),
        ),
        (
            "Stability@10",
            "Stability",
            "#2ca02c",
            "^",
            (
                max(0.0, float(data["Stability@10"].min()) - 0.012),
                min(1.0, float(data["Stability@10"].max()) + 0.004),
            ),
        ),
    ]
    columns = ["Success@1", "Avg_Removed_Edges", "Stability@10"]

    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.25), layout="constrained")
    for ax, column, (title, ylabel, color, marker, ylim) in zip(axes, columns, specs):
        ax.plot(
            data["lambda"],
            data[column],
            color=color,
            marker=marker,
            linewidth=2.4,
            markersize=5.5,
        )
        ax.set_xscale("log")
        ax.set_title(title, pad=5)
        ax.set_xlabel(r"$\lambda$", labelpad=1)
        ax.set_ylabel(ylabel, labelpad=2)
        ax.set_ylim(*ylim)
        ax.grid(True, which="major", axis="both", linestyle="--", linewidth=0.55, alpha=0.38)
        ax.tick_params(axis="both", which="major", pad=2)

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return plot_path


def main() -> int:
    args = parse_args()
    if args.selection_mode == "adaptive_expected":
        run_adaptive_sensitivity(args)
        return 0

    csv_path = Path(args.results_csv)
    plot_path = Path(args.plot_path)
    ensure_csv_header(csv_path)

    completed = existing_successful_lambdas(csv_path) if args.resume else set()
    for lambda_value in args.lambdas:
        if args.resume and float(lambda_value) in completed:
            print(f"[skip] lambda={lambda_value:g} already has a successful CSV row")
            continue

        print(f"[run] lambda={lambda_value:g}")
        try:
            row = run_lambda(float(lambda_value), args)
            append_row(csv_path, row)
            print(
                "[ok] lambda={lambda_value:g} Success@1={s1:.2f}% Success@5={s5:.2f}% "
                "Stability@10={stab:.4f} Avg_Removed_Edges={removed:.2f}".format(
                    lambda_value=float(lambda_value),
                    s1=row["Success@1"],
                    s5=row["Success@5"],
                    stab=row["Stability@10"],
                    removed=row["Avg_Removed_Edges"],
                )
            )
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            append_row(
                csv_path,
                {
                    "lambda": float(lambda_value),
                    "status": "failed",
                    "budget_percent": args.budget_percent,
                    "cosrec_epochs": args.cosrec_epochs,
                    "cosrec_lr": args.cosrec_lr,
                    "beta": args.beta,
                    "objective": args.cosrec_objective,
                    "graph_scope": args.graph_scope,
                    "embedding_readout": args.embedding_readout,
                    "selection_mode": args.selection_mode,
                    "output_dir": str(Path(args.work_dir) / f"lambda_{lambda_label(float(lambda_value))}"),
                    "error": error_text,
                },
            )
            print(f"[failed] lambda={lambda_value:g}: {error_text}", file=sys.stderr)
            traceback.print_exc()
            continue

    plotted = plot_results(
        csv_path,
        plot_path,
        args.plot_metric,
        args.plot_max_lambda if args.plot_max_lambda > 0 else None,
    )
    if plotted:
        print(f"[plot] saved {plotted}")
    print(f"[csv] saved {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
