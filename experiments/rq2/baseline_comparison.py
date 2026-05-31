#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RQ2 all-user counterfactual explanation baseline comparison.

This script compares:
  - Random
  - High-Degree
  - Attention-proxy similarity
  - FIA, a first/second-order influence approximation
  - CoSRec, optimized user-local retention mask

Primary metrics:
  Success@1 = percentage of target users whose original Top-1 recommendation
  is no longer ranked Top-1 after local edge removal.
  Stability@10 = Top-10 set overlap between original and perturbed rankings.
"""

import argparse
import json
import os
import math
import random
import time
import zipfile
from typing import Dict, List, Sequence

import pandas as pd
import torch

torch.set_num_threads(1)

from experiments.rq3.case_study_ml100k import (  # noqa: E402
    lightgcn_forward_native,
    load_edge_csv_from_df,
    load_node_csv_from_df,
    split_by_user,
)
from data.ml_100k import ml_100k_DataLoader  # noqa: E402
from recsys.LightGCN import LightGCN  # noqa: E402
from utils.utils import get_user_positive_items  # noqa: E402


METHODS = ["Random", "High-Degree", "Attention-proxy", "FIA", "CoSRec"]


def load_movie_titles() -> Dict[int, str]:
    with zipfile.ZipFile(os.path.join("data", "ml-100k.zip")) as archive:
        with archive.open("ml-100k/u.item") as item_file:
            items = pd.read_csv(
                item_file,
                sep="|",
                header=None,
                encoding="latin-1",
                usecols=[0, 1],
                names=["item_id", "title"],
            )
    return dict(zip(items["item_id"], items["title"]))


def build_edge_values(train_edge_index: torch.Tensor, user_idx: int, remove_items: Sequence[int]) -> torch.Tensor:
    values = torch.ones(train_edge_index.shape[1])
    if not remove_items:
        return values
    remove_set = {int(item) for item in remove_items}
    mask = (train_edge_index[0] == int(user_idx)) & torch.tensor(
        [int(item) in remove_set for item in train_edge_index[1].tolist()],
        dtype=torch.bool,
    )
    values[mask] = 0.0
    return values


def get_rating(
    model: LightGCN,
    train_edge_index: torch.Tensor,
    edge_values=None,
    include_initial: bool = True,
) -> torch.Tensor:
    users_emb, items_emb = lightgcn_forward_native(
        model,
        train_edge_index,
        model.num_users + model.num_items,
        edge_values,
        include_initial=include_initial,
    )
    return torch.matmul(users_emb, items_emb.T)


def mask_seen(rating: torch.Tensor, train_edge_index: torch.Tensor) -> torch.Tensor:
    rating = rating.clone()
    user_pos_items = get_user_positive_items(train_edge_index)
    exclude_users, exclude_items = [], []
    for user, items in user_pos_items.items():
        exclude_users.extend([user] * len(items))
        exclude_items.extend(items)
    rating[exclude_users, exclude_items] = -torch.inf
    return rating


def topk_for_user(
    model: LightGCN,
    train_edge_index: torch.Tensor,
    user_idx: int,
    k: int,
    edge_values=None,
    seen_items: Sequence[int] = None,
    include_initial: bool = True,
) -> List[int]:
    with torch.no_grad():
        users_emb, items_emb = lightgcn_forward_native(
            model,
            train_edge_index,
            model.num_users + model.num_items,
            edge_values,
            include_initial=include_initial,
        )
        scores = torch.matmul(users_emb[user_idx], items_emb.T)
        if seen_items:
            scores = scores.clone()
            scores[torch.tensor(seen_items, dtype=torch.long)] = -torch.inf
        return torch.topk(scores, k=k).indices.tolist()


def removal_budget(history_size: int, args) -> int:
    if args.budget_percent > 0:
        return max(1, int(math.floor(history_size * args.budget_percent)))
    return int(args.k_edges)


def item_degree(train_edge_index: torch.Tensor, num_items: int) -> torch.Tensor:
    return torch.bincount(train_edge_index[1].long(), minlength=num_items).float()


def random_subset(history: List[int], k: int, rng: random.Random) -> List[int]:
    return rng.sample(history, min(k, len(history)))


def high_degree_subset(history: List[int], k: int, degree: torch.Tensor) -> List[int]:
    hist = torch.tensor(history, dtype=torch.long)
    pos = torch.topk(degree[hist], k=min(k, len(history)), largest=True).indices.tolist()
    return [history[i] for i in pos]


def attention_proxy_subset(model: LightGCN, history: List[int], target_item: int, k: int) -> List[int]:
    hist = torch.tensor(history, dtype=torch.long)
    item_emb = model.items_emb.weight.detach().cpu()
    scores = torch.matmul(item_emb[hist], item_emb[int(target_item)])
    pos = torch.topk(scores, k=min(k, len(history)), largest=True).indices.tolist()
    return [history[i] for i in pos]


def user_edge_positions(train_edge_index: torch.Tensor, user_idx: int) -> torch.Tensor:
    return torch.nonzero(train_edge_index[0] == int(user_idx)).flatten()


def inference_edges_for_user(train_edge_index: torch.Tensor, user_idx: int, graph_scope: str) -> torch.Tensor:
    if graph_scope == "full":
        return train_edge_index
    if graph_scope == "user_local":
        return train_edge_index[:, train_edge_index[0] == int(user_idx)]
    raise ValueError(f"Unknown graph_scope: {graph_scope}")


def fia_subset(
    model: LightGCN,
    train_edge_index: torch.Tensor,
    user_idx: int,
    history: List[int],
    target_item: int,
    k: int,
    hessian_candidates: int,
    include_initial: bool,
) -> List[int]:
    """Fast Influence Analysis using first-order + diagonal Hessian correction.

    Removing edge weight w_j from 1 to 0 changes the target score by:
      s(0) ~= s(1) - g_j + 0.5 * h_jj
    so the estimated target-score decrease is g_j - 0.5*h_jj.
    """
    positions = user_edge_positions(train_edge_index, user_idx)
    values = torch.ones(train_edge_index.shape[1], requires_grad=True)
    users_emb, items_emb = lightgcn_forward_native(
        model,
        train_edge_index,
        model.num_users + model.num_items,
        values,
        include_initial=include_initial,
    )
    target_score = torch.matmul(users_emb[user_idx], items_emb[int(target_item)])
    grad = torch.autograd.grad(target_score, values, create_graph=True)[0]
    user_grad = grad[positions]

    candidate_count = min(max(k, hessian_candidates), len(history))
    candidate_local = torch.topk(user_grad.detach(), k=candidate_count, largest=True).indices

    influence_scores = user_grad.detach().clone()
    for local_pos in candidate_local.tolist():
        edge_pos = positions[local_pos]
        second = torch.autograd.grad(grad[edge_pos], values, retain_graph=True, allow_unused=True)[0]
        h_diag = 0.0 if second is None else float(second[edge_pos].detach().item())
        influence_scores[local_pos] = user_grad[local_pos].detach() - 0.5 * h_diag

    selected_local = torch.topk(influence_scores, k=min(k, len(history)), largest=True).indices.tolist()
    return [history[i] for i in selected_local]


def cosrec_subset(
    model: LightGCN,
    train_edge_index: torch.Tensor,
    user_idx: int,
    history: List[int],
    target_item: int,
    k: int,
    epochs: int,
    lr: float,
    lambda_graph: float,
    beta: float,
    objective: str = "score",
    boundary_item: int = None,
    include_initial: bool = True,
) -> List[int]:
    positions = user_edge_positions(train_edge_index, user_idx)
    p_vec = torch.nn.Parameter(torch.ones(len(history)) * 0.89)
    optimizer = torch.optim.SGD([p_vec], lr=lr)

    for _ in range(epochs):
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
        if objective == "boundary_margin" and boundary_item is not None:
            boundary_score = torch.matmul(users_emb[user_idx], items_emb[int(boundary_item)])
            cf_loss = target_score - boundary_score
        else:
            cf_loss = target_score
        graph_loss = torch.mean((p_vec - 1.0) ** 2)
        l1_loss = torch.linalg.norm(p_vec, ord=1)
        loss = lambda_graph * graph_loss + beta * cf_loss + 0.0002 * l1_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p_vec], 2.0)
        optimizer.step()
        p_vec.data = torch.clamp(p_vec.data, min=0.01, max=1.0)

    selected = torch.topk(p_vec.detach(), k=min(k, len(history)), largest=False).indices.tolist()
    return [history[i] for i in selected]


def evaluate_method_once(
    model: LightGCN,
    train_edge_index: torch.Tensor,
    user_idx: int,
    history: List[int],
    target_item: int,
    method: str,
    k_edges: int,
    degree: torch.Tensor,
    rng: random.Random,
    args,
    boundary_item: int = None,
) -> List[int]:
    if method == "Random":
        return random_subset(history, k_edges, rng)
    if method == "High-Degree":
        return high_degree_subset(history, k_edges, degree)
    if method == "Attention-proxy":
        return attention_proxy_subset(model, history, target_item, k_edges)
    if method == "FIA":
        return fia_subset(
            model,
            train_edge_index,
            user_idx,
            history,
            target_item,
            k_edges,
            args.hessian_candidates,
            args.embedding_readout == "all_layers",
        )
    if method == "CoSRec":
        return cosrec_subset(
            model,
            train_edge_index,
            user_idx,
            history,
            target_item,
            k_edges,
            args.cosrec_epochs,
            args.cosrec_lr,
            args.lambda_graph,
            args.beta,
            args.cosrec_objective,
            boundary_item,
            args.embedding_readout == "all_layers",
        )
    raise ValueError(f"Unknown method: {method}")


def markdown_table(rows: List[Dict[str, float]]) -> str:
    lines = [
        "| Method | Sparsity Budget | Avg. Removed Edges | Users | Success@1 | Success@5 | Stability@10 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['sparsity_budget_label']} | {row['avg_removed_edges']:.2f} | "
            f"{int(row['num_users'])} | {row['success_at_1']:.2f}% | {row['success_at_5']:.2f}% | "
            f"{row['stability_at_10']:.4f} |"
        )
    return "\n".join(lines)


def output_prefix(args) -> str:
    if args.budget_percent > 0:
        return f"rq2_budget{int(round(args.budget_percent * 100))}"
    return f"rq2_k{int(args.k_edges)}"


def main():
    parser = argparse.ArgumentParser(description="RQ2 all-user counterfactual baseline comparison.")
    parser.add_argument("--k_edges", type=int, default=3)
    parser.add_argument(
        "--budget_percent",
        type=float,
        default=0.0,
        help="If > 0, remove max(1, floor(|H_u| * budget_percent)) edges per user; overrides fixed --k_edges.",
    )
    parser.add_argument("--max_users", type=int, default=30)
    parser.add_argument(
        "--user_ids",
        type=str,
        default="",
        help="Optional comma-separated original MovieLens user ids. Overrides max_users.",
    )
    parser.add_argument(
        "--target_pairs",
        type=str,
        default="",
        help="Optional comma-separated original_user_id:rank pairs, using 1-based ranks. Overrides user_ids.",
    )
    parser.add_argument(
        "--target_pairs_file",
        type=str,
        default="",
        help="Optional JSON file with [{'user_id': original_id, 'rank': one_based_rank}, ...].",
    )
    parser.add_argument("--min_history", type=int, default=3)
    parser.add_argument("--target_rank", type=int, default=0, help="0-based original recommendation rank to explain.")
    parser.add_argument(
        "--success_metric",
        choices=["top1_changed", "target_rank_changed", "target_not_in_top10"],
        default="top1_changed",
        help="Legacy detailed-record success flag. Main table always reports Success@1/5 and Stability@10.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--methods", type=str, default=",".join(METHODS), help="Comma-separated method names to run.")
    parser.add_argument("--cosrec_epochs", type=int, default=8)
    parser.add_argument("--cosrec_lr", type=float, default=0.1)
    parser.add_argument("--lambda_graph", type=float, default=10.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--cosrec_objective", choices=["score", "boundary_margin"], default="score")
    parser.add_argument("--hessian_candidates", type=int, default=8)
    parser.add_argument("--model_path", default=os.path.join("model", "light_GCN.pt"))
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument(
        "--graph_scope",
        choices=["full", "user_local"],
        default="full",
        help="Use the full training graph or the target user's local history subgraph for counterfactual inference.",
    )
    parser.add_argument(
        "--embedding_readout",
        choices=["all_layers", "propagation_only"],
        default="all_layers",
        help="LightGCN readout. all_layers is standard LightGCN; propagation_only isolates message-passing layers for edge-removal diagnostics.",
    )
    parser.add_argument("--output_dir", default=os.path.join("results", "paper", "rq2"))
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_loader = ml_100k_DataLoader(dataset_name="ml-100k")
    origin_data = data_loader.load_data()
    user_mapping = load_node_csv_from_df(origin_data, "user_id")
    movie_mapping = load_node_csv_from_df(origin_data, "item_id")
    user_reverse_mapping = {v: k for k, v in user_mapping.items()}
    movie_reverse_mapping = {v: k for k, v in movie_mapping.items()}
    movie_titles = load_movie_titles()

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
    target_pairs = []
    if args.target_pairs_file.strip():
        with open(args.target_pairs_file, "r", encoding="utf-8") as f:
            pair_records = json.load(f)
        for record in pair_records:
            original_user_id = int(record["user_id"])
            rank0 = int(record["rank"]) - 1
            if original_user_id in user_mapping and 0 <= rank0 < 10:
                user_idx = user_mapping[original_user_id]
                if len(user_history[user_idx]) >= max(args.min_history, removal_budget(len(user_history[user_idx]), args)):
                    target_pairs.append((user_idx, rank0))
        candidate_users = sorted({user_idx for user_idx, _ in target_pairs})
    elif args.target_pairs.strip():
        for spec in args.target_pairs.split(","):
            user_text, rank_text = spec.strip().split(":")
            original_user_id = int(user_text)
            rank0 = int(rank_text) - 1
            if original_user_id in user_mapping and 0 <= rank0 < 10:
                user_idx = user_mapping[original_user_id]
                if len(user_history[user_idx]) >= max(args.min_history, removal_budget(len(user_history[user_idx]), args)):
                    target_pairs.append((user_idx, rank0))
        candidate_users = sorted({user_idx for user_idx, _ in target_pairs})
    elif args.user_ids.strip():
        original_user_ids = [int(uid.strip()) for uid in args.user_ids.split(",") if uid.strip()]
        candidate_users = [user_mapping[uid] for uid in original_user_ids if uid in user_mapping]
        candidate_users = [
            user_idx
            for user_idx in candidate_users
            if len(user_history[user_idx]) >= max(args.min_history, removal_budget(len(user_history[user_idx]), args))
        ]
        target_pairs = [(user_idx, args.target_rank) for user_idx in candidate_users]
    else:
        candidate_users = [
            user_idx
            for user_idx, history in user_history.items()
            if len(history) >= max(args.min_history, removal_budget(len(history), args))
        ][: args.max_users]
        target_pairs = [(user_idx, args.target_rank) for user_idx in candidate_users]

    degree = item_degree(train_edge_index, model.num_items)
    rng = random.Random(args.seed)

    print(
        f"Evaluating {len(target_pairs)} target instances from {len(candidate_users)} users, "
        f"budget={'{:.0%}'.format(args.budget_percent) if args.budget_percent > 0 else 'fixed k=' + str(args.k_edges)}, "
        f"target_rank={args.target_rank + 1}, metric={args.success_metric}, "
        f"graph_scope={args.graph_scope}, readout={args.embedding_readout}"
    )
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

    detailed = []
    summary = []
    start_time = time.time()
    methods = [name.strip() for name in args.methods.split(",") if name.strip()]
    for method in methods:
        print(f"Running {method}...")
        successes = []
        success_at_1_values = []
        success_at_5_values = []
        stabilities = []
        removed_counts = []
        for n, (user_idx, target_rank0) in enumerate(target_pairs, start=1):
            history = user_history[user_idx]
            eval_edge_index = inference_edges_for_user(train_edge_index, user_idx, args.graph_scope)
            target_item = original_top10[user_idx][target_rank0]
            boundary_item = original_top10[user_idx][min(target_rank0 + 1, 9)]
            k_edges = removal_budget(len(history), args)
            remove_items = evaluate_method_once(
                model,
                eval_edge_index,
                user_idx,
                history,
                target_item,
                method,
                k_edges,
                degree,
                rng,
                args,
                boundary_item,
            )
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
            target_drop = int(target_item not in perturbed_top10)
            rank_change = int(perturbed_top10[target_rank0] != target_item)
            original_top1 = original_top10[user_idx][0]
            success_at_1 = int(perturbed_top10[0] != original_top1)
            success_at_5 = int(original_top1 not in perturbed_top10[:5])
            if args.success_metric == "top1_changed":
                success = success_at_1
            elif args.success_metric == "target_rank_changed":
                success = rank_change
            else:
                success = target_drop
            stability = len(set(original_top10[user_idx]).intersection(perturbed_top10)) / 10.0
            successes.append(success)
            success_at_1_values.append(success_at_1)
            success_at_5_values.append(success_at_5)
            stabilities.append(stability)
            removed_counts.append(len(remove_items))

            detailed.append(
                {
                    "method": method,
                    "mapped_user_id": int(user_idx),
                    "user_id": int(user_reverse_mapping[user_idx]),
                    "target_item_id": int(movie_reverse_mapping[target_item]),
                    "target_title": movie_titles.get(int(movie_reverse_mapping[target_item]), str(target_item)),
                    "target_rank": int(target_rank0 + 1),
                    "original_top1_item_id": int(movie_reverse_mapping[original_top1]),
                    "original_top1_title": movie_titles.get(int(movie_reverse_mapping[original_top1]), str(original_top1)),
                    "removed_item_ids": [int(movie_reverse_mapping[item]) for item in remove_items],
                    "removed_titles": [
                        movie_titles.get(int(movie_reverse_mapping[item]), str(item)) for item in remove_items
                    ],
                    "original_top10_item_ids": [int(movie_reverse_mapping[item]) for item in original_top10[user_idx]],
                    "perturbed_top10_item_ids": [int(movie_reverse_mapping[item]) for item in perturbed_top10],
                    "success": success,
                    "success_metric": args.success_metric,
                    "success_at_1": success_at_1,
                    "success_at_5": success_at_5,
                    "target_drop_at_10": target_drop,
                    "rank_change_at_10": rank_change,
                    "stability_at_10": stability,
                    "removed_edges": len(remove_items),
                }
            )
            if n % 10 == 0:
                print(f"  {method}: {n}/{len(target_pairs)} target instances")

        summary.append(
            {
                "method": method,
                "k_edges": args.k_edges,
                "budget_percent": args.budget_percent,
                "graph_scope": args.graph_scope,
                "embedding_readout": args.embedding_readout,
                "sparsity_budget_label": f"{args.budget_percent * 100:.0f}%" if args.budget_percent > 0 else f"k={args.k_edges}",
                "avg_removed_edges": sum(removed_counts) / len(removed_counts),
                "num_users": len(target_pairs),
                "cf_percentage": 100.0 * sum(successes) / len(successes),
                "success_at_1": 100.0 * sum(success_at_1_values) / len(success_at_1_values),
                "success_at_5": 100.0 * sum(success_at_5_values) / len(success_at_5_values),
                "target_drop_at_10": 100.0
                * sum(int(row["target_drop_at_10"]) for row in detailed if row["method"] == method)
                / len(successes),
                "rank_change_at_10": 100.0
                * sum(int(row["rank_change_at_10"]) for row in detailed if row["method"] == method)
                / len(successes),
                "stability_at_10": sum(stabilities) / len(stabilities),
            }
        )

    os.makedirs(args.output_dir, exist_ok=True)
    prefix = output_prefix(args)
    md = markdown_table(summary)
    with open(os.path.join(args.output_dir, f"{prefix}_table.md"), "w", encoding="utf-8") as f:
        f.write(md + "\n")
    with open(os.path.join(args.output_dir, f"{prefix}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.output_dir, f"{prefix}_detailed.json"), "w", encoding="utf-8") as f:
        json.dump(detailed, f, ensure_ascii=False, indent=2)

    print(md)
    print(f"Elapsed: {(time.time() - start_time) / 60:.2f} min")
    print(f"Saved: {os.path.join(args.output_dir, f'{prefix}_table.md')}")


if __name__ == "__main__":
    main()
