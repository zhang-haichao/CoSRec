#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate MovieLens-100K user-level CoSRec case studies for RQ3."""

import argparse
import json
import os
import random
import zipfile

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.nn.utils import clip_grad_norm_

from data.ml_100k import ml_100k_DataLoader
from recsys.LightGCN import LightGCN
from utils.utils import get_user_positive_items


def load_node_csv_from_df(df, column_name):
    return {index: i for i, index in enumerate(df[column_name].unique())}


def load_edge_csv_from_df(
    df,
    src_column_name,
    src_mapping,
    dst_column_name,
    dst_mapping,
    link_column_name,
    rating_threshold=3,
):
    positive_interactions = df[df[link_column_name] >= rating_threshold]
    src = [src_mapping[index] for index in positive_interactions[src_column_name]]
    dst = [dst_mapping[index] for index in positive_interactions[dst_column_name]]
    return torch.tensor([src, dst], dtype=torch.long)


def load_movie_titles():
    item_path = os.path.join("data", "ml-100k", "u.item")
    if os.path.exists(item_path):
        item_source = item_path
    else:
        zip_path = os.path.join("data", "ml-100k.zip")
        with zipfile.ZipFile(zip_path) as archive:
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

    items = pd.read_csv(
        item_source,
        sep="|",
        header=None,
        encoding="latin-1",
        usecols=[0, 1],
        names=["item_id", "title"],
    )
    return dict(zip(items["item_id"], items["title"]))


def split_by_user(edge_index, num_users, seed):
    train_parts = []
    test_parts = []
    for user_idx in range(num_users):
        edge_positions = torch.nonzero(edge_index[0] == user_idx).flatten().tolist()
        if len(edge_positions) < 2:
            train_parts.append(edge_index[:, edge_positions])
            continue
        train_indices, test_indices = train_test_split(
            edge_positions, test_size=0.2, random_state=seed
        )
        train_parts.append(edge_index[:, train_indices])
        test_parts.append(edge_index[:, test_indices])

    train_edge_index = torch.cat(train_parts, dim=1)
    test_edge_index = torch.cat(test_parts, dim=1) if test_parts else torch.empty((2, 0), dtype=torch.long)
    return train_edge_index, test_edge_index


def lightgcn_forward_native(model, edge_index, num_nodes, values=None, include_initial=True):
    """Native PyTorch equivalent of the repository LightGCN propagation path."""
    row, col = edge_index
    if values is None:
        values = torch.ones(row.numel(), dtype=model.users_emb.weight.dtype)

    # LightGCN uses one joint node space: users occupy [0, num_users),
    # items occupy [num_users, num_users + num_items). Raw MovieLens item
    # ids are local item indices and must be shifted before message passing.
    item_nodes = col + model.num_users
    row_bi = torch.cat([row, item_nodes], dim=0)
    col_bi = torch.cat([item_nodes, row], dim=0)
    values_bi = torch.cat([values, values], dim=0)

    deg = torch.zeros(num_nodes, dtype=values_bi.dtype)
    deg.index_add_(0, row_bi, values_bi)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    norm_values = deg_inv_sqrt[row_bi] * values_bi * deg_inv_sqrt[col_bi]

    adj = torch.sparse_coo_tensor(
        torch.stack([row_bi, col_bi], dim=0),
        norm_values,
        (num_nodes, num_nodes),
    ).coalesce()

    emb_0 = torch.cat([model.users_emb.weight, model.items_emb.weight])
    embs = [emb_0] if include_initial else []
    emb_k = emb_0
    for _ in range(model.K):
        emb_k = torch.sparse.mm(adj, emb_k)
        embs.append(emb_k)

    emb_final = torch.mean(torch.stack(embs, dim=1), dim=1)
    return torch.split(emb_final, [model.num_users, model.num_items])


def top_recommendations(model, edge_index, values, k, user_idx, exclude_edge_indices):
    with torch.no_grad():
        num_nodes = model.num_users + model.num_items
        users_emb, items_emb = lightgcn_forward_native(model, edge_index, num_nodes, values)
        rating = torch.matmul(users_emb, items_emb.T)

    for exclude_edge_index in exclude_edge_indices:
        user_pos_items = get_user_positive_items(exclude_edge_index)
        exclude_users = []
        exclude_items = []
        for user, items in user_pos_items.items():
            exclude_users.extend([user] * len(items))
            exclude_items.extend(items)
        if exclude_users:
            rating[exclude_users, exclude_items] = -(1 << 10)

    scores, items = torch.topk(rating[user_idx], k=k)
    return items.tolist(), scores.tolist()


def optimize_counterfactual(
    model,
    train_edge_index,
    num_users,
    num_items,
    user_idx,
    target_item,
    boundary_item,
    user_hist_count,
    epochs,
    lr,
    optimizer_name,
    lambda_val,
    beta,
    seed,
    objective,
):
    torch.manual_seed(seed)
    p_vec = torch.nn.Parameter(torch.ones(user_hist_count) * 0.89)
    user_edge_positions = torch.nonzero(train_edge_index[0] == user_idx).flatten()

    if optimizer_name == "Adam":
        optimizer = optim.Adam([p_vec], lr=lr)
    elif optimizer_name == "Adadelta":
        optimizer = optim.Adadelta([p_vec], lr=lr)
    else:
        optimizer = optim.SGD([p_vec], lr=lr)

    for _ in range(epochs):
        optimizer.zero_grad()
        edge_values = torch.ones(train_edge_index.shape[1])
        edge_values[user_edge_positions] = p_vec
        users_emb, items_emb = lightgcn_forward_native(
            model, train_edge_index, num_users + num_items, edge_values
        )
        rating = torch.matmul(users_emb, items_emb.T)

        graph_dist = torch.mean((p_vec - torch.ones_like(p_vec)) ** 2)
        l1_loss = torch.linalg.norm(p_vec, ord=1)
        target_score = rating[user_idx, target_item]
        if objective == "boundary_margin" and boundary_item is not None:
            boundary_score = rating[user_idx, boundary_item]
            cf_term = target_score - boundary_score
        else:
            cf_term = target_score
        loss = lambda_val * graph_dist + beta * cf_term + 0.0002 * l1_loss

        loss.backward()
        clip_grad_norm_([p_vec], 2.0)
        optimizer.step()
        p_vec.data = torch.clamp(p_vec.data, min=0.01, max=1.0)

    return p_vec.detach().clone()


def build_perturbed_values(train_edge_index, user_idx, support_indices):
    values = torch.ones(train_edge_index.shape[1])
    user_edge_positions = torch.nonzero(train_edge_index[0] == user_idx).flatten()
    if len(support_indices) > 0:
        values[user_edge_positions[torch.tensor(support_indices, dtype=torch.long)]] = 0.0
    return values


def title_for(mapped_item, movie_reverse_mapping, movie_titles):
    original_id = int(movie_reverse_mapping[mapped_item])
    return original_id, movie_titles.get(original_id, f"Item {original_id}")


def make_case_for_user(
    args,
    model,
    train_edge_index,
    num_users,
    num_items,
    user_idx,
    user_reverse_mapping,
    movie_reverse_mapping,
    movie_titles,
):
    user_edges = train_edge_index[:, train_edge_index[0] == user_idx]
    user_hist_count = user_edges.shape[1]
    if user_hist_count < args.min_history:
        return None

    original_items, original_scores = top_recommendations(
        model, train_edge_index, None, args.top_k, user_idx, [train_edge_index]
    )
    if args.target_index >= len(original_items):
        return None

    target_item = original_items[args.target_index]
    boundary_item = original_items[min(args.target_index + 1, len(original_items) - 1)]
    p_vec = optimize_counterfactual(
        model=model,
        train_edge_index=train_edge_index,
        num_users=num_users,
        num_items=num_items,
        user_idx=user_idx,
        target_item=target_item,
        boundary_item=boundary_item,
        user_hist_count=user_hist_count,
        epochs=args.epochs,
        lr=args.lr,
        optimizer_name=args.optimizer,
        lambda_val=args.lambda_val,
        beta=args.beta,
        seed=args.seed,
        objective=args.objective,
    )

    support_count = max(1, int(args.rate * user_hist_count))
    support_count = min(args.max_support, support_count, user_hist_count)
    _, support_tensor = torch.sort(p_vec)
    support_indices = support_tensor[:support_count].tolist()

    perturbed_values = build_perturbed_values(train_edge_index, user_idx, support_indices)
    perturbed_items, perturbed_scores = top_recommendations(
        model, train_edge_index, perturbed_values, args.top_k, user_idx, [train_edge_index]
    )

    user_history_items = train_edge_index[1, train_edge_index[0] == user_idx].tolist()
    support_items = []
    for rank, edge_position in enumerate(support_indices, start=1):
        mapped_item = int(user_history_items[edge_position])
        original_id, title = title_for(mapped_item, movie_reverse_mapping, movie_titles)
        support_items.append(
            {
                "rank": rank,
                "item_id": original_id,
                "title": title,
                "retention": round(float(p_vec[edge_position].item()), 4),
                "contribution": round(float(1.0 - p_vec[edge_position].item()), 4),
            }
        )

    original_id, original_title = title_for(target_item, movie_reverse_mapping, movie_titles)
    replacement_item = perturbed_items[args.target_index]
    replacement_id, replacement_title = title_for(replacement_item, movie_reverse_mapping, movie_titles)

    return {
        "user_id": int(user_reverse_mapping[user_idx]),
        "mapped_user_id": int(user_idx),
        "history_count": int(user_hist_count),
        "rate": args.rate,
        "target_rank": int(args.target_index + 1),
        "target_item_mapped_id": int(target_item),
        "original_recommendation": {
            "item_id": original_id,
            "title": original_title,
            "score": round(float(original_scores[args.target_index]), 6),
        },
        "p_vector": [round(float(v), 6) for v in p_vec.tolist()],
        "counterfactual_support_set": support_items,
        "replacement_recommendation": {
            "item_id": replacement_id,
            "title": replacement_title,
            "score": round(float(perturbed_scores[args.target_index]), 6),
        },
        "original_top_k": [
            {
                "rank": rank,
                "item_id": title_for(item, movie_reverse_mapping, movie_titles)[0],
                "title": title_for(item, movie_reverse_mapping, movie_titles)[1],
            }
            for rank, item in enumerate(original_items, start=1)
        ],
        "perturbed_top_k": [
            {
                "rank": rank,
                "item_id": title_for(item, movie_reverse_mapping, movie_titles)[0],
                "title": title_for(item, movie_reverse_mapping, movie_titles)[1],
            }
            for rank, item in enumerate(perturbed_items, start=1)
        ],
        "changed": original_items[args.target_index] != perturbed_items[args.target_index],
    }


def format_markdown_table(case):
    support_titles = "<br>".join(
        f"{item['rank']}. {item['title']} (w={item['contribution']:.4f})"
        for item in case["counterfactual_support_set"]
    )
    return "\n".join(
        [
            "| User | Original recommendation | Counterfactual support set | Recommendation after removal |",
            "| --- | --- | --- | --- |",
            (
                f"| {case['user_id']} ({case['history_count']} train positives; rank {case['target_rank']}) "
                f"| {case['original_recommendation']['title']} "
                f"| {support_titles} "
                f"| {case['replacement_recommendation']['title']} |"
            ),
        ]
    )


def main():
    parser = argparse.ArgumentParser(description="Generate an ML-100K CoSRec user-level case-study table.")
    parser.add_argument("--user_id", type=int, help="Original MovieLens user id. If omitted, scan long-tail users.")
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--target_index", type=int, default=0, help="0-based recommendation rank to explain.")
    parser.add_argument("--rate", type=float, default=0.2, help="Fraction of user history removed after mask learning.")
    parser.add_argument("--max_support", type=int, default=3, help="Maximum support-set size shown in the table.")
    parser.add_argument("--top_k", type=int, default=10, help="Recommendation list length.")
    parser.add_argument("--epochs", type=int, default=50, help="Mask optimization epochs.")
    parser.add_argument("--lr", type=float, default=0.08, help="Mask optimizer learning rate.")
    parser.add_argument("--optimizer", choices=["SGD", "Adam", "Adadelta"], default="SGD")
    parser.add_argument("--objective", choices=["score", "boundary_margin"], default="score")
    parser.add_argument("--lambda_val", type=float, default=10.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--min_history", type=int, default=5)
    parser.add_argument("--max_history", type=int, default=20)
    parser.add_argument("--scan_limit", type=int, default=80)
    parser.add_argument("--model_path", default=os.path.join("model", "light_GCN.pt"))
    parser.add_argument("--output", default=os.path.join("results", "paper", "rq3", "case_study_ml100k.json"))
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_loader = ml_100k_DataLoader(dataset_name="ml-100k")
    origin_data = data_loader.load_data()
    movie_titles = load_movie_titles()

    user_mapping = load_node_csv_from_df(origin_data, "user_id")
    movie_mapping = load_node_csv_from_df(origin_data, "item_id")
    user_reverse_mapping = {v: k for k, v in user_mapping.items()}
    movie_reverse_mapping = {v: k for k, v in movie_mapping.items()}

    edge_index = load_edge_csv_from_df(
        origin_data,
        "user_id",
        user_mapping,
        "item_id",
        movie_mapping,
        "score",
        rating_threshold=3,
    )
    num_users, num_items = len(user_mapping), len(movie_mapping)
    train_edge_index, _ = split_by_user(edge_index, num_users, args.seed)

    model = LightGCN(num_users, num_items, K=args.layers)
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model.eval()

    if args.user_id is not None:
        if args.user_id not in user_mapping:
            raise ValueError(f"User {args.user_id} is not in MovieLens 100K.")
        candidate_users = [user_mapping[args.user_id]]
    else:
        history_counts = []
        for user_idx in range(num_users):
            count = int((train_edge_index[0] == user_idx).sum().item())
            if args.min_history <= count <= args.max_history:
                history_counts.append((count, user_idx))
        candidate_users = [user_idx for _, user_idx in sorted(history_counts)[: args.scan_limit]]

    selected_case = None
    fallback_case = None
    for user_idx in candidate_users:
        case = make_case_for_user(
            args,
            model,
            train_edge_index,
            num_users,
            num_items,
            user_idx,
            user_reverse_mapping,
            movie_reverse_mapping,
            movie_titles,
        )
        if case is None:
            continue
        fallback_case = fallback_case or case
        if case["changed"]:
            selected_case = case
            break

    selected_case = selected_case or fallback_case
    if selected_case is None:
        raise RuntimeError("No eligible user produced a case study.")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(selected_case, f, ensure_ascii=False, indent=2)

    markdown_table = format_markdown_table(selected_case)
    markdown_output = os.path.splitext(args.output)[0] + ".md"
    with open(markdown_output, "w", encoding="utf-8") as f:
        f.write(markdown_table + "\n")

    print(markdown_table)
    print(f"\nSaved JSON: {args.output}")
    print(f"Saved Markdown: {markdown_output}")


if __name__ == "__main__":
    main()
