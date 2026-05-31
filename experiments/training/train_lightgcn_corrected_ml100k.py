#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train a corrected LightGCN checkpoint for MovieLens-100K.

The original repository builds the bipartite graph with raw item indices in
the same node id space as users. This script trains with the corrected joint
node space used by `case_study_ml100k.lightgcn_forward_native`.
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from experiments.rq3.case_study_ml100k import (
    lightgcn_forward_native,
    load_edge_csv_from_df,
    load_node_csv_from_df,
    split_by_user,
)
from data.ml_100k import ml_100k_DataLoader
from recsys.LightGCN import LightGCN


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_positive_mask(edge_index: torch.Tensor, num_users: int, num_items: int) -> torch.Tensor:
    mask = torch.zeros((num_users, num_items), dtype=torch.bool)
    mask[edge_index[0].long(), edge_index[1].long()] = True
    return mask


def sample_negative_items_for_users(
    user_idx: torch.Tensor,
    positive_mask: torch.Tensor,
    num_items: int,
) -> torch.Tensor:
    neg_items = torch.randint(0, num_items, (user_idx.numel(),), device=user_idx.device)
    user_idx_cpu = user_idx.cpu()
    invalid = positive_mask[user_idx_cpu, neg_items.cpu()].to(user_idx.device)
    while bool(invalid.any()):
        neg_items[invalid] = torch.randint(0, num_items, (int(invalid.sum().item()),), device=user_idx.device)
        invalid = positive_mask[user_idx_cpu, neg_items.cpu()].to(user_idx.device)
    return neg_items


def bpr_loss(
    users_emb: torch.Tensor,
    items_emb: torch.Tensor,
    user_idx: torch.Tensor,
    pos_item_idx: torch.Tensor,
    neg_item_idx: torch.Tensor,
) -> torch.Tensor:
    pos_scores = (users_emb[user_idx] * items_emb[pos_item_idx]).sum(dim=1)
    neg_scores = (users_emb[user_idx] * items_emb[neg_item_idx]).sum(dim=1)
    return -F.logsigmoid(pos_scores - neg_scores).mean()


def evaluate_model(
    model: LightGCN,
    train_edge_index: torch.Tensor,
    test_edge_index: torch.Tensor,
    include_initial: bool,
    k: int = 20,
) -> str:
    with torch.no_grad():
        users_emb, items_emb = lightgcn_forward_native(
            model,
            train_edge_index,
            model.num_users + model.num_items,
            include_initial=include_initial,
        )
        scores = torch.matmul(users_emb, items_emb.T)
        scores = scores.clone()
        scores[train_edge_index[0], train_edge_index[1]] = -torch.inf
        topk = torch.topk(scores, k=k, dim=1).indices
        top1 = topk[:, 0]
        unique_top1 = int(top1.unique().numel())
        most_common_item = int(torch.mode(top1).values.item())
        most_common_count = int((top1 == most_common_item).sum().item())

        test_items_by_user = {}
        for user, item in zip(test_edge_index[0].tolist(), test_edge_index[1].tolist()):
            test_items_by_user.setdefault(int(user), set()).add(int(item))

        recalls = []
        ndcgs = []
        for user, positives in test_items_by_user.items():
            ranked = topk[user].tolist()
            hits = [1 if item in positives else 0 for item in ranked]
            recalls.append(sum(hits) / len(positives))
            dcg = sum(hit / np.log2(rank + 2) for rank, hit in enumerate(hits))
            ideal_hits = min(len(positives), k)
            idcg = sum(1.0 / np.log2(rank + 2) for rank in range(ideal_hits))
            ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    recall = float(np.mean(recalls)) if recalls else 0.0
    ndcg = float(np.mean(ndcgs)) if ndcgs else 0.0
    return (
        f"recall@{k}={recall:.4f} ndcg@{k}={ndcg:.4f} "
        f"unique_top1={unique_top1} most_common_top1_count={most_common_count}/{model.num_users}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train corrected LightGCN on ML-100K.")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--negatives_per_positive", type=int, default=1)
    parser.add_argument("--eval_every", type=int, default=25)
    parser.add_argument(
        "--embedding_readout",
        choices=["all_layers", "propagation_only"],
        default="all_layers",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rating_threshold", type=int, default=3)
    parser.add_argument("--output", default=os.path.join("model", "light_GCN_corrected.pt"))
    args = parser.parse_args()

    torch.set_num_threads(1)
    set_seed(args.seed)

    origin_data = ml_100k_DataLoader(dataset_name="ml-100k").load_data()
    user_mapping = load_node_csv_from_df(origin_data, "user_id")
    movie_mapping = load_node_csv_from_df(origin_data, "item_id")
    edge_index = load_edge_csv_from_df(
        origin_data,
        "user_id",
        user_mapping,
        "item_id",
        movie_mapping,
        "score",
        rating_threshold=args.rating_threshold,
    )
    train_edge_index, test_edge_index = split_by_user(edge_index, len(user_mapping), args.seed)

    model = LightGCN(len(user_mapping), len(movie_mapping), K=args.layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    user_idx = train_edge_index[0].long()
    pos_item_idx = train_edge_index[1].long()
    positive_mask = build_positive_mask(edge_index, len(user_mapping), len(movie_mapping))
    include_initial = args.embedding_readout == "all_layers"

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        users_emb, items_emb = lightgcn_forward_native(
            model,
            train_edge_index,
            model.num_users + model.num_items,
            include_initial=include_initial,
        )
        train_users = user_idx.repeat_interleave(args.negatives_per_positive)
        train_pos_items = pos_item_idx.repeat_interleave(args.negatives_per_positive)
        neg_item_idx = sample_negative_items_for_users(train_users, positive_mask, model.num_items)
        loss = bpr_loss(users_emb, items_emb, train_users, train_pos_items, neg_item_idx)
        reg_loss = (
            model.users_emb(train_users).pow(2).sum(dim=1).mean()
            + model.items_emb(train_pos_items).pow(2).sum(dim=1).mean()
            + model.items_emb(neg_item_idx).pow(2).sum(dim=1).mean()
        )
        total_loss = loss + args.weight_decay * reg_loss
        total_loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()
            print(
                f"epoch={epoch:04d} loss={loss.item():.4f} reg={reg_loss.item():.4f} "
                f"{evaluate_model(model, train_edge_index, test_edge_index, include_initial)}"
            )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"Saved corrected checkpoint: {args.output}")


if __name__ == "__main__":
    main()
