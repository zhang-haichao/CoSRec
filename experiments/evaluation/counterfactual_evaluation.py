#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Counterfactual explanation evaluation for LightGCN recommenders.

This module compares CoSRec against three local edge-removal heuristics:
Random, item Popularity/Degree, and target-item Similarity.

Expected external inputs:
    model: trained LightGCN model.
    train_edge_index: LongTensor [2, num_edges], rows are users and cols are items.
    user_history_dict: dict[user_id, list[item_id]].
    learned_mask_dict: dict[user_id, Tensor], where mask order matches user_history_dict[user_id].

The implementation masks only the selected target-user history edges during
inference. It does not retrain the recommender.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

try:
    from torch_sparse import SparseTensor
except Exception:  # pragma: no cover - optional dependency guard
    SparseTensor = None


StrategyName = str
RatingFn = Callable[[torch.nn.Module, torch.Tensor, Optional[torch.Tensor], "GraphMaskCache"], torch.Tensor]


STRATEGIES: Tuple[StrategyName, ...] = ("random", "popularity", "similarity", "cosrec")


@dataclass
class GraphMaskCache:
    """Cached graph structure for efficient local edge masking.

    The expensive part of graph construction, the COO row/col index tensors and
    user-item edge lookup, is built once. Per counterfactual trial we only clone
    the base edge-weight vector and set the selected target-user edges to zero.
    """

    train_edge_index: torch.Tensor
    num_users: int
    num_items: int
    device: torch.device

    def __post_init__(self) -> None:
        self.train_edge_index = self.train_edge_index.to(self.device).long()
        self.row = self.train_edge_index[0]
        self.col = self.train_edge_index[1]
        self.base_edge_weight = torch.ones(self.train_edge_index.size(1), device=self.device)
        self.sparse_sizes = (self.num_users + self.num_items, self.num_users + self.num_items)

        self._edge_pos: Dict[Tuple[int, int], List[int]] = {}
        self._user_edge_pos: Dict[int, List[int]] = {}
        for pos, (user, item) in enumerate(zip(self.row.tolist(), self.col.tolist())):
            self._edge_pos.setdefault((user, item), []).append(pos)
            self._user_edge_pos.setdefault(user, []).append(pos)

    def edge_weights_with_removed(self, user_id: int, remove_items: Sequence[int]) -> torch.Tensor:
        """Return edge weights where only selected target-user edges are zeroed."""
        edge_weight = self.base_edge_weight.clone()
        positions: List[int] = []
        for item_id in remove_items:
            positions.extend(self._edge_pos.get((int(user_id), int(item_id)), []))
        if positions:
            edge_weight[torch.tensor(positions, device=self.device, dtype=torch.long)] = 0.0
        return edge_weight

    def to_sparse_tensor(self, edge_weight: Optional[torch.Tensor] = None) -> "SparseTensor":
        """Create a SparseTensor using cached row/col and the supplied weights."""
        if SparseTensor is None:
            raise ImportError("torch_sparse is required for SparseTensor-based LightGCN inference.")
        value = self.base_edge_weight if edge_weight is None else edge_weight.to(self.device)
        return SparseTensor(row=self.row, col=self.col, value=value, sparse_sizes=self.sparse_sizes)


def infer_num_users_items(model: torch.nn.Module, train_edge_index: torch.Tensor) -> Tuple[int, int]:
    num_users = int(getattr(model, "num_users", int(train_edge_index[0].max().item()) + 1))
    num_items = int(getattr(model, "num_items", int(train_edge_index[1].max().item()) + 1))
    return num_users, num_items


def item_degree(train_edge_index: torch.Tensor, num_items: Optional[int] = None) -> torch.Tensor:
    if num_items is None:
        num_items = int(train_edge_index[1].max().item()) + 1
    return torch.bincount(train_edge_index[1].long().cpu(), minlength=num_items).float()


def get_item_embedding_matrix(model: torch.nn.Module) -> torch.Tensor:
    """Return the item embedding table used by the similarity baseline."""
    if hasattr(model, "items_emb"):
        return model.items_emb.weight.detach()
    if hasattr(model, "item_embedding"):
        return model.item_embedding.weight.detach()
    if hasattr(model, "embedding_item"):
        return model.embedding_item.weight.detach()
    raise AttributeError(
        "Cannot find item embeddings on model. Expected one of: items_emb, "
        "item_embedding, embedding_item. Pass a custom similarity implementation "
        "or expose the trained item embeddings."
    )


def _budget_to_k(history_size: int, budget: float) -> int:
    if history_size <= 0:
        return 0
    return max(1, int(math.floor(history_size * budget)))


def get_removal_subset(
    strategy: StrategyName,
    user_id: int,
    history_items: Sequence[int],
    target_item: int,
    budget: float,
    *,
    train_edge_index: torch.Tensor,
    learned_mask_dict: Optional[Mapping[int, torch.Tensor]] = None,
    degree: Optional[torch.Tensor] = None,
    item_embeddings: Optional[torch.Tensor] = None,
    similarity: str = "dot",
    rng: Optional[random.Random] = None,
) -> List[int]:
    """Select the size-k historical item subset to remove for one strategy."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}. Expected one of {STRATEGIES}.")

    history = [int(item) for item in history_items]
    k = _budget_to_k(len(history), budget)
    if k == 0:
        return []
    k = min(k, len(history))

    if strategy == "random":
        local_rng = rng or random.Random()
        return local_rng.sample(history, k)

    history_tensor = torch.tensor(history, dtype=torch.long)

    if strategy == "popularity":
        if degree is None:
            degree = item_degree(train_edge_index)
        scores = degree[history_tensor.cpu()]
        top_pos = torch.topk(scores, k=k, largest=True).indices
        return [history[pos] for pos in top_pos.tolist()]

    if strategy == "similarity":
        if item_embeddings is None:
            raise ValueError("item_embeddings is required for the similarity strategy.")
        emb = item_embeddings.detach().cpu()
        hist_emb = emb[history_tensor.cpu()]
        target_emb = emb[int(target_item)].unsqueeze(0)
        if similarity == "cosine":
            hist_emb = torch.nn.functional.normalize(hist_emb, dim=-1)
            target_emb = torch.nn.functional.normalize(target_emb, dim=-1)
        elif similarity != "dot":
            raise ValueError("similarity must be 'dot' or 'cosine'.")
        scores = torch.matmul(hist_emb, target_emb.squeeze(0))
        top_pos = torch.topk(scores, k=k, largest=True).indices
        return [history[pos] for pos in top_pos.tolist()]

    if learned_mask_dict is None or user_id not in learned_mask_dict:
        raise KeyError(f"Missing learned CoSRec mask for user {user_id}.")
    mask = learned_mask_dict[user_id].detach().cpu().float()
    if mask.numel() != len(history):
        raise ValueError(
            f"Mask length mismatch for user {user_id}: mask has {mask.numel()} values, "
            f"but history has {len(history)} items. The mask order must match user_history_dict[user_id]."
        )
    low_pos = torch.topk(mask, k=k, largest=False).indices
    return [history[pos] for pos in low_pos.tolist()]


def _mask_seen_items(scores: torch.Tensor, seen_items: Sequence[int]) -> torch.Tensor:
    scores = scores.clone()
    if seen_items:
        scores[torch.tensor(list(seen_items), device=scores.device, dtype=torch.long)] = -torch.inf
    return scores


@torch.no_grad()
def get_user_scores(
    model: torch.nn.Module,
    user_id: int,
    graph_cache: GraphMaskCache,
    edge_weight: Optional[torch.Tensor] = None,
    rating_fn: Optional[RatingFn] = None,
) -> torch.Tensor:
    """Return scores for all items for a single user.

    If your LightGCN exposes a custom graph-aware scoring function, pass it as
    rating_fn(model, user_tensor, edge_weight, graph_cache). Otherwise this
    function supports:
      1. model.get_users_rating(user_tensor, edge_weight=..., graph_cache=...)
      2. model.forward(SparseTensor(...)) followed by user/item dot products.
      3. model.get_users_rating(user_tensor) only for unperturbed inference.
    """
    user_tensor = torch.tensor([int(user_id)], device=graph_cache.device, dtype=torch.long)

    if rating_fn is not None:
        scores = rating_fn(model, user_tensor, edge_weight, graph_cache)
        return scores.squeeze(0)

    if hasattr(model, "get_users_rating"):
        try:
            scores = model.get_users_rating(
                user_tensor,
                edge_weight=edge_weight,
                graph_cache=graph_cache,
            )
            return scores.squeeze(0)
        except TypeError:
            if edge_weight is None:
                scores = model.get_users_rating(user_tensor)
                return scores.squeeze(0)

    if hasattr(model, "forward"):
        sparse_adj = graph_cache.to_sparse_tensor(edge_weight)
        forward_out = model.forward(sparse_adj)
        users_emb, items_emb = forward_out[0], forward_out[2]
        return torch.matmul(users_emb[int(user_id)], items_emb.T)

    raise RuntimeError(
        "No graph-aware scoring path is available. Provide rating_fn or expose "
        "model.forward(SparseTensor) / model.get_users_rating(..., edge_weight=...)."
    )


def topk_items(scores: torch.Tensor, k: int, seen_items: Sequence[int], exclude_seen: bool = True) -> List[int]:
    ranked_scores = _mask_seen_items(scores, seen_items) if exclude_seen else scores
    return torch.topk(ranked_scores, k=k).indices.detach().cpu().tolist()


def evaluate_single_user(
    model: torch.nn.Module,
    user_id: int,
    strategy: StrategyName,
    budget: float,
    *,
    top_k: int,
    graph_cache: GraphMaskCache,
    train_edge_index: torch.Tensor,
    user_history_dict: Mapping[int, Sequence[int]],
    learned_mask_dict: Optional[Mapping[int, torch.Tensor]] = None,
    target_item: Optional[int] = None,
    target_rank: int = 0,
    degree: Optional[torch.Tensor] = None,
    item_embeddings: Optional[torch.Tensor] = None,
    similarity: str = "dot",
    exclude_seen: bool = True,
    rng: Optional[random.Random] = None,
    rating_fn: Optional[RatingFn] = None,
) -> Dict[str, object]:
    """Evaluate one user/strategy/budget tuple."""
    history_items = list(user_history_dict.get(int(user_id), []))
    if not history_items:
        raise ValueError(f"User {user_id} has no history in user_history_dict.")

    original_scores = get_user_scores(model, user_id, graph_cache, edge_weight=None, rating_fn=rating_fn)
    original_topk = topk_items(original_scores, top_k, history_items, exclude_seen=exclude_seen)
    if target_item is None:
        target_item = original_topk[target_rank]

    removal_subset = get_removal_subset(
        strategy,
        user_id,
        history_items,
        int(target_item),
        budget,
        train_edge_index=train_edge_index,
        learned_mask_dict=learned_mask_dict,
        degree=degree,
        item_embeddings=item_embeddings,
        similarity=similarity,
        rng=rng,
    )

    perturbed_weight = graph_cache.edge_weights_with_removed(user_id, removal_subset)
    perturbed_scores = get_user_scores(model, user_id, graph_cache, edge_weight=perturbed_weight, rating_fn=rating_fn)
    perturbed_topk = topk_items(perturbed_scores, top_k, history_items, exclude_seen=exclude_seen)

    original_set = set(original_topk)
    perturbed_set = set(perturbed_topk)
    success = float(int(int(target_item) not in perturbed_set))
    stability = float(len(original_set.intersection(perturbed_set)) / top_k)

    return {
        "user_id": int(user_id),
        "strategy": strategy,
        "budget": float(budget),
        "top_k": int(top_k),
        "target_item": int(target_item),
        "removed_items": removal_subset,
        "original_topk": original_topk,
        "perturbed_topk": perturbed_topk,
        "success": success,
        "stability": stability,
    }


def _markdown_table(rows: Sequence[Dict[str, object]], top_k: int) -> str:
    header = (
        f"| Strategy | Budget | Success@{top_k} | Stability@{top_k} | Users |\n"
        "| --- | ---: | ---: | ---: | ---: |"
    )
    body = []
    for row in rows:
        body.append(
            f"| {row['strategy']} | {row['budget']:.2f} | "
            f"{row['success_rate']:.4f} | {row['stability_rate']:.4f} | {row['num_users']} |"
        )
    return "\n".join([header] + body)


def run_all_experiments(
    model: torch.nn.Module,
    train_edge_index: torch.Tensor,
    user_history_dict: Mapping[int, Sequence[int]],
    learned_mask_dict: Mapping[int, torch.Tensor],
    test_users: Optional[Iterable[int]] = None,
    *,
    budgets: Sequence[float] = (0.05, 0.10, 0.15, 0.20),
    top_k: int = 10,
    strategies: Sequence[StrategyName] = STRATEGIES,
    target_item_dict: Optional[Mapping[int, int]] = None,
    target_rank: int = 0,
    exclude_seen: bool = True,
    similarity: str = "dot",
    random_repeats: int = 1,
    seed: int = 42,
    rating_fn: Optional[RatingFn] = None,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Run all strategy/budget evaluations and print a markdown table.

    Returns:
        detailed_results: one record per evaluated user/trial.
        summary_rows: averaged Success@K and Stability@K per strategy/budget.
    """
    device = next(model.parameters()).device
    num_users, num_items = infer_num_users_items(model, train_edge_index)
    graph_cache = GraphMaskCache(train_edge_index, num_users, num_items, device)
    degree = item_degree(train_edge_index, num_items=num_items)
    item_embeddings = get_item_embedding_matrix(model)

    if test_users is None:
        test_users = list(user_history_dict.keys())
    else:
        test_users = list(test_users)

    detailed_results: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    for strategy in strategies:
        repeats = random_repeats if strategy == "random" else 1
        for budget in budgets:
            trial_results: List[Dict[str, object]] = []
            for repeat_idx in range(repeats):
                rng = random.Random(seed + repeat_idx)
                for user_id in test_users:
                    try:
                        result = evaluate_single_user(
                            model,
                            int(user_id),
                            strategy,
                            budget,
                            top_k=top_k,
                            graph_cache=graph_cache,
                            train_edge_index=train_edge_index,
                            user_history_dict=user_history_dict,
                            learned_mask_dict=learned_mask_dict,
                            target_item=(
                                int(target_item_dict[int(user_id)])
                                if target_item_dict is not None and int(user_id) in target_item_dict
                                else None
                            ),
                            target_rank=target_rank,
                            degree=degree,
                            item_embeddings=item_embeddings,
                            similarity=similarity,
                            exclude_seen=exclude_seen,
                            rng=rng,
                            rating_fn=rating_fn,
                        )
                    except (ValueError, KeyError) as exc:
                        trial_results.append(
                            {
                                "user_id": int(user_id),
                                "strategy": strategy,
                                "budget": float(budget),
                                "skipped": True,
                                "reason": str(exc),
                            }
                        )
                        continue
                    trial_results.append(result)

            valid_results = [r for r in trial_results if not r.get("skipped", False)]
            success_rate = (
                sum(float(r["success"]) for r in valid_results) / len(valid_results)
                if valid_results
                else float("nan")
            )
            stability_rate = (
                sum(float(r["stability"]) for r in valid_results) / len(valid_results)
                if valid_results
                else float("nan")
            )
            summary_rows.append(
                {
                    "strategy": strategy,
                    "budget": float(budget),
                    "top_k": int(top_k),
                    "success_rate": success_rate,
                    "stability_rate": stability_rate,
                    "num_users": len(valid_results),
                }
            )
            detailed_results.extend(trial_results)

    print(_markdown_table(summary_rows, top_k))
    return detailed_results, summary_rows


if __name__ == "__main__":
    print(
        "Import this module and call run_all_experiments(model, train_edge_index, "
        "user_history_dict, learned_mask_dict)."
    )
