# CoSRec

**Counterfactual Support Sets for Graph Recommendation**

CoSRec is a research prototype for explaining graph-based recommendations through sparse, user-local counterfactual interventions. Instead of asking which historical interactions merely correlate with a recommendation, CoSRec asks a sharper question:

> Which observed user-item interactions must be weakened or removed for the frozen recommender to change its top-ranked result?

The project implements this idea on top of LightGCN. A trained recommender is kept fixed, a continuous retention mask is optimized over the target user's historical edges, and the lowest-retention edges are returned as a counterfactual support set with contribution weights. The result is an explanation that can be inspected, ranked, and verified by recomputing the recommendation under the perturbed graph.

## Why CoSRec

Modern graph recommenders propagate collaborative signals across user-item interaction graphs. This gives strong ranking performance, but it also makes individual recommendations difficult to interpret: a top item can be supported by direct history, high-order graph neighbors, and ranking competition against many plausible alternatives.

CoSRec turns explanation into an intervention problem:

- **Factual pass:** record the original Top-K recommendation list from a frozen LightGCN model.
- **Counterfactual pass:** learn a retention mask only over the target user's historical interactions.
- **Sparse intervention:** weaken a small number of user-local edges while preserving most of the graph.
- **Verification:** recompute the recommendation list with the same frozen model.
- **Explanation:** return the removed or weakened history items as a counterfactual support set, ranked by contribution strength.

This makes the explanation stronger than a similarity score, attention score, or prediction-preserving subgraph: an interaction is explanatory only when intervening on it changes the recommendation.

## Method at a Glance

For a target user `u`, CoSRec optimizes a retention vector `p_u` over the user's historical item set `H_u`.

```text
Original graph A
        |
        v
Frozen LightGCN f_theta
        |
        v
Original recommendation list R_K(u; A)

Target-user history H_u
        |
        v
Retention mask p_u in [0, 1]
        |
        v
Perturbed graph A' = A * P
        |
        v
Same frozen LightGCN f_theta
        |
        v
Counterfactual list R_K(u; A')
```

After optimization, low-retention edges become the support set:

```text
S_u = { i in H_u : p_ui < threshold }
w_u(i) = 1 - p_ui
E_u = (S_u, w_u)
```

Here, `S_u` identifies the historical interactions that support the recommendation, and `w_u` ranks them by removal strength.

## Repository Structure

```text
CoSRec/
|-- main.py                         # Main single-user counterfactual explanation entry point
|-- recsys/
|   |-- LightGCN.py                 # LightGCN backbone
|   `-- train.py                    # LightGCN training script
|-- cf/
|   |-- cf_explainer.py             # Counterfactual explainer around frozen LightGCN
|   `-- gcn_perturb.py              # Perturbable LightGCN module with learnable edge mask
|-- data/
|   |-- ml_100k.py                  # MovieLens 100K loader
|   `-- ml-100k/u.data              # Included MovieLens 100K interaction file
|-- model/
|   `-- light_GCN.pt                # Saved LightGCN checkpoint used by explanation scripts
|-- unlearning/
|   |-- cci.py                      # Causal Contribution Identifier
|   |-- ura.py                      # Unlearning-driven Rank Adjuster
|   `-- ceu_framework.py            # Explain-first-then-forget framework
|-- utils/
|   `-- utils.py                    # Ranking, graph, and utility helpers
`-- *_explanation_generator.py      # Convenience scripts for user-level explanation reports
```

## Core Components

| Component | Role |
| --- | --- |
| `recsys/LightGCN.py` | Implements the LightGCN message-passing recommender with user and item embeddings. |
| `cf/gcn_perturb.py` | Adds a learnable perturbation vector `P_vec` over the target user's historical edges. |
| `cf/cf_explainer.py` | Freezes the trained LightGCN, optimizes the mask, and compares original vs. perturbed recommendations. |
| `main.py` | Loads MovieLens 100K, builds train/test splits, selects a user, and writes counterfactual examples to `results/`. |
| `unlearning/` | Extends the explanation signal into a Causal Explanation-based Unlearning workflow. |

## Installation

The code is written in Python and uses PyTorch, PyTorch Geometric, and PyTorch Sparse.

```bash
git clone git@github.com:zhang-haichao/CoSRec.git
cd CoSRec

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

Install the core dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib
pip install torch
pip install torch-geometric torch-sparse
```

Depending on your operating system, Python version, CUDA version, and PyTorch version, `torch-sparse` may require the wheel index from the PyTorch Geometric installation guide.

## Quick Start

### 1. Train or refresh the LightGCN checkpoint

The repository includes `model/light_GCN.pt`. To regenerate it from MovieLens 100K:

```bash
python recsys/train.py
```

The training script uses:

| Setting | Value |
| --- | --- |
| Dataset | MovieLens 100K |
| Split | 80/10/10 train/validation/test |
| Embedding size | 64 |
| Propagation layers | 3 |
| Optimizer | Adam |
| Iterations | 500 |
| Batch size | 2048 |
| Learning rate | 0.002 |
| Evaluation cutoff | Top-20 |

### 2. Generate a counterfactual explanation

Run the main explainer for a target MovieLens user:

```bash
python main.py --user_id 196 --target_index 2 --rate 0.2 --lr 0.08 --num_epochs 50
```

Important arguments:

| Argument | Meaning |
| --- | --- |
| `--user_id` | Original MovieLens user id to explain. |
| `--target_index` | Zero-based index of the recommendation position to target. |
| `--rate` | Fraction of the user's historical edges selected for hard perturbation after mask optimization. |
| `--lr` | Learning rate for the counterfactual mask optimizer. |
| `--optimizer` | Optimizer for the explainer, such as `SGD`, `Adam`, or `Adadelta`. |
| `--num_epochs` | Number of optimization epochs for the retention mask. |

Outputs are written to:

```text
results/ml-100k/
```

Each saved explanation records the user, history size, perturbation rate, original recommendation list, and perturbed recommendation list.

### 3. Generate richer explanation reports

The repository also includes convenience scripts for user-level analysis:

```bash
python generate_explanations.py --user_id 196
python final_explanation_generator.py --user_id 196 --top_k 3
python top_influence_edges_generator.py --user_id 196
```

These scripts map optimized mask values back to historical user-item edges and help inspect which past interactions most strongly support specific recommendations.

## Causal Explanation-based Unlearning

The `unlearning/` package builds on the same counterfactual support idea and implements an explain-first-then-forget workflow:

- **CCI: Causal Contribution Identifier** finds support sets and contribution degrees for recommendations.
- **URA: Unlearning-driven Rank Adjuster** adjusts recommendation scores using causal contribution signals without retraining the full model.
- **CEUFramework** coordinates single-user and batch unlearning requests.

Run a simple demonstration:

```bash
python unlearning/simple_unlearning_demo.py
```

Run a fuller experiment:

```bash
python unlearning/full_unlearning_experiment.py --forget_ratio 0.01
```

## Experimental Results Reported in the Paper

The accompanying paper evaluates CoSRec on MovieLens 100K and MovieLens 1M. The shipped code and data in this repository focus on the MovieLens 100K implementation path; the paper reports the broader evaluation.

Base LightGCN quality before explanation:

| Metric | Value |
| --- | ---: |
| Recall@20 | 15.068% |
| Precision@20 | 6.444% |
| NDCG@20 | 11.788% |

Counterfactual success rates:

| Dataset | Target | 5% | 10% | 15% | 20% |
| --- | --- | ---: | ---: | ---: | ---: |
| MovieLens 100K | Top-1 | 83.53 | 85.53 | 86.73 | 87.72 |
| MovieLens 100K | Top-2 | 81.73 | 83.53 | 84.63 | 85.63 |
| MovieLens 100K | Top-3 | 72.14 | 72.66 | 76.34 | 77.18 |
| MovieLens 1M | Top-1 | 81.03 | 83.32 | 85.17 | 86.45 |
| MovieLens 1M | Top-2 | 80.62 | 81.41 | 83.23 | 84.76 |
| MovieLens 1M | Top-3 | 73.94 | 75.83 | 77.48 | 78.06 |

Recommendation-list stability:

| Dataset | Target | 5% | 10% | 15% | 20% |
| --- | --- | ---: | ---: | ---: | ---: |
| MovieLens 100K | Top-1 | 91.70 | 91.73 | 90.72 | 90.70 |
| MovieLens 100K | Top-2 | 82.56 | 82.34 | 81.68 | 81.43 |
| MovieLens 100K | Top-3 | 74.04 | 73.95 | 73.14 | 73.09 |
| MovieLens 1M | Top-1 | 93.50 | 93.55 | 93.53 | 93.50 |
| MovieLens 1M | Top-2 | 84.60 | 84.68 | 84.72 | 83.75 |
| MovieLens 1M | Top-3 | 73.15 | 72.90 | 72.20 | 71.25 |

These results show the key trade-off CoSRec is designed to expose: sparse interventions can flip the target recommendation while preserving most of the surrounding recommendation context.

## Output Interpretation

A CoSRec explanation should be read as:

```text
For user u, recommendation v is supported by historical interactions S_u.
If the interactions in S_u are weakened or removed, v drops from the target recommendation position.
The weight w_u(i) tells how strongly each history item i contributed to the original recommendation.
```

This makes the output useful for:

- auditing individual graph recommendations;
- debugging how historical interactions support a top-ranked item;
- generating user-facing explanation snippets;
- downstream unlearning or rank-adjustment experiments.

## Notes and Limitations

- The current implementation path is centered on MovieLens 100K and LightGCN.
- The paper also reports MovieLens 1M results; reproducing that setting requires adding the corresponding data loader and preprocessing path.
- The explainer is post-hoc: it does not retrain the recommender while generating explanations.
- The code is a research prototype and intentionally exposes intermediate scripts for experimentation and inspection.

## Citation

If this repository supports your research, please cite the accompanying manuscript:

```bibtex
@inproceedings{cosrec2026,
  title     = {Explaining Graph Recommendations via Counterfactual Support Sets},
  author    = {Anonymous Authors},
  booktitle = {ICONIP},
  year      = {2026}
}
```
