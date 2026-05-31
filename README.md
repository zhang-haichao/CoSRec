# CoSRec

Counterfactual Support Sets for Graph Recommendation.

This repository is organized around the experiments reported in the current paper. The active pipeline uses a LightGCN recommender backbone, CoSRec support-set learning, RQ-specific experiment entry points, paper figure scripts, and compact paper-facing results under `results/paper/`.

## Repository Layout

```text
CoSRec/
|-- recsys/                         # LightGCN backbone and original trainer
|-- cf/                             # CoSRec counterfactual mask implementation
|-- experiments/
|   |-- training/                   # Backbone training
|   |-- rq1/                        # Validity-specificity trade-off
|   |-- rq2/                        # Random, High-Degree, Attention-proxy, FIA, CoSRec comparison
|   |-- rq3/                        # User-level qualitative case generation
|   `-- rq4/                        # Lambda sensitivity
|-- analysis/figures/               # Scripts that generate paper figures
|-- data/                           # MovieLens-100K and MovieLens-1M files
|-- model/                          # Trained LightGCN checkpoints
|-- results/paper/
|   |-- rq1/                        # Table data and Top-1 trade-off figure
|   |-- rq2/                        # Baseline comparison summaries and figure
|   |-- rq3/                        # Case-study data and RQ3 figure
|   `-- rq4/                        # Lambda sensitivity CSV and figure
|-- legacy/                         # Old exploratory scripts, not part of reproduction
`-- README.md
```

## Installation

Run commands from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy pandas scikit-learn matplotlib
pip install torch
pip install torch-geometric torch-sparse
```

Depending on your PyTorch version, install `torch-sparse` from the wheel index recommended by the PyG installation guide.

## Active Checkpoints

The paper-facing experiments use:

```text
model/light_GCN_corrected_negfix.pt
model/light_GCN_ml1m_strict_t4.pt
```

The original `model/light_GCN.pt` is kept for historical comparison only.

## RQ1: Validity and Specificity Trade-off

Code:

```bash
python -m experiments.rq1.validity_specificity_tradeoff \
  --datasets ML-100K ML-1M \
  --max_users 100 \
  --seed 42 \
  --cosrec_epochs 8 \
  --cosrec_lr 0.2 \
  --lambda_graph 2 \
  --beta 5 \
  --ml100k_model model/light_GCN_corrected_negfix.pt \
  --ml1m_model model/light_GCN_ml1m_strict_t4.pt \
  --output_dir results/paper/rq1
```

Paper-facing stored outputs:

```text
results/paper/rq1/rq1_main_results.csv
results/paper/rq1/rq1_top1_tradeoff.pdf
results/paper/rq1/rq1_top1_tradeoff.png
```

Regenerate the paper figure:

```bash
python -m analysis.figures.rq1_tradeoff_plot
```

## RQ2: Counterfactual Performance Comparison

Run the all-user baseline comparison at 5% and 10% budgets:

```bash
python -m experiments.rq2.baseline_comparison \
  --budget_percent 0.05 \
  --max_users 943 \
  --target_rank 0 \
  --success_metric top1_changed \
  --methods Random,High-Degree,Attention-proxy,FIA,CoSRec \
  --cosrec_epochs 20 \
  --cosrec_lr 0.2 \
  --lambda_graph 2 \
  --beta 5.0 \
  --cosrec_objective boundary_margin \
  --hessian_candidates 8 \
  --model_path model/light_GCN_corrected_negfix.pt \
  --layers 3 \
  --graph_scope full \
  --embedding_readout all_layers \
  --output_dir results/paper/rq2
```

```bash
python -m experiments.rq2.baseline_comparison \
  --budget_percent 0.10 \
  --max_users 943 \
  --target_rank 0 \
  --success_metric top1_changed \
  --methods Random,High-Degree,Attention-proxy,FIA,CoSRec \
  --cosrec_epochs 20 \
  --cosrec_lr 0.2 \
  --lambda_graph 2 \
  --beta 5.0 \
  --cosrec_objective boundary_margin \
  --hessian_candidates 8 \
  --model_path model/light_GCN_corrected_negfix.pt \
  --layers 3 \
  --graph_scope full \
  --embedding_readout all_layers \
  --output_dir results/paper/rq2
```

The compact paper summaries are stored as:

```text
results/paper/rq2/rq2_budget5_summary.json
results/paper/rq2/rq2_budget5_table.md
results/paper/rq2/rq2_budget10_summary.json
results/paper/rq2/rq2_budget10_table.md
```

Regenerate the grouped comparison figure:

```bash
python -m analysis.figures.rq2_baseline_bar
```

Figure outputs:

```text
results/paper/rq2/rq2_success_at1_grouped_bar.pdf
results/paper/rq2/rq2_success_at1_grouped_bar.png
```

## RQ3: Qualitative User-Level Explanations

Generate a single user-level case:

```bash
python -m experiments.rq3.case_study_ml100k \
  --user_id 53 \
  --target_index 6 \
  --rate 0.20 \
  --max_support 3 \
  --top_k 10 \
  --epochs 50 \
  --lr 0.08 \
  --optimizer SGD \
  --objective score \
  --lambda_val 10 \
  --beta 0.1 \
  --seed 1 \
  --model_path model/light_GCN.pt \
  --output results/paper/rq3/case_user53.json
```

Paper-facing qualitative cases:

```text
results/paper/rq3/rq3_case_studies.json
results/paper/rq3/rq3_case_studies.md
results/paper/rq3/RQ3.png
```

Regenerate the RQ3 figure:

```bash
python -m analysis.figures.rq3_case_graph
```

## RQ4: Hyperparameter Sensitivity

Run lambda sensitivity:

```bash
python -m experiments.rq4.lambda_sensitivity \
  --lambdas 0.01 0.1 1 10 30 100 300 \
  --budget_percent 0.10 \
  --max_users 100 \
  --ordered_users \
  --cosrec_epochs 10 \
  --cosrec_lr 0.2 \
  --beta 5.0 \
  --cosrec_objective boundary_margin \
  --model_path model/light_GCN_corrected_negfix.pt \
  --layers 3 \
  --graph_scope full \
  --embedding_readout all_layers \
  --selection_mode adaptive_expected \
  --results_csv results/paper/rq4/rq4_lambda_sensitivity.csv \
  --plot_path results/paper/rq4/rq4_lambda_sensitivity.pdf \
  --work_dir results/paper/rq4/lambda_runs
```

Paper-facing stored outputs:

```text
results/paper/rq4/rq4_lambda_sensitivity.csv
results/paper/rq4/rq4_lambda_sensitivity_1x3.pdf
results/paper/rq4/rq4_lambda_sensitivity_1x3.png
```

Regenerate the paper figure:

```bash
python -m analysis.figures.rq4_lambda_sensitivity_plot
```

## Current Paper Results

RQ2 all-user comparison on MovieLens-100K:

| Method | 5% Success@1 | 5% Stability@10 | 10% Success@1 | 10% Stability@10 |
| --- | ---: | ---: | ---: | ---: |
| Random | 1.48 | 0.9839 | 2.23 | 0.9765 |
| High-Degree | 1.59 | 0.9841 | 2.65 | 0.9726 |
| Attention-proxy | 5.73 | 0.9760 | 9.86 | 0.9581 |
| FIA | 6.89 | 0.9682 | 12.20 | 0.9452 |
| CoSRec | 13.47 | 0.9566 | 22.16 | 0.9315 |

RQ4 lambda sensitivity is stored in `results/paper/rq4/rq4_lambda_sensitivity.csv`.

## Cleanup Policy

The repository keeps only compact paper-facing results. Old probe, smoke, autotune, log, and duplicate figure outputs were removed. Historical scripts are isolated in `legacy/` and should not be used for paper reproduction.
