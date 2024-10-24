# Counterfactual-Explainer-for-Graph-based-Recommendation

> This project focuses on developing a counterfactual explanation framework for graph-based recommendation systems using causal reasoning. The implementation leverages PyTorch and PyTorch Geometric to build the core models. The base recommendation model used is LightGCN, which serves as the foundation for generating recommendations.

## Project Overview

This project aims to develop a counterfactual explanation framework for graph-based recommendation systems using causal reasoning. The project uses PyTorch and PyTorch Geometric to build the core models, with LightGCN serving as the foundation for the recommendation system.

## Features

- Counterfactual explanations based on causal reasoning
- Uses LightGCN as the core recommendation system
- Supports recommendation tasks with graph-structured data
- Efficient implementation using PyTorch and PyTorch Geometric

## Repository URL

[Counterfactual-Explainer-for-Graph-based-Recommendation](https://github.com/Sein-Hust/Counterfactual-Explainer-for-Graph-based-Recommendation)

## Installation Guide

### Prerequisites

Ensure your system has the following software installed:

- Python 3.x
- PyTorch
- PyTorch Geometric
- Other dependencies can be installed via `requirements.txt`

### Installation Steps

1. Clone the repository locally:

   ```bash
   git clone https://github.com/Sein-Hust/Counterfactual-Explainer-for-Graph-based-Recommendation.git
   ```

2. Train the LightGCN model:

   First, run the `train.py` file in the `recsys` directory to train the LightGCN model. The model parameters will be saved in the `model` directory:

   ```bash
   python recsys/train.py
   ```

3. Generate counterfactual explanations:

   After the model training is complete, run the `main.py` file to generate counterfactual explanations for a specific user. You can modify the model parameters and the user ID to be explained in lines 251-270 of `main.py`. The generated explanations will be saved in the `result` directory under the corresponding dataset folder:

   ```bash
   python main.py
   ```