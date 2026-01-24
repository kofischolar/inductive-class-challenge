# GNN Mini-Challenge: Inductive Node Classification

This repository hosts a mini-competition on **inductive node classification**
using Graph Neural Networks (GNNs).

## Task
Given a citation graph with node features, predict the research topic
of unseen nodes using **inductive learning**. I want to reiterate that the model must make predictions for nodes that were completely unseen during training, possibly in new graph regions or new graphs, using only learned parameters, not memorized node IDs or embeddings.

## Dataset
We use the **Cora citation network**:
- Nodes: scientific papers
- Edges: citation relationships
- Node features: bag-of-words
- Labels: research topics

## Learning Setting
- Inductive learning
- No access to test node labels during training
- Message passing allowed only on training graph

## Evaluation Metric
- Macro F1-score

## Restrictions
- No external datasets
- Training time < 5 minutes (Colab)
- Any GNN from DGL lectures 1.1–4.6 allowed

## Baseline
A simple 2-layer GCN is provided.

## Goal
Beat the baseline macro-F1 score.
