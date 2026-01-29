# 🏆 GNN Mini-Challenge: Inductive Node Classification

Welcome to the **Rising Stars GNN Mini-Competition**!

This repository hosts a challenge on **inductive node classification** using Graph Neural Networks (GNNs). Your goal is to build a model that learns from a training graph and generalizes to completely unseen nodes.

## 🎯 The Task
Given a citation graph with node features, predict the research topic (class) of **unseen nodes** using inductive learning.

**The Inductive Constraint:**
The model must make predictions for nodes that were **completely unseen** during training (possibly in new graph regions). You must use only learned parameters (weights), not memorized node IDs or embeddings.

## 📂 Dataset
We use the **Cora citation network**:
* **Nodes:** Scientific papers
* **Edges:** Citation relationships
* **Node features:** Bag-of-words vectors
* **Labels:** Research topics (7 classes)

### File Description
The `data/` folder contains:
* `train.csv`: Labeled nodes for training (IDs, Features, Labels).
* `edge_list.csv`: Edges connecting training nodes.
* `test.csv`: **Unseen** test nodes (IDs, Features). **No Labels**.
* `test_edges.csv`: Edges involving test nodes (used for the inductive inference step).

## 🚀 How to Participate

### 1. Get the Code
Clone this repository and install the dependencies:
```bash
git clone https://github.com/emmakowu3579-ui/inductive-class-challenge.git
cd inductive-class-challenge
pip install -r starter_code/requirements.txt
