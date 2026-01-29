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

### 2. Run the Baseline
We provide a pure PyTorch GCN baseline in the `starter_code/` folder.

```bash
python starter_code/baseline.py

This script will train a simple GCN and generate a submission file at submissions/baseline_submission.csv.

3. Make a Submission
Generate your predictions. Your CSV file must follow this exact format:

Code snippet

id,label
1800,3
1801,0
1802,4
...
(Header is required: id,label)

Upload to GitHub:

Save your file in the submissions/ folder (e.g., submissions/my_solution.csv).

Commit the file to a new branch.

Open a Pull Request (PR) against the main branch.

🤖 Instant Grading
Once you open a PR, our Auto-Grader Bot will run instantly.

It calculates your Macro F1-Score.

It posts a comment on your PR with your result.

If your score is valid, the admin will merge it, and your name will appear on the Leaderboard!

📏 Rules & Restrictions
Evaluation Metric: Macro F1-score.

Inductive Setting: No access to test node labels during training.

Message Passing: Allowed only on the training graph during training; test edges are for inference only.

External Data: Strictly forbidden.

Time Limit: Training must take < 5 minutes on Google Colab (Standard GPU/CPU).

Libraries: Any standard GNN library is allowed (PyTorch, DGL, PyG).

🏆 Leaderboard
Check the current rankings here: View Leaderboard
