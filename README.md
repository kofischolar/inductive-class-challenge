## 🏆 GNN Mini-Challenge: Inductive Node Classification

Welcome to the **Rising Stars GNN Mini-Competition** 🚀

This repository hosts a hands-on challenge on **inductive node classification** using **Graph Neural Networks (GNNs)**. Your task is to train a model on a given graph and **generalize to completely unseen nodes**.

---

## 🎯 Challenge Overview

You are given a citation network with node features and labels for training nodes only.  
Your goal is to **predict the research topic of unseen nodes** using an **inductive GNN model**.

### 🔍 What Makes This Inductive?

* Test nodes are **not present during training**
* Their IDs and labels are **never seen**
* The model must rely **only on learned parameters**, not memorized node embeddings

> **Train once, generalize to new nodes.**

---

## 📂 Dataset Description

We use the **Cora citation network**, a standard benchmark in graph learning.

### Graph Components

* **Nodes:** Scientific papers
* **Edges:** Citation relationships
* **Node features:** Bag-of-words vectors
* **Labels:** Research topics (**7 classes**)

### 📁 Files in `data/`

* `train.csv` — Training nodes (**IDs, features, labels**)
* `edge_list.csv` — Edges between training nodes
* `test.csv` — **Unseen test nodes** (**IDs, features only**)
* `test_edges.csv` — Edges involving test nodes (**used only at inference time**)

---

## 🚀 How to Participate

### 1️⃣ Clone the Repository

```
git clone https://github.com/emmakowu3579-ui/inductive-class-challenge.git
cd inductive-class-challenge
pip install -r starter_code/requirements.txt
```

---

### 2️⃣ Run the Baseline Model

A simple **PyTorch GCN baseline** is provided in the `starter_code/` directory.

```
python starter_code/baseline.py
```

This will:

* Train a basic GCN on the training graph
* Generate a submission file at  
  `submissions/baseline_submission.csv`

---

### 3️⃣ Create a Submission

Your prediction file **must** follow this exact format:

```
id,label
1800,3
1801,0
1802,4
...
```

**Important:**

* Header (`id,label`) is **required**
* One row per test node
* Labels must be integers in **[0–6]**

---

### 4️⃣ Submit via GitHub

1. Save your file in the `submissions/` folder  
   *(e.g., `submissions/my_solution.csv`)*
2. Commit your changes to a **new branch**
3. Open a **Pull Request (PR)** against the `main` branch

---

## 🤖 Instant Grading

Once your PR is opened:

* ✅ An **Auto-Grader Bot** runs automatically
* 📊 Your **Macro F1-Score** is computed
* 💬 The score is posted as a comment on your PR

If the submission is valid:

* The PR will be merged by an admin
* 🎉 Your name appears on the **Leaderboard**

---

## 📏 Rules & Restrictions

* **Evaluation Metric:** Macro F1-Score
* **Inductive Setting:**

  * No access to test labels during training
  * No memorization of node IDs or embeddings
* **Message Passing:**

  * Allowed **only on the training graph** during training
  * Test edges may be used **only at inference time**
* **External Data:** ❌ **Strictly forbidden**
* **Runtime Constraint:**

  * Training must finish in **< 5 minutes** on Google Colab (CPU/GPU)
* **Libraries:**

  * Any standard GNN library is allowed
  * Examples: **PyTorch, PyTorch Geometric (PyG), DGL**

---

## 🏆 Leaderboard
📈 View Leaderboard
