# 🎗️ Tumor Diagnosis Challenge: Cell-Graph Node Classification

Welcome to the **Tumor Diagnosis Challenge** 🚀

This competition bridges **Biomedical Engineering** and **Graph Machine Learning**. Your task is to build a model that can diagnose the type of a cell (Tumor, Stroma, Immune, etc.) based on its features and its spatial neighbors in a tissue biopsy.

---

## 🎯 The Task: Inductive Node Classification

You are provided with cell graphs constructed from H&E stained histology images.
* **Training Phase:** You receive full graphs (nodes, edges, and labels) from a set of patients (e.g., Patient A, Patient B).
* **Testing Phase:** You must predict the cell types for **completely unseen patients** (e.g., Patient C).

### 🔍 Why "Inductive"?
Unlike standard transductive tasks (like Cora), the test nodes belong to **entirely new graphs** (new tissue slides) that were not present during training. Your model must learn general rules about tissue organization, not just memorize a specific graph structure.

---

## 📂 The Dataset (NuCLS-Based)

The data is derived from the **NuCLS dataset** (breast cancer).

Gemini said
Here is the concise "Graph Construction" section ready to be pasted into your README.md.

I recommend adding this right after the "📂 The Dataset" section.

## 🏗️ Graph Construction Pipeline
The graph was built using the following inductive pipeline to ensure biological realism:

**Node Definition:**
Raw bounding boxes from the NuCLS dataset were converted into centroids (x,y).
Each node represents a single cell nucleus.

**Edge Construction (k-NN):**
For every cell, we computed its 5 nearest spatial neighbors within the same tissue image.
Edges represent the local tissue microenvironment (e.g., cell-cell interactions).
**Note:** Edges strictly connect cells within the same image; there are no edges between different patients.

**Inductive Split:**
The dataset was split by Image ID, not by random cells.

**Training Graph:** Contains 80% of the tissue images.

**Test Graph:** Contains the remaining 20% of images (completely unseen patients).

**Feature Normalization:**
Node features (x, y, width, height) are standardized (zero mean, unit variance) to ensure stable GNN training.

### 1. The Graph Components
* **Nodes:** Individual cells (nuclei).
* **Edges:** Spatial proximity (k-Nearest Neighbors, $k=5$). If two cells are physically close, they are connected.
* **Node Features ($X$):**
    * `x`, `y`: Normalized coordinates.
    * `width`, `height`: Morphological features of the nucleus.
* **Labels ($Y$):** The biological type of the cell.
    * `0`: **Tumor** (Malignant)
    * `1`: **Stromal** (Connective tissue)
    * `2`: **Lymphocyte** (Immune cells)
    * `3`: **Macrophage** (Immune cells)

### 2. File Structure (`data/public/`)
* `train.csv`: Training nodes with labels. Columns: `[id, x, y, width, height, label]`
* `edge_list.csv`: Edges for the training graph. Columns: `[source, target]`
* `test_nodes.csv`: **Unseen test nodes** (No labels). Columns: `[id, x, y, width, height]`
* `test_edges.csv`: Edges for the test graph. Columns: `[source, target]`
* `sample_submission.csv`: Example format for your predictions.

---

## 📝 Submission Format

You must submit a single CSV file named `predictions.csv`.

**Format:**
```csv
id,y_pred
41269,0
41270,2
...

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
[📈 View Leaderboard](LEADERBOARD.md)
