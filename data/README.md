# 📂 Dataset Overview: NuCLS

This folder contains the **Cell-Graph dataset** for the inductive node classification challenge. The data represents cells (nodes) and their spatial neighbors (edges) derived from H&E stained tissue images.

---

## 📄 File Descriptions

### 1. Training Graph (Labeled)
* **`train.csv`**: Contains the node features and ground-truth labels for training.
    * **Columns:** `id`, `x`, `y`, `width`, `height`, `label`
    * **Labels:** `0` (Tumor), `1` (Stromal), `2` (Lymphocyte), `3` (Macrophage)
* **`edge_list.csv`**: Contains the edges (adjacency list) for the training nodes.
    * **Columns:** `source`, `target` (Refers to `id` in `train.csv`)
    * **Construction:** 5-Nearest Neighbors (Spatial Proximity)

### 2. Test Graph (Unlabeled)
* **`test_nodes.csv`**: Contains the node features for the test set. **Labels are hidden.**
    * **Columns:** `id`, `x`, `y`, `width`, `height`
    * **Task:** Predict the `label` for every `id` in this file.
* **`test_edges.csv`**: Contains the edges for the test nodes.
    * **Columns:** `source`, `target` (Refers to `id` in `test_nodes.csv`)
    * **Usage:** Use these edges to perform message passing (graph convolution) during inference.

### 3. Submission Example
* **`sample_submission.csv`**: A valid example file showing the required format.
    * **Columns:** `id`, `y_pred` (Class Integer: 0, 1, 2, or 3)

---

## 📊 Feature Details
All features have been **normalized** (Standard Scaler).
* **`x`, `y`**: Centroid coordinates of the cell nucleus.
* **`width`, `height`**: Morphological dimensions of the nucleus bounding box.
