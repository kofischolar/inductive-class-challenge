import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import sys

print("🚀 Starting Baseline Tumor-GNN...")

# 1. CONFIGURATION
# Assumes you run this from the repository ROOT folder
TRAIN_PATH = 'data/public/train.csv'
TEST_PATH = 'data/public/test_nodes.csv'     # FIXED: filename
EDGE_PATH = 'data/public/edge_list.csv'
TEST_EDGE_PATH = 'data/public/test_edges.csv'
OUTPUT_PATH = 'submission.csv'

# 2. LOAD DATA
print("   - Loading CSVs...")
try:
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    edge_list = pd.read_csv(EDGE_PATH)
    test_edges = pd.read_csv(TEST_EDGE_PATH)
except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    print("   (Make sure you are running this from the repo root!)")
    sys.exit(1)

# 3. PREPARE GRAPH
# In this baseline, we cheat slightly by merging Train and Test into one big disconnected graph.
# This works because there are NO edges between Train and Test, so information won't leak.

print("   - Building Adjacency Matrix...")

# Concatenate Nodes to create a unified ID map
all_nodes = pd.concat([train_df, test_df], ignore_index=True)
# Map original "id" -> 0, 1, 2... index
id_map = {row_id: i for i, row_id in enumerate(all_nodes['id'].values)}
num_nodes = len(all_nodes)

# Combine Edges
all_edges = pd.concat([edge_list, test_edges])

# FIXED: Use 'source' and 'target' (not source_id)
src = [id_map[i] for i in all_edges['source']]
dst = [id_map[i] for i in all_edges['target']]

# Add Self-Loops (A cell is its own neighbor)
src.extend(range(num_nodes))
dst.extend(range(num_nodes))

# Build Sparse Matrix
indices = torch.tensor([src, dst], dtype=torch.long)
values = torch.ones(len(src), dtype=torch.float32)

# Degree Normalization (D^-0.5 * A * D^-0.5)
# This is standard GCN preprocessing
row_sum = torch.zeros(num_nodes)
row_sum.index_add_(0, indices[0], values)
deg_inv_sqrt = row_sum.pow(-0.5)
deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
values = values * deg_inv_sqrt[indices[0]] * deg_inv_sqrt[indices[1]]

adj = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes))

# 4. PREPARE FEATURES
# Drop non-feature columns
# We keep only: x, y, width, height (normalized)
feat_cols = [c for c in train_df.columns if c not in ['id', 'label', 'y_pred']]
print(f"   - Using features: {feat_cols}")

# Stack features in the correct order (0..N)
features = torch.tensor(all_nodes[feat_cols].values, dtype=torch.float32)

# Prepare Labels (Only for Train nodes)
# We need to map train IDs to our new 0..N indices to be safe
train_indices = [id_map[i] for i in train_df['id']]
train_mask = torch.tensor(train_indices, dtype=torch.long)
train_labels = torch.tensor(train_df['label'].values, dtype=torch.long)

# 5. DEFINE MODEL
class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        # Simple 2-Layer GCN
        self.W1 = nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(in_feats, h_feats)))
        self.W2 = nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(h_feats, num_classes)))

    def forward(self, adj, x):
        h = torch.spmm(adj, x)       # Aggregate neighbors
        h = torch.mm(h, self.W1)     # Linear 1
        h = F.relu(h)
        h = torch.spmm(adj, h)       # Aggregate neighbors
        h = torch.mm(h, self.W2)     # Linear 2
        return h

# 6. TRAIN
# FIXED: num_classes = 4 (Tumor, Stroma, Lymphocyte, Macrophage)
model = GCN(features.shape[1], 16, 4) 
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("   - Training...")
model.train()
for e in range(201):
    logits = model(adj, features)
    # Only calculate loss on Training Nodes
    loss = F.cross_entropy(logits[train_mask], train_labels)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if e % 50 == 0:
        print(f"     Epoch {e}, Loss: {loss.item():.4f}")

# 7. PREDICT & SAVE
print("   - Generating Predictions...")
model.eval()
with torch.no_grad():
    logits = model(adj, features)
    
    # Extract predictions for Test Nodes only
    test_indices = [id_map[i] for i in test_df['id']]
    test_logits = logits[test_indices]
    test_preds = test_logits.argmax(1)

# Create Submission DataFrame
submission = pd.DataFrame({
    'id': test_df['id'],
    'y_pred': test_preds.numpy()
})

submission.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Saved submission to {OUTPUT_PATH}")
print(f"   (Rows: {len(submission)})")
