import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

print("🚀 Starting Baseline GCN...")

# 1. CONFIGURATION
# Update paths to match the new folder structure (data/public)
TRAIN_PATH = '../data/public/train.csv'
TEST_PATH = '../data/public/test.csv'
EDGE_PATH = '../data/public/edge_list.csv'
TEST_EDGE_PATH = '../data/public/test_edges.csv'
OUTPUT_PATH = '../submissions/baseline_submission.csv'

# 2. LOAD DATA
print("   - Loading CSVs from data/public/...")
try:
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    edge_list = pd.read_csv(EDGE_PATH)
    test_edges = pd.read_csv(TEST_EDGE_PATH)
except FileNotFoundError:
    print(f"❌ Error: Files not found in data/public/. Check your directory structure.")
    exit()

# 3. PREPARE GRAPH (Pure PyTorch Implementation)
# Map IDs to 0..N
all_ids = np.concatenate([train_df['id'].values, test_df['id'].values])
id_map = {original_id: i for i, original_id in enumerate(all_ids)}
num_nodes = len(all_ids)

print("   - Building Adjacency Matrix...")
# Combine train and test edges for the structure
all_edges = pd.concat([edge_list, test_edges])
src = [id_map[i] for i in all_edges['source_id']]
dst = [id_map[i] for i in all_edges['target_id']]

# Add self-loops (crucial for GCN)
src.extend(range(num_nodes))
dst.extend(range(num_nodes))

# Build Sparse Matrix (Normalized)
indices = torch.tensor([src, dst], dtype=torch.long)
values = torch.ones(len(src), dtype=torch.float32)

# Calculate Degree Normalization (D^-0.5 * A * D^-0.5)
row_sum = torch.zeros(num_nodes)
row_sum.index_add_(0, indices[0], values)
deg_inv_sqrt = row_sum.pow(-0.5)
deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
values = values * deg_inv_sqrt[indices[0]] * deg_inv_sqrt[indices[1]]

adj = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes))

# 4. PREPARE FEATURES & LABELS
# Extract feature columns (assuming they are all columns except id, label, y_pred)
feat_cols = [c for c in train_df.columns if c not in ['id', 'label', 'y_pred']]
train_feats = torch.tensor(train_df[feat_cols].values, dtype=torch.float32)
test_feats = torch.tensor(test_df[feat_cols].values, dtype=torch.float32)
features = torch.cat([train_feats, test_feats])

train_labels = torch.tensor(train_df['label'].values, dtype=torch.long)
train_mask = torch.arange(len(train_df)) # First N nodes are train

# 5. DEFINE MODEL
class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.W1 = nn.Parameter(torch.empty(in_feats, h_feats))
        self.W2 = nn.Parameter(torch.empty(h_feats, num_classes))
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)

    def forward(self, adj, x):
        h = torch.spmm(adj, x)       # Layer 1
        h = torch.mm(h, self.W1)
        h = F.relu(h)
        h = torch.spmm(adj, h)       # Layer 2
        h = torch.mm(h, self.W2)
        return h

# 6. TRAIN
model = GCN(features.shape[1], 16, 7)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("   - Training...")
for e in range(101):
    logits = model(adj, features)
    loss = F.cross_entropy(logits[train_mask], train_labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if e % 20 == 0:
        print(f"     Epoch {e}, Loss: {loss.item():.4f}")

# 7. PREDICT & SAVE (The Critical Step)
print("   - Generating Predictions...")
model.eval()
with torch.no_grad():
    logits = model(adj, features)
    test_logits = logits[len(train_df):] # Extract test nodes only
    test_preds = test_logits.argmax(1)

# Save strictly according to the manual: 'id' and 'y_pred'
submission = pd.DataFrame({
    'id': test_df['id'],
    'y_pred': test_preds.numpy()  # <--- UPDATED: Uses 'y_pred' per Source 26
})

submission.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Saved submission to {OUTPUT_PATH}")
print("   (Columns: id, y_pred)")
