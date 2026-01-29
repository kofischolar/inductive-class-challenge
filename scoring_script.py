import pandas as pd
from sklearn.metrics import f1_score
import sys
import os

# usage: python scoring_script.py <submission_file> <truth_file>
submission_path = sys.argv[1]
ground_truth_path = sys.argv[2] # We pass the path explicitly

print(f"📂 Grading submission: {submission_path}")
print(f"📂 Against truth: {ground_truth_path}")

try:
    # 1. Load Data
    pred_df = pd.read_csv(submission_path)
    true_df = pd.read_csv(ground_truth_path)

    # 2. Validation
    # Ensure columns exist
    if 'id' not in pred_df.columns or 'label' not in pred_df.columns:
        print("❌ Error: Submission must have 'id' and 'label' columns.")
        sys.exit(1)

    # Ensure IDs match (Sort both to be safe)
    pred_df = pred_df.sort_values('id').reset_index(drop=True)
    true_df = true_df.sort_values('id').reset_index(drop=True)

    # Check if we have the same number of rows
    if len(pred_df) != len(true_df):
        print(f"❌ Error: Submission has {len(pred_df)} rows. Expected {len(true_df)}.")
        sys.exit(1)

    # 3. Calculate Score
    score = f1_score(true_df['label'], pred_df['label'], average='macro')

    # 4. Output for GitHub Actions
    # We print a specific tag that the bot looks for
    print(f"F1_SCORE:{score:.4f}")

except Exception as e:
    print(f"❌ Grading Failed: {e}")
    sys.exit(1)
