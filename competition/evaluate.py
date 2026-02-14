import pandas as pd
from sklearn.metrics import f1_score
import sys
import os

# usage: python evaluate.py <submission_file> <truth_file>

def main():
    if len(sys.argv) < 3:
        print("❌ Error: Missing arguments. Usage: python evaluate.py <pred_file> <truth_file>")
        sys.exit(1)

    submission_path = sys.argv[1]
    ground_truth_path = sys.argv[2] 

    print(f"📂 Grading submission: {submission_path}")
    print(f"📂 Against truth: {ground_truth_path}")

    try:
        # 1. Load Data
        pred_df = pd.read_csv(submission_path)
        true_df = pd.read_csv(ground_truth_path)

        # 2. Validation (Double check)
        # Note: validate_submission.py already checks this, but safety first!
        if 'id' not in pred_df.columns or 'y_pred' not in pred_df.columns:
            print("❌ Error: Submission must have 'id' and 'y_pred' columns.")
            sys.exit(1)

        # Ensure IDs match exactly (Sort both to align them)
        pred_df = pred_df.sort_values('id').reset_index(drop=True)
        true_df = true_df.sort_values('id').reset_index(drop=True)

        if not pred_df['id'].equals(true_df['id']):
            print("❌ Error: Submission IDs do not match the test set IDs exactly.")
            sys.exit(1)

        # 3. Calculate Score (Macro-F1)
        score = f1_score(true_df['label'], pred_df['y_pred'], average='macro')

        # 4. Output for GitHub Actions (CRITICAL CHANGE)
        # We changed "F1_SCORE:" to "SCORE=" to match the new scoring.yml grep command
        print(f"SCORE={score:.4f}")

    except Exception as e:
        print(f"❌ Grading Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
