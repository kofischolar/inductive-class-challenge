import pandas as pd
import sys
import os
import json
from sklearn.metrics import f1_score

# Config
SUBMISSION_FOLDER = 'submissions'
TRUTH_FILE = 'data/test_labels_hidden.csv'
LEADERBOARD_FILE = 'LEADERBOARD.md'

def get_score(submission_path, truth_df):
    try:
        pred_df = pd.read_csv(submission_path)
        # Basic validation
        if len(pred_df) != len(truth_df): return 0.0
        pred_df = pred_df.sort_values('id').reset_index(drop=True)
        return f1_score(truth_df['label'], pred_df['label'], average='macro')
    except:
        return 0.0


def main(pred_path, test_nodes_path, metadata_path, leaderboard_path):
    # 1. Load Metadata (To get Team Name)
    try:
        with open(metadata_path, 'r') as f:
            meta = json.load(f)
        team_name = meta.get("team", "").strip()
        if not team_name:
            raise ValueError("Metadata must contain a 'team' name.")
    except Exception as e:
        raise ValueError(f"Could not read metadata.json: {e}")

    # 2. ENFORCE ONE SUBMISSION POLICY
    # Check if team is already in the leaderboard
    if os.path.exists(leaderboard_path):
        lb = pd.read_csv(leaderboard_path)
        # Check if team exists (case insensitive)
        existing_teams = set(lb['team'].str.lower())
        if team_name.lower() in existing_teams:
            print(f"❌ REJECTED: Team '{team_name}' has already submitted!")
            print("Policy: Only one submission attempt per participant is allowed.")
            sys.exit(1) # Exit with error code to stop the process

    # 3. Standard Validation (Same as before)
    preds = pd.read_csv(pred_path)
    test_nodes = pd.read_csv(test_nodes_path)

    if "id" not in preds.columns or "y_pred" not in preds.columns:
        raise ValueError("predictions.csv must contain 'id' and 'y_pred'")

    # Check for NaN or duplicates
    if preds["y_pred"].isna().any(): raise ValueError("NaN predictions found")
    if preds["id"].duplicated().any(): raise ValueError("Duplicate IDs found")

    # Check IDs match
    if set(preds["id"]) != set(test_nodes["id"]):
        raise ValueError("Prediction IDs do not match test_nodes.csv")

    print(f"✅ VALID SUBMISSION for Team: {team_name}")

if __name__ == "__main__":
    # Expects: [predictions.csv] [test_nodes.csv] [metadata.json] [leaderboard.csv]
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
