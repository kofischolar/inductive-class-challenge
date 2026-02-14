import pandas as pd
import sys
import os
import json

def main(pred_path, test_nodes_path, metadata_path, leaderboard_path):
    # --- 1. METADATA & POLICY CHECKS ---
    try:
        with open(metadata_path, 'r') as f:
            meta = json.load(f)
        team_name = meta.get("team", "").strip()
        if not team_name:
            raise ValueError("Metadata must contain a 'team' name.")
    except Exception as e:
        raise ValueError(f"Could not read metadata.json: {e}")

    # Check One-Submission Policy
    if os.path.exists(leaderboard_path):
        lb = pd.read_csv(leaderboard_path)
        existing_teams = set(lb['team'].str.lower())
        if team_name.lower() in existing_teams:
            print(f"❌ REJECTED: Team '{team_name}' has already submitted!")
            sys.exit(1)

    # --- 2. DATA FORMAT CHECKS ---
    preds = pd.read_csv(pred_path)
    test_nodes = pd.read_csv(test_nodes_path)

    # Column existence
    if "id" not in preds.columns or "y_pred" not in preds.columns:
        raise ValueError("predictions.csv must contain 'id' and 'y_pred'")

    # ID Matching
    if set(preds["id"]) != set(test_nodes["id"]):
        raise ValueError("Prediction IDs do not match test_nodes.csv")

    # --- 3. MULTI-CLASS CHECKS (The Update) ---
    # Must be Integers
    if not pd.api.types.is_integer_dtype(preds["y_pred"]):
         raise ValueError("Predictions must be integers (0, 1, 2, 3), not decimals.")
         
    # Must be within 0-3 range (Tumor, Stromal, Lymphocyte, Macrophage)
    if ((preds["y_pred"] < 0) | (preds["y_pred"] > 3)).any():
        raise ValueError("Predictions must be class IDs between 0 and 3.")

    print(f"✅ VALID SUBMISSION for Team: {team_name}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
