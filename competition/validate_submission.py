import pandas as pd
import sys
import os

def main(submission_path, test_nodes_path, leaderboard_path):
    # Extract Team Name from Filename (e.g., 'TeamA.csv' -> 'TeamA')
    team_name = os.path.splitext(os.path.basename(submission_path))[0]
    print(f"🔍 Validating submission for Team: {team_name}")

    # --- 1. SUPERVISOR RULE: ONE SUBMISSION ONLY ---
    # Check if this team is already on the leaderboard
    if os.path.exists(leaderboard_path):
        try:
            lb_df = pd.read_csv(leaderboard_path)
            # Normalize to lowercase to prevent "TeamA" vs "teama" exploits
            existing_teams = set(lb_df['team'].astype(str).str.lower())
            
            if team_name.lower() in existing_teams:
                print(f"❌ REJECTED: Team '{team_name}' has already submitted.")
                print("   Per competition rules, only one submission is allowed.")
                sys.exit(1)
        except Exception as e:
            print(f"⚠️ Warning: Could not read leaderboard (Skipping check): {e}")

    # --- 2. LOAD DATA ---
    try:
        preds = pd.read_csv(submission_path)
        test_nodes = pd.read_csv(test_nodes_path)
    except Exception as e:
        print(f"❌ Error: Could not read files. {e}")
        sys.exit(1)

    # --- 3. COLUMN CHECKS ---
    required_cols = {'id', 'label'}
    if not required_cols.issubset(preds.columns):
        print(f"❌ Error: CSV headers must be 'id,label'. Found: {list(preds.columns)}")
        sys.exit(1)

    # --- 4. ID MATCHING ---
    if set(preds['id']) != set(test_nodes['id']):
        print(f"❌ Error: Submission IDs do not match the test set.")
        sys.exit(1)

    # --- 5. DATA TYPE & RANGE CHECKS ---
    if not pd.api.types.is_integer_dtype(preds['label']):
        print("❌ Error: 'label' column must contain integers.")
        sys.exit(1)

    valid_classes = {0, 1, 2, 3}
    if not preds['label'].isin(valid_classes).all():
        print(f"❌ Error: Found invalid class labels. Allowed: {valid_classes}")
        sys.exit(1)

    print(f"✅ VALID SUBMISSION for Team: {team_name}")

if __name__ == "__main__":
    # Now takes 3 arguments
    if len(sys.argv) != 4:
        print("Usage: python validate_submission.py <submission_csv> <test_nodes_csv> <leaderboard_csv>")
        sys.exit(1)
        
    main(sys.argv[1], sys.argv[2], sys.argv[3])
