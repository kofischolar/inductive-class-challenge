import pandas as pd
import os
import glob
from sklearn.metrics import f1_score

# CONFIG
SUBMISSION_DIR = "submissions"
TRUTH_FILE = "data/test_labels_hidden.csv"
LEADERBOARD_CSV = "leaderboard/leaderboard.csv"
LEADERBOARD_MD = "leaderboard/leaderboard.md"

def main():
    # 1. Load Truth
    if not os.path.exists(TRUTH_FILE):
        print(f"❌ Error: Truth file {TRUTH_FILE} not found")
        return
    
    true_df = pd.read_csv(TRUTH_FILE).sort_values('id').reset_index(drop=True)
    
    # 2. Find All Submissions
    # Recursive search for any csv file in submissions/
    files = glob.glob(f"{SUBMISSION_DIR}/**/*.csv", recursive=True)
    results = []

    print(f"🔍 Found {len(files)} submission files.")

    for file_path in files:
        # Skip the example file
        if "sample_submission.csv" in file_path:
            continue
            
        try:
            # Load Submission
            pred_df = pd.read_csv(file_path)
            
            # Validation: Columns
            if 'id' not in pred_df.columns or 'y_pred' not in pred_df.columns:
                print(f"⚠️ Skipping {file_path}: Missing columns")
                continue 
            
            # Validation: IDs match
            pred_df = pred_df.sort_values('id').reset_index(drop=True)
            if not pred_df['id'].equals(true_df['id']):
                print(f"⚠️ Skipping {file_path}: ID mismatch")
                continue

            # Score (Macro F1)
            score = f1_score(true_df['label'], pred_df['y_pred'], average='macro')
            
            # Extract Team Name from folder structure
            # Structure: submissions/inbox/TEAM_NAME/run_01/predictions.csv
            parts = file_path.split(os.sep)
            # We assume the folder directly inside 'submissions' (or 'inbox') is the team name
            # Adjust index based on your exact folder depth. 
            # If path is submissions/inbox/teamA/..., team is parts[2]
            if "inbox" in parts:
                idx = parts.index("inbox") + 1
                if idx < len(parts):
                    team_name = parts[idx]
                else:
                    team_name = "Unknown"
            else:
                # Fallback: parent folder name
                team_name = os.path.basename(os.path.dirname(file_path))

            results.append({
                'team': team_name,
                'score': score,
                'date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')
            })

        except Exception as e:
            print(f"⚠️ Failed to process {file_path}: {e}")

    # 3. Save Leaderboard CSV
    if not results:
        print("No valid submissions found.")
        return

    df = pd.DataFrame(results)
    
    # Sort by Score (Descending) and keep best score per team
    df = df.sort_values(by='score', ascending=False)
    df = df.drop_duplicates(subset=['team'], keep='first')
    
    os.makedirs("leaderboard", exist_ok=True)
    df.to_csv(LEADERBOARD_CSV, index=False)
    
    # 4. Render Markdown
    render_markdown(df)
    print("✅ Leaderboard Updated!")

def render_markdown(df):
    md = "# 🏆 Tumor-GNN Leaderboard\n\n"
    md += "| Rank | Team | Macro-F1 Score | Last Updated |\n"
    md += "| :--- | :--- | :--- | :--- |\n"
    
    # Add Rank with Dense logic (1, 2, 2, 3)
    df['rank'] = df['score'].rank(method='dense', ascending=False).astype(int)
    
    for _, row in df.iterrows():
        r = row['rank']
        medal = "🥇" if r == 1 else "🥈" if r == 2 else "🥉" if r == 3 else str(r)
        md += f"| {medal} | {row['team']} | {row['score']:.4f} | {row['date']} |\n"
        
    with open(LEADERBOARD_MD, "w") as f:
        f.write(md)

if __name__ == "__main__":
    main()
