import pandas as pd
import os
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

def main():
    # 1. Load Truth
    if not os.path.exists(TRUTH_FILE):
        print("Error: Truth file not found")
        return
    truth_df = pd.read_csv(TRUTH_FILE).sort_values('id').reset_index(drop=True)

    # 2. Score Every Submission
    results = []
    for filename in os.listdir(SUBMISSION_FOLDER):
        if filename.endswith(".csv") and filename != "sample_submission.csv":
            path = os.path.join(SUBMISSION_FOLDER, filename)
            score = get_score(path, truth_df)
            
            # Clean filename to get username (e.g., "john_doe.csv" -> "john_doe")
            user = filename.replace('.csv', '')
            results.append({'User': user, 'Score': score})

    # 3. Sort by Score (Highest first)
    results.sort(key=lambda x: x['Score'], reverse=True)

    # 4. Generate Markdown Table
    md = "# 🏆 GNN Challenge Leaderboard\n\n"
    md += "| Rank | Participant | Macro F1 Score |\n"
    md += "| :--- | :--- | :--- |\n"
    
    for rank, entry in enumerate(results, 1):
        # Add a medal emoji for top 3
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}"
        md += f"| {medal} | {entry['User']} | {entry['Score']:.4f} |\n"
    
    md += "\n*Updated automatically by GitHub Actions*"

    # 5. Save
    with open(LEADERBOARD_FILE, "w", encoding="utf-8") as f:
        f.write(md)
    print("✅ Leaderboard updated!")

if __name__ == "__main__":
    main()
