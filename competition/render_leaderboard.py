import pandas as pd
import os

LEADERBOARD_CSV = "leaderboard/leaderboard.csv"
# FIXED: Points to the root file so the README link works
LEADERBOARD_MD = "LEADERBOARD.md"

def render():
    if not os.path.exists(LEADERBOARD_CSV):
        print("❌ No leaderboard file found at:", LEADERBOARD_CSV)
        return

    # 1. Load Data
    df = pd.read_csv(LEADERBOARD_CSV)
    
    # 2. Sort by Score (Descending for F1/Accuracy)
    df = df.sort_values(by="score", ascending=False)

    # 3. Assign Ranks (Handling Ties)
    # method='dense' gives 1, 2, 2, 3
    df['rank'] = df['score'].rank(method='dense', ascending=False).astype(int)

    # 4. Format the Table
    def format_rank(r):
        if r == 1: return "🥇 1"
        if r == 2: return "🥈 2"
        if r == 3: return "🥉 3"
        return str(r)
    
    df['rank_display'] = df['rank'].apply(format_rank)
    
    # Create Markdown Content
    md_lines = []
    md_lines.append("# 🏆 Tumor Diagnosis Leaderboard")
    md_lines.append("")
    md_lines.append(f"**Last Updated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    md_lines.append("")
    md_lines.append("| Rank | Team | Macro F1 Score | Date |")
    md_lines.append("| :--- | :--- | :--- | :--- |")
    
    for _, row in df.iterrows():
        # Clean team name to prevent markdown injection
        team = str(row['team']).replace("|", "")
        # Format score to 4 decimal places
        md_lines.append(f"| {row['rank_display']} | {team} | {row['score']:.4f} | {row['date']} |")

    # 5. Save to Root Directory
    with open(LEADERBOARD_MD, "w") as f:
        f.write("\n".join(md_lines))
    
    print(f"✅ Leaderboard Markdown updated at {LEADERBOARD_MD}")

if __name__ == "__main__":
    render()
