import pandas as pd
import os

LEADERBOARD_CSV = "leaderboard/leaderboard.csv"
LEADERBOARD_MD = "leaderboard/leaderboard.md"

def render():
    if not os.path.exists(LEADERBOARD_CSV):
        print("No leaderboard file found.")
        return

    # 1. Load Data
    df = pd.read_csv(LEADERBOARD_CSV)
    
    # 2. Sort by Score (Descending for F1/Accuracy)
    # If using Loss (lower is better), change ascending=True
    df = df.sort_values(by="score", ascending=False)

    # 3. Assign Ranks (Handling Ties)
    # method='min' gives 1, 2, 2, 4 (Standard competition ranking)
    # method='dense' gives 1, 2, 2, 3 (Your requirement: "tied scores share rank")
    df['rank'] = df['score'].rank(method='dense', ascending=False).astype(int)

    # 4. Format the Table
    # Reorder columns
    display_cols = ['rank', 'team', 'score', 'date']
    # Add emojis to top 3
    def format_rank(r):
        if r == 1: return "🥇 1"
        if r == 2: return "🥈 2"
        if r == 3: return "🥉 3"
        return str(r)
    
    df['rank_display'] = df['rank'].apply(format_rank)
    
    # Create Markdown
    md_lines = []
    md_lines.append("# 🏆 Competition Leaderboard")
    md_lines.append(f"**Last Updated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    md_lines.append("")
    md_lines.append("| Rank | Team | Score (Macro-F1) | Date |")
    md_lines.append("| :--- | :--- | :--- | :--- |")
    
    for _, row in df.iterrows():
        # Clean team name to prevent markdown injection
        team = str(row['team']).replace("|", "")
        md_lines.append(f"| {row['rank_display']} | {team} | {row['score']:.4f} | {row['date']} |")

    # 5. Save
    with open(LEADERBOARD_MD, "w") as f:
        f.write("\n".join(md_lines))
    
    print("✅ Leaderboard Markdown updated.")

if __name__ == "__main__":
    render()
