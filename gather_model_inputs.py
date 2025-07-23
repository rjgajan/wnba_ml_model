import pandas as pd
from datetime import datetime

# 🔧 EDIT HERE
PLAYER_NAME = "A'ja Wilson"       # Full name as in PLAYER_NAME column
OPPONENT_ABBR = "ATL"             # Opponent abbreviation (e.g., "NYL")

# 📄 Load and clean data
df = pd.read_csv("processed_2024_2025.csv")
df['game_date'] = pd.to_datetime(df['GAME_DATE'])  # Make sure dates are datetime

# 🎯 Filter to games by this player, sorted by most recent
player_df = df[df['PLAYER_NAME'] == PLAYER_NAME].sort_values('game_date', ascending=False)

if player_df.empty:
    print(f"No games found for player: {PLAYER_NAME}")
    exit()

# ⏳ Days since last game
last_game_date = player_df.iloc[0]['game_date']
days_since_last_game = (pd.Timestamp.today() - last_game_date).days

# 📊 Compute averages
avg_last_15 = player_df.head(15)['PTS'].mean()
avg_last_5 = player_df.head(5)['PTS'].mean()

# 🎯 Games vs specific opponent
vs_opp_df = player_df[player_df['OPP'] == OPPONENT_ABBR].head(3)
avg_last_3_vs_opp = vs_opp_df['PTS'].mean() if not vs_opp_df.empty else None

# 🖨️ Display results
print(f"\n📊 Stats for {PLAYER_NAME} vs {OPPONENT_ABBR}:")
print(f"- Last 15 games avg: {avg_last_15:.2f} PTS")
print(f"- Last 5 games avg:  {avg_last_5:.2f} PTS")
if avg_last_3_vs_opp is not None:
    print(f"- Last 3 games vs {OPPONENT_ABBR}: {avg_last_3_vs_opp:.2f} PTS")
else:
    print(f"- No recent games found vs {OPPONENT_ABBR}")
print(f"- Days since last game: {days_since_last_game} days\n")