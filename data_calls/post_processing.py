"""
Post-processes a WNBA game-logs CSV: filters by games played and computes days since last game.
Configure input, output file names and thresholds in the CONFIG section below.
"""
import pandas as pd

INPUT_CSV    = "wnba_all_players_2024_2025.csv"  # Path to your combined logs CSV
OUTPUT_CSV   = "processed_2024_2025.csv"         # Path to save the filtered CSV
MIN_GAMES    = 15                                 # Minimum number of games to include a player
DEFAULT_DAYS = 5                                  # Default days_since_last_game for first game
DROP_COLUMNS = ["GAME_ID", "FGM", "FGA", "FG_PCT", 
                "FG3M", "FG3A", "FG3_PCT", "FTM", 
                "FTA", "FT_PCT", "OREB", "DREB",
                "REB", "AST", "TOV", "STL", "BLK",
                "BLKA", "PF", "PFD", "PLUS_MINUS"]      # List of column names to remove from final output

# Load the combined logs
df = pd.read_csv(INPUT_CSV, parse_dates=["GAME_DATE"])

# Filter players by games played
game_counts = df.groupby("PLAYER_ID").size()
valid_players = game_counts[game_counts >= MIN_GAMES].index
filtered = df[df["PLAYER_ID"].isin(valid_players)].copy()

# Sort by player and date
filtered.sort_values(["PLAYER_ID", "GAME_DATE"], inplace=True)

# Compute team_rest_days for each team-game
team_rest = (
    filtered.groupby(["GAME_ID", "TEAM_ABBREVIATION"], as_index=False)
    .agg({"GAME_DATE": "first"})  # one row per team per game
    .sort_values(["TEAM_ABBREVIATION", "GAME_DATE"])
)
team_rest["team_rest_days"] = (
    team_rest.groupby("TEAM_ABBREVIATION")["GAME_DATE"]
    .diff()
    .dt.days
    .fillna(DEFAULT_DAYS)
    .astype(int)
)
# Compute signed rest difference at team-game level
opponent_rest = (
    team_rest.merge(team_rest, on="GAME_ID", suffixes=("", "_opp"))
    .query("TEAM_ABBREVIATION != TEAM_ABBREVIATION_opp")
)

opponent_rest["rest_diff"] = (
    opponent_rest["team_rest_days"] - opponent_rest["team_rest_days_opp"]
)
# Keep only necessary columns
opponent_rest = opponent_rest[["GAME_ID", "TEAM_ABBREVIATION", "team_rest_days", "rest_diff"]]
# Merge with original df
filtered = filtered.merge(opponent_rest, on=["GAME_ID", "TEAM_ABBREVIATION"], how="left")

# Compute rolling average points for previous games (excludes current)
filtered["avg_prev_5"] = filtered.groupby("PLAYER_ID")["PTS"].transform(
    lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
)
filtered["avg_prev_15"] = filtered.groupby("PLAYER_ID")["PTS"].transform(
    lambda x: x.shift(1).rolling(window=15, min_periods=1).mean()
)

# Home/Away
filtered["HOME_AWAY"] = filtered["MATCHUP"].apply(
    lambda m: "Away" if "@" in m else "Home"
)
# Opponent code
filtered["OPP"] = filtered["MATCHUP"].apply(lambda m: m.split()[-1])
# Define defensive ratings by year
def_ratings_2024 = {
    "ATL": 99.4, "CHI": 108.3, "CON": 112.2, "DAL": 107.7, "IND": 100.6,
    "GSV": 98.8, "LVA": 102.8, "LAS": 107.5, "MIN": 94.8,  "NYL": 97.7,
    "PHO": 98.5, "SEA": 98.9,  "WSH": 100.1
}
def_ratings_2025 = {
    "ATL": 100.1, "CHI": 103.5, "CON": 94.1, "DAL": 111.7, "IND": 107.5,
    "LVA": 99.6, "LAS": 105.7, "MIN": 94.8, "NYL": 95.3,
    "PHO": 105.4, "SEA": 96.4, "WSH": 101.7
}
# Add ratings based on year
filtered["OPP_DEF_RATING"] = filtered.apply(
    lambda row: def_ratings_2024.get(row["OPP"]) if row["SEASON_YEAR"] == 2024
    else def_ratings_2025.get(row["OPP"]),
    axis=1
)
# Rolling average vs same opponent
filtered["avg_prev_opp_3"] = filtered.groupby(["PLAYER_ID", "OPP"])["PTS"].transform(
    lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
)

# Add pace metric (poss/40)
# Pace dictionaries
pace_2025 = {
    "ATL": 78.8, "CHI": 79.82, "CON": 78.87, "DAL": 80.21, "IND": 80.04,
    "GSV": 78.63, "LVA": 79.98, "LAS": 80.8, "MIN": 78.64, "NYL": 81.65,
    "PHO": 80.66, "SEA": 79.12, "WSH": 80.04
}
pace_2024 = {
    "ATL": 78.96, "CHI": 79.76, "CON": 77.76, "DAL": 81.71, "IND": 81.28,
    "LVA": 81.08, "LAS": 80.66, "MIN": 79.02, "NYL": 80.12,
    "PHO": 79.9, "SEA": 81.09, "WSH": 80.41
}
# Apply opponent pace based on season year
filtered["OPP_PACE"] = filtered.apply(
    lambda row: pace_2024.get(row["OPP"]) if row["SEASON_YEAR"] == 2024
    else pace_2025.get(row["OPP"]),
    axis=1
)

# Drop unwanted columns
if DROP_COLUMNS:
    filtered.drop(columns=DROP_COLUMNS, inplace=True, errors='ignore')

# Round to 5 decimal places
float_cols = filtered.select_dtypes(include="number").columns
filtered[float_cols] = filtered[float_cols].round(5)

# Save the processed CSV
print(f"Saving processed data to {OUTPUT_CSV}...")
filtered.to_csv(OUTPUT_CSV, index=False)
print("Done.")