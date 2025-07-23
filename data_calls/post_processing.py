"""
Script: wnba_csv_postprocess.py

Post-processes a WNBA game-logs CSV: filters by games played and computes days since last game.
Configure input, output file names and thresholds in the CONFIG section below.
"""
import pandas as pd

# === CONFIGURATION ===
INPUT_CSV    = "wnba_all_players_2024_2025.csv"  # Path to your combined logs CSV
OUTPUT_CSV   = "processed_2024_2025.csv"         # Path to save the filtered CSV
MIN_GAMES    = 15                                 # Minimum number of games to include a player
DEFAULT_DAYS = 5                                  # Default days_since_last_game for first game
DROP_COLUMNS = ["GAME_ID", "FGM", "FGA", "FG_PCT", 
                "FG3M", "FG3A", "FG3_PCT", "FTM", 
                "FTA", "FT_PCT", "OREB", "DREB",
                "REB", "AST", "TOV", "STL", "BLK",
                "BLKA", "PF", "PFD", "PLUS_MINUS"]                 # List of column names to remove from final output

# === SCRIPT ===
# Load the combined logs
print(f"Loading data from {INPUT_CSV}...")
df = pd.read_csv(INPUT_CSV, parse_dates=["GAME_DATE"])

# Filter players by games played
print(f"Filtering players with at least {MIN_GAMES} games...")
game_counts = df.groupby("PLAYER_ID").size()
valid_players = game_counts[game_counts >= MIN_GAMES].index
filtered = df[df["PLAYER_ID"].isin(valid_players)].copy()

# Sort by player and date
filtered.sort_values(["PLAYER_ID", "GAME_DATE"], inplace=True)

# Compute days since last game
print("Computing days_since_last_game...")
filtered["days_since_last_game"] = (
    filtered
    .groupby("PLAYER_ID")["GAME_DATE"]
    .diff()
    .dt.days
    .fillna(DEFAULT_DAYS)
    .astype(int)
)

# Compute rolling average points for previous games (excludes current)
print("Computing rolling average PTS for last 5 and 15 games...")
filtered["avg_prev_5"] = filtered.groupby("PLAYER_ID")["PTS"].transform(
    lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
)
filtered["avg_prev_15"] = filtered.groupby("PLAYER_ID")["PTS"].transform(
    lambda x: x.shift(1).rolling(window=15, min_periods=1).mean()
)

# Derive home/away, opponent code, and map defensive ratings
if "MATCHUP" in filtered.columns:
    print("Deriving HOME_AWAY, OPP, and opponent defensive ratings...")
    # Home/Away
    filtered["HOME_AWAY"] = filtered["MATCHUP"].apply(
        lambda m: "Away" if "@" in m else "Home"
    )
    # Opponent code
    filtered["OPP"] = filtered["MATCHUP"].apply(lambda m: m.split()[-1])
    # Defensive ratings lookup
    def_ratings = {
        "ATL": 99.4, "CHI": 108.3, "CON": 112.2, "DAL": 107.7, "IND": 100.6,
        "GSV": 98.8, "LVA": 102.8, "LAS": 107.5, "MIN": 94.8,  "NYL": 97.7,
        "PHO": 98.5, "SEA": 98.9,  "WSH": 100.1
    }
    filtered["OPP_DEF_RATING"] = filtered["OPP"].map(def_ratings)
    # Rolling average vs same opponent
    print("Computing avg_prev_opp_3 (last 3 games vs same opponent)...")
    filtered["avg_prev_opp_3"] = filtered.groupby(["PLAYER_ID", "OPP"])["PTS"].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    )
else:
    print("MATCHUP column not found; skipping HOME_AWAY and opponent stats")

# Drop unwanted columns
if DROP_COLUMNS:
    print(f"Dropping columns: {DROP_COLUMNS}...")
    filtered.drop(columns=DROP_COLUMNS, inplace=True, errors='ignore')

# Round numeric metrics to 5 decimal places
print("Rounding numeric metrics to 5 decimal places...")
# Identify float columns and round
float_cols = filtered.select_dtypes(include="number").columns
filtered[float_cols] = filtered[float_cols].round(5)

# Save the processed CSV
print(f"Saving processed data to {OUTPUT_CSV}...")
filtered.to_csv(OUTPUT_CSV, index=False)
print("Done.")