import requests
import pandas as pd
import os
import time
import random

# === CONFIGURATION ===
SEASON       = "2025"
LEAGUE_ID    = "10"     # WNBA
THROTTLE_SEC = 1.1      # seconds between requests
OUT_DIR      = "player_parquets"

HEADERS = {
    "Accept":             "application/json, text/plain, */*",
    "Origin":             "https://stats.wnba.com",
    "Referer":            "https://stats.wnba.com",
    "User-Agent":         "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token":  "true"
}

DESIRED_COLS = [
    "SEASON_YEAR","PLAYER_ID","PLAYER_NAME","TEAM_ABBREVIATION",
    "GAME_ID","GAME_DATE","MATCHUP","WL","MIN","FGM","FGA","FG_PCT",
    "FG3M","FG3A","FG3_PCT","FTM","FTA","FT_PCT","OREB","DREB","REB",
    "AST","TOV","STL","BLK","BLKA","PF","PFD","PTS","PLUS_MINUS"
]

# === HELPERS ===

def fetch_roster(season=SEASON):
    """Return list of PERSON_IDs active in the season."""
    url = "https://stats.wnba.com/stats/commonallplayers"
    resp = requests.get(
        url,
        params={
            "LeagueID":           LEAGUE_ID,
            "Season":             season,
            "IsOnlyCurrentSeason":"1"
        },
        headers=HEADERS
    )
    resp.raise_for_status()
    data = resp.json()["resultSets"][0]
    df = pd.DataFrame(data["rowSet"], columns=data["headers"])
    return df["PERSON_ID"].astype(str).tolist()

def one_player_call(player_id: str, season=SEASON) -> pd.DataFrame:
    """Fetch one player’s game logs and return a slimmed DataFrame."""
    url = "https://stats.wnba.com/stats/playergamelogs"
    params = {
        "DateFrom":       "",
        "DateTo":         "",
        "GameSegment":    "",
        "LastNGames":     "0",
        "LeagueID":       LEAGUE_ID,
        "Location":       "",
        "MeasureType":    "Base",
        "Month":          "0",
        "OpponentTeamID": "0",
        "Outcome":        "",
        "PORound":        "0",
        "PaceAdjust":     "N",
        "PerMode":        "Totals",
        "Period":         "0",
        "PlayerID":       player_id,
        "PlusMinus":      "N",
        "Rank":           "N",
        "Season":         season,
        "SeasonSegment":  "",
        "SeasonType":     "Regular Season",
        "ShotClockRange": "",
        "VsConference":   "",
        "VsDivision":     ""
    }
    headers = HEADERS.copy()
    headers["Referer"] = f"https://stats.wnba.com/player/{player_id}/boxscores-traditional/"

    resp = requests.get(url, params=params, headers=headers)
    resp.raise_for_status()
    data = resp.json()["resultSets"][0]
    df = pd.DataFrame(data["rowSet"], columns=data["headers"])

    keep = [c for c in DESIRED_COLS if c in df.columns]
    df_small = df[keep].copy()
    if "GAME_DATE" in df_small.columns:
        df_small["GAME_DATE"] = pd.to_datetime(df_small["GAME_DATE"])
    return df_small

# === MAIN SCRIPT ===

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    roster = fetch_roster()
    all_dfs = []

    for idx, pid in enumerate(roster, start=1):
        print(f"[{idx}/{len(roster)}] Fetching PlayerID={pid}", end="")
        try:
            dfp = one_player_call(pid)
            all_dfs.append(dfp)
            # Save per-player Parquet
            path = os.path.join(OUT_DIR, f"player_{pid}_logs_{SEASON}.parquet")
            dfp.to_parquet(path, index=False)
            print(f" → {dfp.shape[0]} rows, saved to {path}")
        except Exception as e:
            print(f" ERROR: {e}")
        time.sleep(THROTTLE_SEC + random.random()*0.3)

    if not all_dfs:
        print("No data fetched; exiting.")
        exit(1)

    master_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nCombined total rows: {master_df.shape[0]}")

    master_df.to_parquet(f"wnba_all_players_{SEASON}.parquet", index=False)
    master_df.to_csv(f"wnba_all_players_{SEASON}.csv", index=False)
    print(f"Saved wnba_all_players_{SEASON}.parquet and .csv")