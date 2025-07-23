import os
import time
import random

import requests
import pandas as pd

# === CONFIGURATION ===
SEASON       = "2024"            # use a completed season for testing
LEAGUE_ID    = "10"              # WNBA
THROTTLE_SEC = 1.1               # seconds between requests
OUT_DIR      = "player_parquets"

BASE_URL = "https://stats.wnba.com"

HEADERS = {
    "Accept":             "application/json, text/plain, */*",
    "Accept-Language":    "en-US,en;q=0.9",
    "Accept-Encoding":    "gzip, deflate, br",
    "Connection":         "keep-alive",
    "Host":               "stats.wnba.com",
    "Origin":             BASE_URL,
    "Referer":            f"{BASE_URL}/",
    "User-Agent":         ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/138.0.0.0 Safari/537.36"),
    "Sec-Fetch-Site":     "same-origin",
    "Sec-Fetch-Mode":     "cors",
    "Sec-Fetch-Dest":     "empty",
    "sec-ch-ua":          '"Not)A;Brand";v="8", "Chromium";v="138", "Google Chrome";v="138"',
    "sec-ch-ua-mobile":   "?0",
    "sec-ch-ua-platform": '"macOS"',
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token":  "true",
}

DESIRED_COLS = [
    "SEASON_YEAR", "PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION",
    "GAME_ID", "GAME_DATE", "MATCHUP", "WL", "MIN", "FGM", "FGA", "FG_PCT",
    "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB",
    "AST", "TOV", "STL", "BLK", "BLKA", "PF", "PFD", "PTS", "PLUS_MINUS"
]

# === HELPERS ===

def fetch_roster(session, season=SEASON):
    """Return list of PLAYER_IDs who logged at least one game."""
    url = BASE_URL + "/stats/leaguedashplayerstats"
    params = {
        "LeagueID":   LEAGUE_ID,           # ← add this
        "Season":     season,
        "SeasonType": "Regular Season",
        "PerMode":    "Totals",
    }
    resp = session.get(url, params=params)
    # for debugging you can uncomment:
    # print(resp.status_code, resp.url, resp.text[:200])
    resp.raise_for_status()
    result = resp.json()["resultSets"][0]
    df = pd.DataFrame(result["rowSet"], columns=result["headers"])
    return df["PLAYER_ID"].astype(str).unique().tolist()

def one_player_call(session, player_id: str, season=SEASON) -> pd.DataFrame:
    """Fetch one player’s game logs and return a slimmed DataFrame."""
    url = BASE_URL + "/stats/playergamelogs"
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
    # use the session headers (including Referer)
    session.headers["Referer"] = f"{BASE_URL}/player/{player_id}/boxscores-traditional/"
    resp = session.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()["resultSets"][0]
    df = pd.DataFrame(data["rowSet"], columns=data["headers"])

    keep = [c for c in DESIRED_COLS if c in df.columns]
    df_small = df[keep].copy()
    if "GAME_DATE" in df_small:
        df_small["GAME_DATE"] = pd.to_datetime(df_small["GAME_DATE"])
    return df_small

# === MAIN SCRIPT ===

if __name__ == "__main__":
    # prepare output directory
    os.makedirs(OUT_DIR, exist_ok=True)

    # prime a session so cookies/cloudflare tokens are set
    session = requests.Session()
    session.headers.update(HEADERS)
    session.get(BASE_URL + "/")

    # fetch roster
    roster = fetch_roster(session)
    print(f"→ roster.size = {len(roster)}; sample = {roster[:5]}")

    all_dfs = []
    for idx, pid in enumerate(roster, start=1):
        print(f"[{idx}/{len(roster)}] Fetching PlayerID={pid}", end="")
        try:
            dfp = one_player_call(session, pid)
            all_dfs.append(dfp)
            path = os.path.join(OUT_DIR, f"player_{pid}_logs_{SEASON}.parquet")
            dfp.to_parquet(path, index=False)
            print(f" → {dfp.shape[0]} rows, saved to {path}")
        except Exception as e:
            print(f" ERROR: {e}")
        time.sleep(THROTTLE_SEC + random.random() * 0.3)

    if not all_dfs:
        print("No data fetched; exiting.")
        exit(1)

    master_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nCombined total rows: {master_df.shape[0]}")
    master_df.to_parquet(f"wnba_all_players_{SEASON}.parquet", index=False)
    master_df.to_csv(f"wnba_all_players_{SEASON}.csv", index=False)
    print(f"Saved wnba_all_players_{SEASON}.parquet and .csv")