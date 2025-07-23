import os
import time
import random
import requests
import pandas as pd

# get box scores for every game of every player for SEASONS
SEASONS      = ["2024", "2025"]  # list of seasons for roster and logs
LEAGUE_ID    = "10"                          # "10" for wnba
THROTTLE_SEC = 1.1                           # seconds between requests (avoid 429 error)
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
    "SEASON_YEAR","PLAYER_ID","PLAYER_NAME","TEAM_ABBREVIATION",
    "GAME_ID","GAME_DATE","MATCHUP","WL","MIN","FGM","FGA","FG_PCT",
    "FG3M","FG3A","FG3_PCT","FTM","FTA","FT_PCT","OREB","DREB","REB",
    "AST","TOV","STL","BLK","BLKA","PF","PFD","PTS","PLUS_MINUS"
]

# get all player ids
def fetch_roster(session, season: str) -> list[str]:
    """Return list of players active in a given season."""
    url = BASE_URL + "/stats/commonallplayers"
    params = {
        "LeagueID":            LEAGUE_ID,
        "Season":              season,
        "IsOnlyCurrentSeason": "0",
    }
    resp = session.get(url, params=params)
    resp.raise_for_status()
    result = resp.json()["resultSets"][0]
    df = pd.DataFrame(result["rowSet"], columns=result["headers"])
    season_int = int(season)
    df_active = df[(df["FROM_YEAR"].astype(int) <= season_int) &
                   (df["TO_YEAR"].astype(int) >= season_int)]
    return df_active["PERSON_ID"].astype(str).unique().tolist()

# individual player data call given player id
def one_player_call(session, player_id: str, season: str) -> pd.DataFrame:
    """Acquire one player's game logs for a given season and return a df."""
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
    session.headers["Referer"] = f"{BASE_URL}/player/{player_id}/boxscores-traditional/"
    resp = session.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()["resultSets"][0]
    df = pd.DataFrame(data["rowSet"], columns=data["headers"])

    keep = [c for c in DESIRED_COLS if c in df.columns]
    df_small = df[keep].copy()
    if "GAME_DATE" in df_small.columns:
        df_small["GAME_DATE"] = pd.to_datetime(df_small["GAME_DATE"])
    df_small["SEASON_YEAR"] = season
    return df_small

# main
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    session = requests.Session()
    session.headers.update(HEADERS)
    session.get(BASE_URL + "/")  # seed cookies / Cloudflare tokens

    master_dfs = []
    for season in SEASONS:
        print(f"Fetching season {season} roster…")
        roster = fetch_roster(session, season)
        print(f"→ {len(roster)} players found for {season}")
        for idx, pid in enumerate(roster, start=1):
            print(f"[{idx}/{len(roster)}] {season} | PlayerID={pid}", end="")
            try:
                dfp = one_player_call(session, pid, season)
                master_dfs.append(dfp)
                path = os.path.join(OUT_DIR, f"player_{pid}_logs_{season}.parquet")
                dfp.to_parquet(path, index=False)
                print(f" → {dfp.shape[0]} rows, saved to {path}")
            except Exception as e:
                print(f" ERROR: {e}")
            time.sleep(THROTTLE_SEC + random.random() * 0.3)

    if not master_dfs:
        print("No data.")
        exit(1)

    master_df = pd.concat(master_dfs, ignore_index=True)
    out_parquet = f"wnba_all_players_{'_'.join(SEASONS)}.parquet"
    out_csv     = f"wnba_all_players_{'_'.join(SEASONS)}.csv"
    master_df.to_parquet(out_parquet, index=False)
    master_df.to_csv(out_csv, index=False)
    print(f"\nCombined rows: {master_df.shape[0]}. Saved {out_parquet} and {out_csv}.")