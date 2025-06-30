import requests
import pandas as pd

# 1) Endpoint URL
url = "https://stats.wnba.com/stats/playergamelogs"

# 2) Every query parameter exactly as shown in DevTools
params = {
    "DateFrom":       "",
    "DateTo":         "",
    "GameSegment":    "",
    "LastNGames":     "0",
    "LeagueID":       "10",                 # WNBA
    "Location":       "",
    "MeasureType":    "Base",
    "Month":          "0",
    "OpponentTeamID": "0",
    "Outcome":        "",
    "PORound":        "0",
    "PaceAdjust":     "N",
    "PerMode":        "Totals",             # or "PerGame", etc.
    "Period":         "0",
    "PlayerID":       "1642777",            # change to the player you want
    "PlusMinus":      "N",
    "Rank":           "N",
    "Season":         "2025",               # change to season you want
    "SeasonSegment":  "",
    "SeasonType":     "Regular Season",
    "ShotClockRange": "",
    "VsConference":   "",
    "VsDivision":     ""
}

# 2.5) Delete unwanted parameters
    # del params["DateFrom"]
    # del params["ShotClockRange"]

# 3) Minimal headers needed
headers = {
    "Accept":             "application/json, text/plain, */*",
    "Origin":             "https://stats.wnba.com",
    "Referer":            "https://stats.wnba.com/player/1629483/boxscores-traditional/",
    "User-Agent":         "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token":  "true"
}

# 4) Fetch and parse
resp = requests.get(url, params=params, headers=headers)
resp.raise_for_status()
data = resp.json()

# 5) Normalize into a DataFrame
rs = data["resultSets"][0]
df = pd.DataFrame(rs["rowSet"], columns=rs["headers"])

# 5.5) Keep only necessary fields
keep = ["SEASON_YEAR","PLAYER_ID","PLAYER_NAME","TEAM_ABBREVIATION","GAME_ID","GAME_DATE","MATCHUP","WL","MIN","FGM","FGA","FG_PCT","FG3M","FG3A","FG3_PCT","FTM","FTA","FT_PCT","OREB","DREB","REB","AST","TOV","STL","BLK","BLKA","PF","PFD","PTS","PLUS_MINUS"]
df_small = df[keep]

# 6) Tidy up dates if present
if "GAME_DATE" in df_small.columns:
    df_small["GAME_DATE"] = pd.to_datetime(df_small["GAME_DATE"])

# 7) Inspect
print(df_small.shape)
print(df_small.head())

# 8) (Optional) Save for reuse
df_small.to_parquet(f"player_{params['PlayerID']}_logs_{params['Season']}.parquet", index=False)