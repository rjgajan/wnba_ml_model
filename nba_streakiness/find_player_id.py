from nba_api.stats.static import players

# 1. Pull the full list of players
all_players = players.get_players()

# 2. Find matches by full name (case-insensitive substring)
matches = players.find_players_by_full_name("Trae Young")
#    (this returns a list of dicts like {'id':2544, 'full_name':'LeBron James', ...})

if matches:
    player_id = matches[0]['id']
    print(f"Found {matches[0]['full_name']} → player_id = {player_id}")
else:
    print("No exact match; try a shorter or different name fragment.")