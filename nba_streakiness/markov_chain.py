from nba_api.stats.endpoints import PlayerGameLog as pgl
import numpy as np
import pandas as pd

def points_per_minute(p_id: str, season_list: list[str]) -> list[float]:
    ppm_list = []
    for season in season_list:
        game_log = pgl(player_id=p_id, season=season).get_data_frames()[0]
        game_log = game_log[game_log['MIN'] > 0]
        game_log['PPM'] = game_log['PTS'] / game_log['MIN']
        ppm_list.append(game_log['PPM'])
    ppm_list = pd.concat(ppm_list, ignore_index=True)
    return ppm_list.round(3).tolist()

p_id = '1629027'

seasons = ['2019-20','2020-21','2021-22','2022-23']

ppm_list = points_per_minute(p_id, seasons)

ppm = pd.Series(ppm_list, name='PPM')
ppm.index = np.arange(1, len(ppm_list) + 1)

cold, hot = np.percentile(ppm, [20, 80])
print(f"Cold ≤ {cold:.2f} ppm, Hot ≥ {hot:.2f} ppm")

def categorize(x):
    if x <= cold:
        return 0  # cold
    elif x >= hot:
        return 2  # hot
    else:
        return 1  # typical

states = ppm.apply(categorize)


K = 3
counts = np.zeros((K, K), dtype=int)

for prev, curr in zip(states[:-1], states[1:]):
    counts[prev, curr] += 1

P = counts / counts.sum(axis=1, keepdims=True)

P_df = pd.DataFrame(P, index=["cold","typical","hot"], columns=["cold","typical","hot"])
print(P_df)