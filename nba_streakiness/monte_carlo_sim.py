import numpy as np
from nba_api.stats.endpoints import PlayerGameLog as pgl
import pandas as pd
import statistics

def points_per_minute(p_id: str, season_list: list[str]) -> list[float]:
    """Creates a list of points per minute for a specific player and list of seasons."""
    ppm_list = []
    for season in season_list:
        game_log = pgl(player_id=p_id, season=season).get_data_frames()[0]
        game_log = game_log[game_log['MIN'] > 0]
        game_log['PPM'] = game_log['PTS'] / game_log['MIN']
        ppm_list.append(game_log['PPM'])
    ppm_list = pd.concat(ppm_list, ignore_index=True)
    return ppm_list.round(3).tolist()

## Modify function parameter defaults to change definitin of hot streak
def count_hot_streaks(ppm_list: list[float], threshold: float = 1.0, window: int = 3) -> int:       ### <---- Parameters
    """Count how many sub-arrays of length 'window' have average >= threshold."""
    count = 0
    for i in range(len(ppm_list) - window + 1):
        if np.mean(ppm_list[i : i + window]) >= threshold:
            count += 1
    return count

## Choose player and seasons to run the simulation for
p_id = '1629027'                                                   ### <---- Parameter
seasons = ['2019-20','2020-21','2021-22','2022-23']             ### <---- Parameter
player_ppm = points_per_minute(p_id, seasons)
hot_streaks = count_hot_streaks(player_ppm)

## Monte Carlo parameters
n_games  = len(player_ppm)
n_sims   = 10_000                           ### <---- Parameter

## Bootstrap MC simulation
bootstrap_sim_counts = np.zeros(n_sims, dtype=int)
for i in range(n_sims):
    sim_ppm = np.random.choice(player_ppm, size=n_games, replace=True)
    bootstrap_sim_counts[i] = count_hot_streaks(sim_ppm)

## Summarize bootstrap simulation
bootstrap_mean_streaks = bootstrap_sim_counts.mean()
bootstrap_std_streaks  = bootstrap_sim_counts.std(ddof=1)

## Classic MC simulation
mean = statistics.mean(player_ppm)
stdev = statistics.stdev(player_ppm)
classic_sim_counts = np.zeros(n_sims, dtype=int)
for i in range(n_sims):
    sim_ppm = np.random.normal(mean, stdev, size=n_games)
    classic_sim_counts[i] = count_hot_streaks(sim_ppm)

## Summarize classic simulation
classic_mean_streaks = classic_sim_counts.mean()
classic_std_streaks  = classic_sim_counts.std(ddof=1)

print("\n------------------------Historical Player Data-------------------------\n")
print(f"Number of hot-streaks: {hot_streaks}\n")
print(f"Over {n_sims} sims of {n_games} games:\n")
print("--------------------Bootstrap Monte Carlo Simulation-------------------\n")
print(f"Average number of hot-streaks: {bootstrap_mean_streaks:.2f}")
print(f"Standard deviation of hot-streaks: {bootstrap_std_streaks:.2f}\n")
print("---------------------Classic Monte Carlo Simulation--------------------\n")
print(f"Average number of hot-streaks: {classic_mean_streaks:.2f}")
print(f"Standard deviation of hot-streaks: {classic_std_streaks:.2f}\n")
print("-----------------------------------------------------------------------\n")