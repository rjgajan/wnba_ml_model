# Monte Carlo Simulation

from nba_api.stats.endpoints import PlayerGameLog as pgl
import pandas as pd
import matplotlib.pyplot as plt

## Function that imports a player's points and minutes for an array of seasons to create a list of points/minute
def points_per_minute(p_id: str, season_list: list[str]) -> list[float]:
    ppm_list = []
    for season in season_list:
        game_log = pgl(player_id=p_id, season=season).get_data_frames()[0]
        game_log = game_log[game_log['MIN'] > 0]
        game_log['PPM'] = game_log['PTS'] / game_log['MIN']
        ppm_list.append(game_log['PPM'])
    ppm_list = pd.concat(ppm_list, ignore_index=True)
    return ppm_list.round(3).tolist()

## Lebron James 2nd time w/ Cavs
p_id1 = '2544'
seasons1 = ['2014-15','2015-16','2016-17','2017-18']

## Prime Tim Duncan
p_id2 = '1495'
seasons2 = ['2000-01','2001-02','2002-03','2003-04']

## Trae Young Villain Era
p_id3 = '1629027'
seasons3 = ['2019-20','2020-21','2021-22','2022-23']

lebron_ppm = points_per_minute(p_id1, seasons1)
tim_ppm = points_per_minute(p_id2, seasons2)
trae_ppm = points_per_minute(p_id3, seasons3)

## Visualize ppm distributions
if __name__ == "__main__":
    ppms = [lebron_ppm, tim_ppm, trae_ppm]
    for ppm in ppms:
        data = ppm
        plt.figure()
        plt.hist(data, bins=30, edgecolor='black', alpha=0.7, density=False)
        plt.title(f"{data}")
        plt.xlabel("ppm")
        plt.ylabel("count")
    plt.show()