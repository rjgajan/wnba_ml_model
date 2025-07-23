"""ARIMA (Autoregressive Integrated Moving Average) is a            GARCH (Generalized Autoregressive Conditional
statistical analysis model that uses time series data to            Heteroskedasticity) is a statistical model used to
predict future data points based on past data points.               predict the volatility of time series data."""

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

## 1.) Create series of length N where ppm[i] is the ppm from game i
ppm = pd.Series(lebron_ppm, name='PPM')
ppm.index = np.arange(1, len(lebron_ppm) + 1)

## 2.) Check stationarity (ADF test)
from statsmodels.tsa.stattools import adfuller

adf_stat, adf_pvalue, _, _, _, _ = adfuller(ppm)
print(f"ADF statistic = {adf_stat:.3f},  p-value = {adf_pvalue:.3f}")

## 3.) Identify ARIMA orders (p, d, q) by interpreting the following plots
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(8,6))
plot_acf(ppm, lags=15, ax=axes[0]) ### Autocorrelation function
plot_pacf(ppm, lags=15, ax=axes[1]) ### Partial autocorrelation function
axes[0].set_title("ACF of Differenced Series")
axes[1].set_title("PACF of Differenced Series")
plt.tight_layout()
plt.show()

p, d, q = 0, 0, 0

## 4.) Fit the ARIMA model
from statsmodels.tsa.arima.model import ARIMA

arima_mod = ARIMA(ppm, order=(p, d, q))
arima_res = arima_mod.fit()
print(arima_res.summary())

## 5.) Diagnose residuals
from statsmodels.stats.diagnostic import acorr_ljungbox

ljung = acorr_ljungbox(arima_res.resid, lags=[10], return_df=True)
print(ljung)

## 6.) Extract residuals for GARCH
from statsmodels.stats.diagnostic import het_arch

resid = arima_res.resid
arch_test = het_arch(resid)
print(f"ARCH test p-value = {arch_test[1]:.3f}") ### if < 0.05, GARCH, if > 0.1 not necessary

## 7.) Specifiy and fit GARCH
from arch import arch_model

garch_mod = arch_model(resid, mean='Zero', vol='Garch', p=1, q=1)

garch_res = garch_mod.fit(disp='off')
print(garch_res.summary()) # α + β > 0.8: strong volatility clustering, < 0.5: weak volatility clustering

## 8.) Plot conditional volatility

plt.figure(figsize=(10,3))
plt.plot(garch_res.conditional_volatility, label='σₜ (cond. vol)')
plt.title('Estimated Game-by-Game Volatility')
plt.xlabel('Game')
plt.ylabel('Volatility')
plt.legend()
plt.tight_layout()
plt.show()