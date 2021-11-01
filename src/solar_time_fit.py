import numpy as np
from scipy.stats import norm
import scipy
import matplotlib.pyplot as plt
from pvlib import location, atmosphere
import pandas as pd
import xarray as xr

"""Location parameters"""
# Guttannen
lat = 46.65
long = 8.2834
alt = 1047.6
utc = 2

if __name__ == "__main__":
    times = pd.date_range("2019-01-01", freq="H", periods=1 * 24)
    site = location.Location(lat, long, tz=utc, altitude=alt)
    df = site.get_clearsky(times=times)
    df["hour"] = df.index.hour
    print(df.head())

    mean, std = norm.fit(df.ghi)
    print(mean, std)
    # x-axis ranges from -3 and 3 with .001 steps
    x = np.arange(-300, 300, 1)
    # mean, var = scipy.stats.distributions.norm.fit(df.ghi.values)
    # x = np.linspace(0, 24, 1)
    # x = df.hour.values

    # fitted_data = scipy.stats.distributions.norm.cdf(x, mean, var)

    # plt.hist(data, density=True)

    plt.figure()
    ax = plt.gca()
    plt.plot(x / 300 + 10, 90000 * norm.pdf(x, mean, std * 2))
    plt.plot(df.hour, df.ghi)
    # plt.plot(x, fitted_data, "r-")
    plt.legend()
    plt.grid()
    plt.savefig("figs/solar.jpg")
