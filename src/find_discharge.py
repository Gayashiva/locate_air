import pandas as pd
import xarray as xr
from scipy.optimize import curve_fit
from numpy import exp, loadtxt, pi, sqrt
import math
import numpy as np
from datetime import datetime, timedelta
from pvlib import location, atmosphere
import matplotlib.pyplot as plt
from autoDischarge import Automate

"""Location parameters"""
# Guttannen
lat = 46.65
long = 8.2834
alt = 1047.6
utc = 2


def gaussian_plus_line(x, amp, cen, wid, a2, a3, a4, b):
    """line + 1-d gaussian"""

    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]
    x4 = x[:, 3]

    gauss = (amp / (sqrt(2 * pi) * wid)) * exp(-((x1 - cen) ** 2) / (2 * wid ** 2))
    line = a2 * x2 + a3 * x3 + a4 * x4 + b
    return gauss + line


def line(x, a12, a11, a2, a3, a4, b):
    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]
    x4 = x[:, 3]
    # return a12 * x1 ** 2 + a11 * x1 + a2 * x2 + a3 * x3 + a4 * x4 + b
    return a12 * math.sin(x1 - a11) + a2 * x2 + a3 * x3 + a4 * x4 + b


if __name__ == "__main__":
    compile = True
    # compile = False

    # time = datetime(2019, 1, 1, 12)
    # times = pd.date_range("2019-01-01", freq="H", periods=1 * 24)
    times = pd.date_range("2019-01-01", freq="H", periods=1 * 1)
    temp = list(range(-15, 0))
    rh = list(range(10, 100, 10))
    v = list(range(0, 6, 1))

    site = location.Location(lat, long, tz=utc, altitude=alt)
    clearsky = site.get_clearsky(times=times)["ghi"]

    if compile:
        da = xr.DataArray(
            data=np.zeros(len(times) * len(temp) * len(rh) * len(v)).reshape(
                len(times), len(temp), len(rh), len(v)
            ),
            dims=["times", "temp", "rh", "v"],
            coords=dict(
                times=times,
                temp=temp,
                rh=rh,
                v=v,
            ),
            attrs=dict(
                long_name="Freezing rate",
                description="Max. freezing rate",
                units="l min-1",
            ),
        )

        da.temp.attrs["units"] = "deg C"
        da.temp.attrs["description"] = "Air Temperature"
        da.temp.attrs["long_name"] = "Air Temperature"
        da.rh.attrs["units"] = "%"
        da.rh.attrs["long_name"] = "Relative Humidity"
        da.v.attrs["units"] = "m s-1"
        da.v.attrs["long_name"] = "Wind Speed"

        for time in da.times.values:
            SW_global = clearsky.loc[clearsky.index == time].values[0]
            for temp in da.temp.values:
                for rh in da.rh.values:
                    for v in da.v.values:
                        aws = [time, temp, rh, v]
                        aws.append(SW_global)
                        da.sel(times=time, temp=temp, rh=rh, v=v).data += Automate(aws)

        da.to_netcdf("data/sims.nc")

    else:

        da = xr.open_dataarray("data/sims.nc")

        x = []
        y = []
        time = datetime(2019, 1, 1)

        for hour in times.hour:
            for i in temp:
                for j in rh:
                    for k in v:
                        t = time + timedelta(hours=hour)
                        x.append([hour, i, j, k])
                        y.append(da.sel(times=t, temp=i, rh=j, v=k).data)

        popt, pcov = curve_fit(gaussian_plus_line, x, y)
        amp, cen, wid, a2, a3, a4, b = popt
        # print(
        #     "y = %.5f * hour**2 + %.5f * hour + %.5f * temp + %.5f * rh + %.5f * wind + %.5f"
        #     % (amp, wid, a2, a3, a4, b)
        # )

        xdata = []
        ydata = []
        for hour in times.hour:
            xdata.append(time + timedelta(hours=hour))
            x1 = hour
            x2 = -15
            x3 = 10
            x4 = 2
            gauss = (amp / (sqrt(2 * pi) * wid)) * exp(
                -((x1 - cen) ** 2) / (2 * wid ** 2)
            )
            line = a2 * x2 + a3 * x3 + a4 * x4 + b
            ydata.append(gauss + line)

        plt.figure()
        ax = plt.gca()
        # da.sel(temp=slice(-15, None), rh=10, v=2, times=time).plot()
        # da.sel(temp=slice(-15, None), rh=10, v=2).plot()
        da.sel(temp=-15, rh=10, v=2).plot()
        plt.plot(xdata, ydata)
        # da.sel(temp=-15, rh=10, v=2).plot()
        plt.legend()
        plt.grid()
        plt.savefig("figs/temp_wind.jpg")

        # xfine = np.linspace(-10, 0)  # define values to plot the function for
        # plt.figure()
        # ax = plt.gca()
        # plt.plot(xfine, line(xfine, popt[0], popt[1]), "r-")
        # da.sel(temp=slice(-15, None), rh=50, v=2).plot()
        # plt.legend()
        # plt.grid()
        # plt.savefig("fit_temp.jpg")
