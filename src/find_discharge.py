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
from lmfit.models import LinearModel, GaussianModel
import json

"""Location parameters"""
# Guttannen
lat = 46.65
long = 8.2834
alt = 1047.6
utc = 2


def line(x, a1, a2, a3, b):
    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]
    return a1 * x1 + a2 * x2 + a3 * x3 + b


def dis_func(a1, a2, a3, b, temp, time=1000, rh=10, v=2):
    with open("data/sun_model.json") as f:
        param_values = json.load(f)
    print(param_values)
    model = GaussianModel()
    print("Day melt:", model.eval(x=time, **param_values))
    return a1 * temp + a2 * rh + a3 * v + b + model.eval(x=time, **param_values)


if __name__ == "__main__":
    # compile = True
    compile = False

    # time = datetime(2019, 1, 1, 12)
    # times = pd.date_range("2019-02-01", freq="H", periods=1 * 24)
    temp = list(range(-20, 0))
    rh = list(range(10, 100, 10))
    v = list(range(0, 6, 1))

    # site = location.Location(lat, long, tz=utc, altitude=alt)
    # clearsky = site.get_clearsky(times=times)["ghi"]

    if compile:
        da = xr.DataArray(
            data=np.zeros(len(temp) * len(rh) * len(v)).reshape(
                len(temp), len(rh), len(v)
            ),
            dims=["temp", "rh", "v"],
            coords=dict(
                # times=times,
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

        # for time in da.times.values:
        #     SW_global = clearsky.loc[clearsky.index == time].values[0]
        for temp in da.temp.values:
            for rh in da.rh.values:
                for v in da.v.values:
                    aws = [temp, rh, v]
                    # aws.append(SW_global)
                    da.sel(temp=temp, rh=rh, v=v).data += Automate(aws)

        da.to_netcdf("data/sims.nc")

    else:

        da = xr.open_dataarray("data/sims.nc")

        x = []
        y = []
        # time = datetime(2019, 1, 1)

        # for hour in times.hour:
        for i in temp:
            for j in rh:
                for k in v:
                    # t = time + timedelta(hours=hour)
                    x.append([i, j, k])
                    y.append(da.sel(temp=i, rh=j, v=k).data)

        popt, pcov = curve_fit(line, x, y)
        a1, a2, a3, b = popt
        print("y = %.5f * temp + %.5f * rh + %.5f * wind + %.5f" % (a1, a2, a3, b))
        print("Day melt and night freeze:", dis_func(a1, a2, a3, b, -10, time=0))
        with open("data/sun_model.json") as f:
            param_values = json.load(f)
        print(
            "y = %.5f * temp + %.5f * rh + %.5f * wind + %.5f + Gaussian(time; Amplitude = %.5f, center = %.5f, sigma = %.5f) "
            % (
                a1,
                a2,
                a3,
                b,
                param_values["amplitude"],
                param_values["center"],
                param_values["sigma"],
            )
        )

        plt.figure()
        ax = plt.gca()
        da.sel(temp=slice(-15, None), rh=10, v=2).plot()
        # da.sel(temp=slice(-15, None), rh=10, v=2).plot()
        # da.sel(temp=-15, rh=10, v=2).plot()
        # plt.plot(xdata, ydata)
        # da.sel(temp=-15, rh=10, v=2).plot()
        plt.legend()
        plt.grid()
        plt.savefig("figs/temp_wind.jpg")
