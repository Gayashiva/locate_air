import pandas as pd
import xarray as xr
from scipy.optimize import curve_fit
import math
import numpy as np
from datetime import datetime, timedelta
from pvlib import location, atmosphere
import matplotlib
import matplotlib.pyplot as plt
from labview import Discharge

"""Location parameters"""
lat = 46.65
long = 8.2834
alt = 1047.6
utc = 2


def line(x, a1, a2, a3, b):
    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]
    return a1 * x1 + a2 * x2 + a3 * x3 + b


if __name__ == "__main__":
    compile = True
    compile = False

    # time = datetime(2019, 1, 1, 12)
    times = pd.date_range("2019-01-01", freq="H", periods=1 * 24)
    temp = list(range(-15, 0))
    rh = list(range(10, 90, 30))
    v = list(range(0, 10, 2))

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

        for time in times:
            for i in temp:
                for j in rh:
                    for k in v:
                        aws = [time, i, j, k]
                        aws.append(clearsky.loc[clearsky.index == time].values[0])
                        da.sel(times=time, temp=i, rh=j, v=k).data += Discharge(aws)

        da.to_netcdf("sims.nc")
    else:
        da = xr.open_dataarray("sims.nc")

        x = []
        y = []
        time = datetime(2019, 1, 1, 12)
        for i in temp:
            for j in rh:
                for k in v:
                    x.append([i, j, k])
                    y.append(da.sel(times=time, temp=i, rh=j, v=k).data)

        popt, pcov = curve_fit(line, x, y)
        a1, a2, a3, b = popt
        print("y = %.5f * temp + %.5f * rh + %.5f * wind + %.5f" % (a1, a2, a3, b))

        da.sel(temp=slice(-15, None), rh=10, v=2, times=time).plot()
        # y = da.sel(temp=slice(-15, None), rh=10, v=2, times=time).values
        # x = da.sel(temp=slice(-10, None), rh=10, v=2, times=time).temp.values
        # y2 = da.sel(rh=10, times=time).data
        # x1 = da.sel(rh=10, times=time).temp.values
        # x2 = da.sel(rh=10, times=time).v.values
        plt.figure()
        ax = plt.gca()
        # da.sel(temp=slice(-15, None), rh=10, v=2, times=time).plot()
        da.sel(temp=slice(-15, None), rh=10, v=2).plot()
        # da.sel(temp=-15, rh=10, v=2).plot()
        plt.legend()
        plt.grid()
        plt.savefig("temp_wind.jpg")

        # xfine = np.linspace(-10, 0)  # define values to plot the function for
        # plt.figure()
        # ax = plt.gca()
        # plt.plot(xfine, line(xfine, popt[0], popt[1]), "r-")
        # da.sel(temp=slice(-15, None), rh=50, v=2).plot()
        # plt.legend()
        # plt.grid()
        # plt.savefig("fit_temp.jpg")
