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
from lmfit.models import GaussianModel
import json


def line(x, a1, a2, a3, b):
    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]
    return a1 * x1 + a2 * x2 + a3 * x3 + b


def autoDis(a1, a2, a3, b, amplitude, center, sigma, temp, time=10, rh=10, v=2):
    model = GaussianModel()
    params = {"amplitude": amplitude, "center": center, "sigma": sigma}
    return a1 * temp + a2 * rh + a3 * v + b + model.eval(x=time, **params)


if __name__ == "__main__":
    # compile = True
    compile = False

    sites = ["gangles21", "guttannen21"]

    for site in sites:
        with open("data/" + site + "/info.json") as f:
            params = json.load(f)

        temp = list(range(params["temp"][0], params["temp"][1] + 1))
        rh = list(range(params["rh"][0], params["rh"][1] + 1))
        v = list(range(params["wind"][0], params["wind"][1] + 1))

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

            for temp in da.temp.values:
                for rh in da.rh.values:
                    for v in da.v.values:
                        aws = [temp, rh, v]
                        da.sel(temp=temp, rh=rh, v=v).data += Automate(aws, site)

            da.to_netcdf("data/" + site + "/sims.nc")

        else:

            da = xr.open_dataarray("data/" + site + "/sims.nc")

            x = []
            y = []
            hour = 10

            for i in temp:
                for j in rh:
                    for k in v:
                        x.append([i, j, k])
                        y.append(da.sel(temp=i, rh=j, v=k).data)

            popt, pcov = curve_fit(line, x, y)
            a1, a2, a3, b = popt
            print("y = %.5f * temp + %.5f * rh + %.5f * wind + %.5f" % (a1, a2, a3, b))

            with open("data/" + site + "/daymelt.json") as f:
                param_values = json.load(f)

            param_values["a1"] = a1
            param_values["a2"] = a2
            param_values["a3"] = a3
            param_values["b"] = b

            with open("data/" + site + "/coeff.json", "w") as f:
                json.dump(param_values, f)

            print(
                "Day melt and night freeze:",
                autoDis(**param_values, temp=-10, time=hour),
            )

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
