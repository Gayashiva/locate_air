import numpy as np
from scipy.stats import norm
import scipy
import matplotlib.pyplot as plt
from pvlib import location, atmosphere
import pandas as pd
import xarray as xr
import math
from lmfit.models import LinearModel, GaussianModel
import json


def gaussian(x, amp, cen, wid):
    return amp * np.exp(-((x - cen) ** 2) / wid)


def datetime_to_int(dt):
    return int(dt.strftime("%H%M"))


r = 8
L_F = 334 * 1000  # J/kg Fusion

"""Location parameters"""
# Guttannen
lat = 46.65
long = 8.2834
alt = 1047.6
utc = 2

if __name__ == "__main__":
    # times = pd.date_range("2019-02-01", freq="H", periods=1 * 24)
    times = pd.date_range("2019-02-01", freq="T", periods=1 * 24 * 60)
    site = location.Location(lat, long, tz=utc, altitude=alt)
    df = site.get_clearsky(times=times)
    df = df.reset_index()

    df["hour_minute"] = df["index"].apply(lambda x: datetime_to_int(x))
    A = math.pi * r ** 2
    df["dis"] = -1 * df["ghi"] * A / L_F * 1000 / 60

    x = df.hour_minute
    y = df.dis.values
    # model = LorentzianModel()
    model = GaussianModel()
    params = model.guess(y, x)
    result = model.fit(y, params, x=df.hour_minute)
    # params = model.guess(df["dis"], x=df.hour_minute)
    # result = model.fit(df["dis"], params, x=df.hour_minute)

    plt.figure()
    # ax = result.plot_fit()
    plt.plot(x, result.best_fit, "-", label="best fit")
    plt.ylabel("Melt rate [l/min]")
    plt.xlabel("Time [minutes]")
    plt.legend()
    plt.grid()
    plt.savefig("figs/solar.jpg")

    print(result.fit_report())

    print(f"parameter names: {model.param_names}")
    print(f"independent variables: {model.independent_vars}")
    param_values = dict(result.best_values)
    with open("data/sun_model.json", "w") as f:
        json.dump(param_values, f)

    # print(**param_values)
    # save_model(sun_dis_model, "data/sun_dis_model.sav")
    model2 = GaussianModel()
    print(model2.eval(x=1000, **param_values))
