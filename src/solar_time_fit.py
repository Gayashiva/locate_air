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
import pytz


L_F = 334 * 1000  # J/kg Fusion


def datetime_to_int(dt):
    return int(dt.strftime("%H"))


if __name__ == "__main__":
    sites = ["gangles21", "guttannen21"]
    for site in sites:

        with open("data/" + site + "/info.json") as f:
            params = json.load(f)

        times = pd.date_range(
            "2019-02-01",
            freq="H",
            periods=1 * 24,
        )
        times -= pd.Timedelta(hours=params["utc"])
        loc = location.Location(
            params["lat"],
            params["long"],
            altitude=params["alt"],
        )
        df = loc.get_clearsky(times=times)
        df.index += pd.Timedelta(hours=params["utc"])
        df = df.reset_index()

        df["hour_minute"] = df["index"].apply(lambda x: datetime_to_int(x))
        A = math.pi * params["sa_corr"] * params["r"] ** 2
        df["dis"] = -1 * df["ghi"] * params["f_cone"] * A / L_F * 1000 / 60

        x = df.hour_minute
        y = df.dis.values
        model = GaussianModel()
        gauss_params = model.guess(y, x)
        result = model.fit(y, gauss_params, x=df.hour_minute)

        plt.figure()
        plt.plot(x, result.best_fit, "-")
        plt.ylabel("Daymelt [l min-1]")
        plt.xlabel("Time of day [hour]")
        plt.legend()
        plt.grid()
        plt.savefig("data/" + site + "/figs/daymelt.jpg")

        print(f"parameter names: {model.param_names}")
        print(f"independent variables: {model.independent_vars}")
        param_values = dict(result.best_values)
        with open("data/" + site + "/daymelt.json", "w") as f:
            json.dump(param_values, f)
