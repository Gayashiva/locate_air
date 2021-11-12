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
            "2019-01-01",
            freq="H",
            periods=1 * 24,
        )

        times -= pd.Timedelta(hours=params["utc"])
        loc = location.Location(
            params["lat"],
            params["long"],
            altitude=params["alt"],
        )

        solar_position = loc.get_solarposition(times=times, method="ephemeris")
        clearsky = loc.get_clearsky(times=times)

        df = pd.DataFrame(
            {
                "ghi": clearsky["ghi"],
                "sea": np.radians(solar_position["elevation"]),
            }
        )
        df.index += pd.Timedelta(hours=params["utc"])
        df.loc[df["sea"] < 0, "sea"] = 0
        df = df.reset_index()
        df["hour"] = df["index"].apply(lambda x: datetime_to_int(x))
        df["f_cone"] = 0

        A = math.pi * params["sa_corr"] * params["r"] ** 2

        for i in range(0, df.shape[0]):
            df.loc[i, "f_cone"] = (
                math.pi * math.pow(params["r"], 2) * 0.5 * math.sin(df.loc[i, "sea"])
            ) / A
            df.loc[i, "SW_direct"] = (
                (1 - params["cld"])
                * df.loc[i, "f_cone"]
                * (1 - params["a_i"])
                * df.loc[i, "ghi"]
            )
            df.loc[i, "SW_diffuse"] = (
                params["cld"] * (1 - params["a_i"]) * df.loc[i, "ghi"]
            )
        # df["dis"] = -1 * (df["SW_direct"] + df["SW_diffuse"]) * A / L_F * 1000 / 60
        df["dis"] = -1 * (df["SW_direct"] + df["SW_diffuse"]) / L_F * 1000 / 60
        print(df.head())

        x = df.hour
        y = df.dis
        y1 = df.f_cone
        model = GaussianModel()
        gauss_params = model.guess(y, x)
        result = model.fit(y, gauss_params, x=df.hour)

        plt.figure()
        plt.plot(x, result.best_fit, "-")
        # plt.plot(x, y1, "-")
        # plt.plot(x, y2, "--")
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
