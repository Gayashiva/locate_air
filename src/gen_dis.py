import pandas as pd
import xarray as xr
import numpy as np
import json
from find_discharge import autoDis
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # compile = True
    compile = False

    sites = ["gangles21", "guttannen21"]

    for site in sites:

        with open("data/" + site + "/info.json") as f:
            params = json.load(f)

        hours = list(range(0, 24, 1))
        temp = list(range(params["temp"][0], params["temp"][1] + 1))
        rh = list(range(params["rh"][0], params["rh"][1] + 1))
        v = list(range(params["wind"][0], params["wind"][1] + 1))

        if compile:
            da = xr.DataArray(
                data=np.zeros(len(hours) * len(temp) * len(rh) * len(v)).reshape(
                    len(hours), len(temp), len(rh), len(v)
                ),
                dims=["hours", "temp", "rh", "v"],
                coords=dict(
                    hours=hours,
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

            da.hours.attrs["long_name"] = "Time of day"
            da.hours.attrs["units"] = "hour"
            da.temp.attrs["units"] = "deg C"
            da.temp.attrs["description"] = "Air Temperature"
            da.temp.attrs["long_name"] = "Air Temperature"
            da.rh.attrs["units"] = "%"
            da.rh.attrs["long_name"] = "Relative Humidity"
            da.v.attrs["units"] = "m s-1"
            da.v.attrs["long_name"] = "Wind Speed"

            with open("data/" + site + "/coeff.json") as f:
                param_values = json.load(f)

            for hour in da.hours.values:
                for temp in da.temp.values:
                    for rh in da.rh.values:
                        for v in da.v.values:
                            da.sel(hours=hour, temp=temp, rh=rh, v=v).data += autoDis(
                                **param_values, time=hour, temp=temp, rh=rh, v=v
                            )

            da.to_netcdf("data/" + site + "/dis.nc")

        else:
            da = xr.open_dataarray("data/" + site + "/dis.nc")
            plt.figure()
            ax = plt.gca()
            da.sel(
                temp=slice(params["temp"][0], params["temp"][1] + 1),
                rh=params["rh"][0],
                v=params["wind"][0],
            ).plot(cmap="RdBu")
            plt.legend()
            plt.grid()
            plt.savefig("data/" + site + "/figs/dis.jpg")
