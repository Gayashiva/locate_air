import pandas as pd
import xarray as xr
import math
import numpy as np
from datetime import datetime, timedelta
from methods.solar import get_solar
import matplotlib
import matplotlib.pyplot as plt

"""Model hyperparameter"""
DT = 60 * 60  # Model time step

"""Physical Constants"""
L_S = 2848 * 1000  # J/kg Sublimation
L_F = 334 * 1000  # J/kg Fusion
C_A = 1.01 * 1000  # J/kgC Specific heat air
C_I = 2.097 * 1000  # J/kgC Specific heat ice
C_W = 4.186 * 1000  # J/kgC Specific heat water
RHO_W = 1000  # Density of water
RHO_I = 917  # Density of Ice RHO_I
RHO_A = 1.29  # kg/m3 air density at mean sea level
VAN_KARMAN = 0.4  # Van Karman constant
K_I = 2.123  # Thermal Conductivity Waite et al. 2006
STEFAN_BOLTZMAN = 5.670367 * math.pow(10, -8)  # Stefan Boltzman constant
P0 = 1013  # Standard air pressure hPa
G = 9.81  # Gravitational acceleration

"""Surface Properties"""
IE = 0.97  # Ice Emissivity IE
A_I = 0.25  # Albedo of Ice A_I
A_S = 0.85  # Albedo of Fresh Snow A_S
A_DECAY = 16  # Albedo decay rate decay_t_d
Z = 0.003  # Ice Momentum and Scalar roughness length
T_PPT = 1  # Temperature condition for liquid precipitation
H_AWS = 2

vp_ice = np.exp(43.494 - 6545.8 / (0 + 278)) / ((0 + 868) ** 2 * 100)


def max_discharge(data, p_a=700, cld=0):

    T_min = data[0]
    RH_min = data[1]
    v_max = data[2]
    r = data[3]
    h_max = r

    A = (
        math.pi
        * r
        * math.pow(
            (math.pow(r, 2) + math.pow(h_max, 2)),
            1 / 2,
        )
    )
    # A = 3.14 * r ** 2

    Qs = (
        C_A
        * RHO_A
        * p_a
        / P0
        * math.pow(VAN_KARMAN, 2)
        * v_max
        * T_min
        / ((np.log(H_AWS / Z)) ** 2)
    )

    SW = 0

    vp_a = (
        6.107
        * math.pow(
            10,
            7.5 * T_min / (T_min + 237.3),
        )
        * RH_min
        / 100
    )

    e_a = (1.24 * math.pow(abs(vp_a / (T_min + 273.15)), 1 / 7)) * (
        1 + 0.22 * math.pow(cld, 2)
    )

    LW = e_a * STEFAN_BOLTZMAN * math.pow(
        T_min + 273.15, 4
    ) - IE * STEFAN_BOLTZMAN * math.pow(273.15, 4)
    Ql = (
        0.623
        * L_S
        * RHO_A
        / P0
        * math.pow(VAN_KARMAN, 2)
        * v_max
        * (vp_a - vp_ice)
        / ((np.log(H_AWS / Z)) ** 2)
    )
    freeze_rate = -(Ql + Qs + LW) * A / L_F * 1000 / 60

    if freeze_rate < 0:
        freeze_rate = 0

    return round(freeze_rate, 1)


if __name__ == "__main__":

    compile = 1

    if compile:

        locations = ["guttannen", "diavolezza", "schwarzsee", "gangles", "ravat"]
        points = []
        for loc in locations:
            print(loc)
            if loc == "schwarzsee":
                df = pd.read_csv(
                    "/home/suryab/work/air_model/data/"
                    + loc
                    + "19/interim/"
                    + loc
                    + "19_input_model.csv",
                    sep=",",
                    header=0,
                    # parse_dates=["TIMESTAMP"],
                    parse_dates=["When"],
                )
                df = df.rename(
                    columns={"When": "time", "T_a": "temp", "RH": "rh", "v_a": "v"}
                )
                df = df.set_index("time")
                df = df[["temp", "rh", "v"]].to_xarray().sel(time="2019-02")
            if loc == "ravat":
                df = pd.read_csv(
                    "/home/suryab/work/air_model/data/"
                    + loc
                    + "20/interim/"
                    + loc
                    + "20_input_ERA5.csv",
                    sep=",",
                    header=0,
                    # parse_dates=["TIMESTAMP"],
                    parse_dates=["When"],
                )
                df = df.rename(
                    columns={"When": "time", "T_a": "temp", "RH": "rh", "v_a": "v"}
                )
                df = df.set_index("time")
                df = df[["temp", "rh", "v"]].to_xarray().sel(time="2020-01")
            if loc == "diavolezza":
                df = pd.read_csv(
                    "/home/suryab/work/air_model/data/"
                    + loc
                    + "21/interim/"
                    + loc
                    + "21_input_ERA5.csv",
                    sep=",",
                    header=0,
                    # parse_dates=["TIMESTAMP"],
                    parse_dates=["When"],
                )
                df = df.rename(
                    columns={"When": "time", "T_a": "temp", "RH": "rh", "v_a": "v"}
                )
                df = df.set_index("time")
                df = df[["temp", "rh", "v"]].to_xarray().sel(time="2021-01")
            if loc == "guttannen":
                df = pd.read_csv(
                    "/home/suryab/work/air_model/data/"
                    + loc
                    + "21/interim/"
                    + loc
                    + "21_input_ERA5.csv",
                    sep=",",
                    header=0,
                    parse_dates=["TIMESTAMP"],
                )
                df = df.rename(
                    columns={"TIMESTAMP": "time", "T_A": "temp", "RH": "rh", "WS": "v"}
                )
                df = df.set_index("time")
                df = df[["temp", "rh", "v"]].to_xarray().sel(time="2020-01")
            if loc == "gangles":
                df = pd.read_csv(
                    "/home/suryab/work/air_model/data/"
                    + loc
                    + "21/interim/"
                    + loc
                    + "21_input_model.csv",
                    sep=",",
                    header=0,
                    parse_dates=["TIMESTAMP"],
                )
                df = df.rename(
                    columns={"TIMESTAMP": "time", "T_A": "temp", "RH": "rh", "WS": "v"}
                )
                df = df.set_index("time")
                df = df[["temp", "rh", "v"]].to_xarray().sel(time="2021-01")
            q1 = df.temp.quantile([0.25, 0.5, 0.75]).values
            q2 = df.v.quantile([0.25, 0.5, 0.75]).values
            points.append([q2[1], q1[1]])
        print(points)

        temp = list(range(-20, 2))
        rh = list(range(10, 90, 10))
        v = list(range(0, 10))
        r = list(range(6, 11))
        da = xr.DataArray(
            # {
            #     "discharge": (
            #         ("temp", "rh", "v", "r"),
            #         np.zeros(len(temp) * len(rh) * len(v) * len(r)).reshape(
            #             len(temp), len(rh), len(v), len(r)
            #         ),
            #     )
            # },
            data=np.zeros(len(temp) * len(rh) * len(v) * len(r)).reshape(
                len(temp), len(rh), len(v), len(r)
            ),
            dims=["temp", "rh", "v", "r"],
            coords=dict(
                temp=temp,
                rh=rh,
                v=v,
                r=r,
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
        da.r.attrs["units"] = "m"
        da.r.attrs["long_name"] = "Spray Radius"
        da.v.attrs["units"] = "m s-1"
        da.v.attrs["long_name"] = "Wind Speed"

        for i in temp:
            for j in rh:
                for k in v:
                    for l in r:
                        # da.sel(temp=i, rh=j, v=k, r=l)["data"] += max_discharge(
                        da.sel(temp=i, rh=j, v=k, r=l).data += max_discharge(
                            [i, j, k, l]
                        )
        da.to_netcdf("saved_on_disk.nc")
        # da1 = da.sel(temp=slice(-15, None), rh=80, v=6, r=7)

        plt.figure()
        ax = plt.gca()
        da.sel(temp=slice(-15, None), rh=80, r=10).plot()
        for ctr, point in enumerate(points):
            plt.scatter(point[0], point[1], label=locations[ctr])
            discharge = da.sel(
                temp=point[1], v=point[0], rh=80, r=10, method="nearest"
            ).values
            plt.annotate(discharge, (point[0], point[1]), color="yellow")
        plt.legend()
        plt.grid()
        plt.savefig("try2.jpg")
    # else:
    #     with xr.open_dataset("saved_on_disk.nc") as da:
    #         print(da)
    #         da1 = da.sel(temp=slice(-15, None), rh=80, v=6, r=7)
    #         print(da1.discharge)
    #         plt.figure()
    #         ax = plt.gca()
    #         da1.to_dataframe()["discharge"].plot()
    #         plt.legend()
    #         plt.grid()
    #         plt.savefig("try2.jpg")
