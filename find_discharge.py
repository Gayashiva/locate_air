import pandas as pd
import xarray as xr
from scipy.optimize import curve_fit
import math
import numpy as np
from datetime import datetime, timedelta
from methods.solar import get_solar
import matplotlib
import matplotlib.pyplot as plt
from labview import Discharge

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
DX = 20e-03  # m Surface layer thickness growth rate
H_AWS = 2


def line(x, a, b):
    return a * x + b


def line2(x, a1, a2, a3, b):
    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]
    return a1 * x1 + a2 * x2 + a3 * x3 + b


if __name__ == "__main__":
    temp = list(range(-20, 0))
    rh = list(range(10, 90, 10))
    v = list(range(0, 10))
    da = xr.DataArray(
        data=np.zeros(len(temp) * len(rh) * len(v)).reshape(len(temp), len(rh), len(v)),
        dims=["temp", "rh", "v"],
        coords=dict(
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

    time = datetime(2019, 1, 1)
    # aws = [datetime(2019, 1, 1), -2, 50, 5]

    for i in temp:
        for j in rh:
            for k in v:
                aws = [time, i, j, k]
                da.sel(temp=i, rh=j, v=k).data += Discharge(aws)

    # da.sel(temp=slice(-15, None), rh=50, v=2).plot()
    # y = da.sel(temp=slice(-15, None), rh=50, v=2).values
    # x = da.sel(temp=slice(-10, None), rh=50, v=2).temp.values
    # y2 = da.sel(rh=50).data
    # x1 = da.sel(rh=50).temp.values
    # x2 = da.sel(rh=50).v.values
    # plt.figure()
    # ax = plt.gca()
    # da.sel(temp=slice(-15, None), rh=50, v=2).plot()
    # plt.legend()
    # plt.grid()
    # plt.savefig("temp_wind.jpg")

    x = []
    y = []
    for i in temp:
        for j in rh:
            for k in v:
                x.append([i, j, k])
                y.append(da.sel(temp=i, rh=j, v=k).data)
    # print(y)
    # print(x)

    popt, pcov = curve_fit(line2, x, y)
    a1, a2, a3, b = popt
    print("y = %.5f * temp + %.5f * rh + %.5f * wind + %.5f" % (a1, a2, a3, b))

    # xfine = np.linspace(-10, 0)  # define values to plot the function for
    # plt.figure()
    # ax = plt.gca()
    # plt.plot(xfine, line(xfine, popt[0], popt[1]), "r-")
    # da.sel(temp=slice(-15, None), rh=50, v=2).plot()
    # plt.legend()
    # plt.grid()
    # plt.savefig("fit_temp.jpg")
