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
DX = 20e-03  # m Surface layer thickness growth rate
H_AWS = 2


def max_discharge(Temp, RH, Wind, p_a=700, cld=0):

    T_min = Temp
    RH_min = RH
    v_max = Wind
    r = 10
    h_max = r
    shape_corr = 1.5

    A = (
        math.pi
        * r
        * math.pow(
            (math.pow(r, 2) + math.pow(h_max, 2)),
            1 / 2,
        )
    )

    Qs = (
        C_A
        * RHO_A
        * p_a
        / P0
        * math.pow(VAN_KARMAN, 2)
        * v_max
        * (T_min - T_s_min)
        * shape_corr
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

    vp_ice = np.exp(43.494 - 6545.8 / (T_s_min + 278)) / ((T_s_min + 868) ** 2 * 100)

    e_a = (1.24 * math.pow(abs(vp_a / (T_min + 273.15)), 1 / 7)) * (
        1 + 0.22 * math.pow(cld, 2)
    )

    LW = e_a * STEFAN_BOLTZMAN * math.pow(
        T_min + 273.15, 4
    ) - IE * STEFAN_BOLTZMAN * math.pow(273.15 + T_s_min, 4)
    Ql = (
        0.623
        * L_S
        * RHO_A
        / P0
        * math.pow(VAN_KARMAN, 2)
        * v_max
        * (vp_a - vp_ice)
        * shape_corr
        / ((np.log(H_AWS / Z)) ** 2)
    )
    freezing_energy = Ql + Qs + LW
    freezing_energy += T_s_min * RHO_I * DX * C_I / DT
    freeze_rate = -1 * freezing_energy * A / L_F * 1000 / 60

    if freeze_rate < 0:
        freeze_rate = 0

    return round(freeze_rate, 1)
