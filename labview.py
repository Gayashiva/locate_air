import math
import numpy as np
from pvlib import location, atmosphere
from datetime import datetime


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

"""Location parameters"""
lat = 46.65
long = 8.2834
alt = 1047.6
utc = 2


"""Misc parameters"""
cld = 1
r = 10
shape_corr = 1.5
temp_i = 0
dis_min = 5
alb = 0.3


def Discharge(aws, mode="auto"):

    # AWS
    time = aws[0]
    temp = aws[1]
    rh = aws[2]
    wind = aws[3]
    SW_global = aws[4]

    # Derived
    press = atmosphere.alt2pres(alt) / 100
    # site = location.Location(lat, long, tz=utc, altitude=alt)
    # solar_angle = site.get_solarposition(times=time, method="ephemeris")["elevation"][0]
    # clearsky = site.get_clearsky(times=time)

    if mode not in ["demo", "auto", "stop"] or mode == "stop":
        dis = 0
    elif wind >= 10:
        dis = 0
    elif mode == "demo":
        dis = 10
    else:

        A = math.pi * r ** 2

        vp_a = (
            6.107
            * math.pow(
                10,
                7.5 * temp / (temp + 237.3),
            )
            * rh
            / 100
        )

        vp_ice = np.exp(43.494 - 6545.8 / (temp_i + 278)) / ((temp_i + 868) ** 2 * 100)

        e_a = (1.24 * math.pow(abs(vp_a / (temp + 273.15)), 1 / 7)) * (
            1 + 0.22 * math.pow(cld, 2)
        )

        SW = (1 - alb) * SW_global
        # SW = (1 - alb) * clearsky["ghi"]

        LW = e_a * STEFAN_BOLTZMAN * math.pow(
            temp + 273.15, 4
        ) - IE * STEFAN_BOLTZMAN * math.pow(273.15 + temp_i, 4)

        Qs = (
            C_A
            * RHO_A
            * press
            / P0
            * math.pow(VAN_KARMAN, 2)
            * wind
            * (temp - temp_i)
            * shape_corr
            / ((np.log(H_AWS / Z)) ** 2)
        )

        Ql = (
            0.623
            * L_S
            * RHO_A
            / P0
            * math.pow(VAN_KARMAN, 2)
            * wind
            * (vp_a - vp_ice)
            * shape_corr
            / ((np.log(H_AWS / Z)) ** 2)
        )

        freezing_energy = Ql + Qs + LW + SW
        freezing_energy += temp_i * RHO_I * DX * C_I / DT
        dis = -1 * freezing_energy * A / L_F * 1000 / 60

        if dis <= dis_min:
            dis = 0

    return round(dis, 1)


if __name__ == "__main__":
    aws = [datetime(2019, 1, 1), -2, 50, 5]
    print("Recommended discharge", Discharge(aws))
