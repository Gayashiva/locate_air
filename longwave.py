import pandas as pd
import math
import numpy as np
from datetime import datetime
from methods.solar import get_solar

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
A_DECAY = 16 # Albedo decay rate decay_t_d
Z = 0.003  # Ice Momentum and Scalar roughness length
T_PPT = 1  # Temperature condition for liquid precipitation
# DX = 20e-03  # m Surface layer thickness growth rate

H_AWS = 2
vp_ice = (
    np.exp(43.494- 6545.8 / (0+278))
    / ((0+ 868)**2 * 100)
)

def location_energy(p_a = 700, RH = 50, cld = 1, r = 10, v_a = 2, SW_global=1000):

    data = []
    A = 3.14 * r**2
    # f = 1.0016+3.15*math.pow(10,-6) * p_a-0.074*math.pow(p_a,-1)

    for T_a in range(-10, 10):
        vp_a = (
            6.107
            * math.pow(
                10,
                7.5 * T_a / (T_a + 237.3),
            )
            * RH
            / 100
        )
        e_a = (1.24* math.pow(abs(vp_a / (T_a + 273.15)), 1 / 7)) * (1 + 0.22 * math.pow(cld, 2))
        LW = e_a * STEFAN_BOLTZMAN * math.pow(T_a + 273.15, 4) - IE * STEFAN_BOLTZMAN * math.pow(273.15,4)
        Ql = (
            0.623
            * L_S
            * RHO_A
            / P0
            * math.pow(VAN_KARMAN, 2)
            * v_a
            * (vp_a - vp_ice)
            / ((np.log(H_AWS / Z)) ** 2)
        )

        Qs = (
            C_A
            * RHO_A
            * p_a
            / P0
            * math.pow(VAN_KARMAN, 2)
            * v_a
            * T_a
            / ((np.log(H_AWS / Z)) ** 2)
        )

        SW = (1 - A_I) * SW_global

        ice = (Ql+Qs+LW) * A/L_F * 1000/60
        data.append([T_a, LW, SW, Ql, Qs, round(ice,2)])

    df = pd.DataFrame(data, columns=['T_a','LW', 'SW', 'Ql', 'Qs','ice'])
    return df

if __name__ == "__main__":

    latitude=46.649999
    longitude=8.283333
    start_date=datetime(2021, 1, 1)
    end_date=datetime(2021, 1, 31) 
    solar_df = get_solar(
        latitude=latitude,
        longitude=longitude,
        start=start_date,
        end=end_date,
        DT=DT,
    )
    print(solar_df.ghi.max())
    df = location_energy(SW_global=solar_df.ghi.max())
    print(df) 

