import pandas as pd
import math
from methods.solar import get_solar
import numpy as np

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


data = []
p_a = 700
RH = 50
cld = 1
A = 3.14 * 100

f = 1.0016+3.15*math.pow(10,-6) * p_a-0.074*math.pow(p_a,-1)

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
    ice = -LW * A/L_F * 1000/60
    data.append([T_a, LW, round(ice,2)])

df = pd.DataFrame(data, columns=['T_a','LW', 'ice'])
print(df) 
