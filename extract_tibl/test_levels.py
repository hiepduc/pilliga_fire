#!/usr/bin/env python3

from netCDF4 import Dataset
import numpy as np

f = Dataset("../wrfout_d02_2023-12-08_00:00:00")

lat = f.variables["XLAT"][:]
lon = f.variables["XLONG"][:]

LAT_SITE = -33.89
LON_SITE = 151.05

dist = np.sqrt(
    (lat-LAT_SITE)**2 +
    (lon-LON_SITE)**2
)

iy, ix = np.unravel_index(
    np.argmin(dist),
    dist.shape
)

print("Grid =", ix, iy)

ph  = f.variables["PH"][0,:,iy,ix]
phb = f.variables["PHB"][0,:,iy,ix]

z_stag = (ph+phb)/9.81

z_mass = 0.5*(z_stag[:-1] + z_stag[1:])

print("Levels =", len(z_mass))

print(z_mass[:10])

print(z_mass[-5:])

