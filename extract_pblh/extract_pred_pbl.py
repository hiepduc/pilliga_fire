#!/usr/bin/env python3

from netCDF4 import Dataset
from wrf import getvar, ll_to_xy
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# INPUT
# -----------------------
ncfile = "/mnt/scratch_lustre/duch/runpillaga/wrfout_d01_2023-12-17_00:00:00.nc"

lat_site = -32.14   # Merriwa
lon_site = 150.36

nc = Dataset(ncfile)

# -----------------------
# GET TIME SERIES OF PBLH
# -----------------------

pblh_all = getvar(nc, "PBLH", timeidx=None)  # (Time, y, x)

# find nearest grid point
xy = ll_to_xy(nc, lat_site, lon_site)
ix = int(xy[0])
iy = int(xy[1])

print("Grid point:", ix, iy)

# extract time series
pblh_ts = pblh_all[:, iy, ix]

# optional time labels
times = getvar(nc, "times", timeidx=None)

# -----------------------
# PRINT QUICK STATS
# -----------------------
print("PBLH min:", np.min(pblh_ts))
print("PBLH max:", np.max(pblh_ts))

# -----------------------
# PLOT
# -----------------------
plt.figure(figsize=(12,5))
plt.plot(pblh_ts, marker="o")

plt.title("PBL Height Time Series - Merriwa")
plt.ylabel("PBLH (m)")
plt.xlabel("Time index")
plt.grid(True)

plt.show()

