#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from netCDF4 import Dataset

# ======================================================
# SETTINGS
# ======================================================
UTC_OFFSET = 10

LAT_SITE = -33.89
LON_SITE = 151.05

wrf_file = "/mnt/scratch_lustre/duch/runpillaga/jkong/wrfout_d02_2023-12-8to19.nc"

# ======================================================
# OPEN FILE
# ======================================================
ds = Dataset(wrf_file)

# ======================================================
# GRID
# ======================================================
lat_var = ds.variables["XLAT"]
lon_var = ds.variables["XLONG"]

print("XLAT shape:", lat_var.shape)

if lat_var.ndim == 3:
    lat = lat_var[0, :, :]
    lon = lon_var[0, :, :]
elif lat_var.ndim == 2:
    lat = lat_var[:, :]
    lon = lon_var[:, :]
else:
    raise ValueError(f"Unexpected XLAT shape: {lat_var.shape}")

dist = np.sqrt((lat - LAT_SITE)**2 + (lon - LON_SITE)**2)
iy, ix = np.unravel_index(np.argmin(dist), dist.shape)

print("Nearest grid:", ix, iy)

# ======================================================
# TIME (RAW SAFE)
# ======================================================
nt = len(ds.variables["Times"])

wrf_times = pd.date_range(
    start="2023-12-08 00:00:00",
    periods=nt,
    freq="1H"
) + pd.Timedelta(hours=UTC_OFFSET)

# ======================================================
# EXTRACT VARIABLES (RAW NETCDF)
# ======================================================
theta = ds.variables["T"][:]        # perturbation temp (K)
w     = ds.variables["W"][:]        # vertical velocity (m/s)
ph    = ds.variables["PH"][:]       # geopotential perturbation
phb   = ds.variables["PHB"][:]      # base geopotential
pblh  = ds.variables["PBLH"][:]

# ======================================================
# HEIGHT (AGL)
# ======================================================
g = 9.81

z = (ph + phb) / g

# ======================================================
# CONVERT TO NUMPY
# ======================================================
theta = np.array(theta)
w     = np.array(w)
z     = np.array(z)
pblh  = np.array(pblh)

# ======================================================
# SAFE COLUMN EXTRACTION
# ======================================================
def extract_column(var, iy, ix):

    if var.ndim == 4:
        return var[:, :, iy, ix]
    elif var.ndim == 3:
        return var[:, iy, ix]
    else:
        raise ValueError(var.shape)

theta_col = extract_column(theta, iy, ix)
w_col     = extract_column(w, iy, ix)
z_col     = extract_column(z, iy, ix)
pblh_col  = pblh[:, iy, ix]

# ======================================================
# TIBL ESTIMATION
# ======================================================
def estimate_tibl(th, zz):

    mask = ~np.isnan(th) & ~np.isnan(zz)

    th = th[mask]
    zz = zz[mask]

    if len(th) < 5:
        return np.nan

    dth_dz = np.gradient(th, zz)

    return zz[np.nanargmax(dth_dz)]

tibl = []

for t in range(theta_col.shape[0]):
    tibl.append(estimate_tibl(theta_col[t], z_col[t]))

tibl = np.array(tibl)

# ======================================================
# DATAFRAME
# ======================================================
df = pd.DataFrame({
    "time": wrf_times,
    "PBLH": pblh_col,
    "TIBL": tibl
}).set_index("time")

# ======================================================
# PLOT
# ======================================================
plt.figure(figsize=(14,6))

plt.plot(df.index, df["PBLH"], label="WRF PBLH")
plt.plot(df.index, df["TIBL"], label="TIBL")

plt.ylabel("Height (m)")
plt.title("TIBL vs PBLH")
plt.legend()
plt.grid()
plt.show()

