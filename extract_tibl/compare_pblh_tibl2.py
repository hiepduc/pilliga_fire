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
# GRID (SAFE FOR 2D OR 3D)
# ======================================================
xlat = ds.variables["XLAT"]
xlon = ds.variables["XLONG"]

print("XLAT shape:", xlat.shape)

if xlat.ndim == 3:
    lat = xlat[0, :, :]
    lon = xlon[0, :, :]
elif xlat.ndim == 2:
    lat = xlat[:, :]
    lon = xlon[:, :]
else:
    raise ValueError("Unexpected XLAT shape")

dist = np.sqrt((lat - LAT_SITE)**2 + (lon - LON_SITE)**2)
iy, ix = np.unravel_index(np.argmin(dist), dist.shape)

print("Nearest grid:", ix, iy)

# ======================================================
# TIME (SAFE - NO WRF-PYTHON)
# ======================================================
nt = len(ds.variables["Times"])

wrf_times = pd.date_range(
    start="2023-12-08 00:00:00",
    periods=nt,
    freq="1H"
) + pd.Timedelta(hours=UTC_OFFSET)

# ======================================================
# VARIABLES (RAW NETCDF)
# ======================================================
T   = ds.variables["T"][:]        # perturbation potential temperature
W   = ds.variables["W"][:]        # vertical velocity
PH  = ds.variables["PH"][:]
PHB = ds.variables["PHB"][:]
PBLH = ds.variables["PBLH"][:]

g = 9.81

# height AGL
z = (PH + PHB) / g

# potential temperature (approx)
theta = T + 300.0

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
w_col     = extract_column(W, iy, ix)
z_col     = extract_column(z, iy, ix)
pblh_col  = PBLH[:, iy, ix]

# ======================================================
# FIX VERTICAL MISMATCH (IMPORTANT FIX)
# ======================================================
min_k = min(theta_col.shape[1], z_col.shape[1])

theta_col = theta_col[:, :min_k]
w_col     = w_col[:, :min_k]
z_col     = z_col[:, :min_k]

# ======================================================
# TIBL ESTIMATION (ROBUST)
# ======================================================
def estimate_tibl(th, zz):

    th = np.asarray(th)
    zz = np.asarray(zz)

    n = min(len(th), len(zz))
    th = th[:n]
    zz = zz[:n]

    if n < 5:
        return np.nan

    # light smoothing (reduces WRF noise)
    th = pd.Series(th).rolling(3, center=True, min_periods=1).mean().values

    dth_dz = np.gradient(th, zz)

    return zz[np.nanargmax(dth_dz)]

# ======================================================
# COMPUTE TIBL
# ======================================================
tibl = np.array([
    estimate_tibl(theta_col[t], z_col[t])
    for t in range(theta_col.shape[0])
])

# ======================================================
# DATAFRAME
# ======================================================
df = pd.DataFrame({
    "time": wrf_times,
    "PBLH": pblh_col,
    "TIBL": tibl
}).set_index("time")

print("Records:", len(df))

# ======================================================
# CEILOMETER
# ======================================================
csv_files = [
    "ceilo_data/L3_DEFAULT_0_20231206_Lidcombe.csv",
    "ceilo_data/L3_DEFAULT_0_20231207_Lidcombe.csv",
    "ceilo_data/L3_DEFAULT_0_20231208_Lidcombe.csv",
    "ceilo_data/L3_DEFAULT_0_20231209_Lidcombe.csv",
    "ceilo_data/L3_DEFAULT_0_20231210_Lidcombe.csv",
    "ceilo_data/L3_DEFAULT_0_20231211_Lidcombe.csv",
    "ceilo_data/L3_DEFAULT_0_20231212_Lidcombe.csv",
    "ceilo_data/L3_DEFAULT_0_20231213_Lidcombe.csv",
    "ceilo_data/L3_DEFAULT_0_20231214_Lidcombe.csv",
    "ceilo_data/L3_DEFAULT_0_20231215_Lidcombe.csv",
    "ceilo_data/L3_DEFAULT_0_20231216_Lidcombe.csv",
    "ceilo_data/L3_DEFAULT_0_20231217_Lidcombe.csv",
    "ceilo_data/L3_DEFAULT_0_20231218_Lidcombe.csv",
    "ceilo_data/L3_DEFAULT_0_20231220_Lidcombe.csv"
]

dfs = []

for f in csv_files:
    print("Reading", f)

    d = pd.read_csv(f)

    d["Time"] = pd.to_datetime(d["# Time"], dayfirst=True, errors="coerce")
    d["bl_height"] = pd.to_numeric(d["bl_height"], errors="coerce")

    d.loc[d["bl_height"] < 0, "bl_height"] = np.nan

    dfs.append(d)

ceilo = pd.concat(dfs)
ceilo = ceilo.sort_values("Time")

ceilo_hourly = ceilo.set_index("Time").resample("1H").median(numeric_only=True)

# ======================================================
# PLOT TIME SERIES
# ======================================================
plt.figure(figsize=(14,6))

plt.plot(df.index, df["PBLH"], label="WRF PBLH")
plt.plot(df.index, df["TIBL"], label="TIBL (estimated)")
plt.plot(ceilo_hourly.index, ceilo_hourly["bl_height"], label="Ceilometer")

plt.ylabel("Height (m)")
plt.title("TIBL vs PBLH vs Ceilometer")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# ======================================================
# VERTICAL PROFILES (example time)
# ======================================================
t0 = 10

plt.figure()
plt.plot(theta_col[t0], z_col[t0])
plt.xlabel("Theta (K)")
plt.ylabel("Height (m)")
plt.title("Potential Temperature Profile")
plt.grid()
plt.show()

plt.figure()
plt.plot(w_col[t0], z_col[t0])
plt.axvline(0, color="black")
plt.xlabel("W (m/s)")
plt.ylabel("Height (m)")
plt.title("Vertical Velocity Profile")
plt.grid()
plt.show()

