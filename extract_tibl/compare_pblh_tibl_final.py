#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from netCDF4 import Dataset

from wrf import (
    getvar,
    ll_to_xy,
    ALL_TIMES
)

# =====================================================
# SETTINGS
# =====================================================

LAT_SITE = -33.89
LON_SITE = 151.05

UTC_OFFSET = 10

# =====================================================
# WRF FILES
# =====================================================

wrf_files = [
    "/mnt/scratch_lustre/duch/runpillaga/wrfout_d02_2023-12-06_00:00:00",
    "/mnt/scratch_lustre/duch/runpillaga/wrfout_d02_2023-12-07_00:00:00",
    "/mnt/scratch_lustre/duch/runpillaga/wrfout_d02_2023-12-08_00:00:00",
    "/mnt/scratch_lustre/duch/runpillaga/wrfout_d02_2023-12-09_00:00:00",
    "/mnt/scratch_lustre/duch/runpillaga/wrfout_d02_2023-12-10_00:00:00",
    "/mnt/scratch_lustre/duch/runpillaga/wrfout_d02_2023-12-11_00:00:00",
    "/mnt/scratch_lustre/duch/runpillaga/wrfout_d02_2023-12-12_00:00:00",
    "/mnt/scratch_lustre/duch/runpillaga/wrfout_d02_2023-12-13_00:00:00",
    "/mnt/scratch_lustre/duch/runpillaga/wrfout_d02_2023-12-14_01:00:00",
    "/mnt/scratch_lustre/duch/runpillaga/wrfout_d02_2023-12-15_01:00:00",
    "/mnt/scratch_lustre/duch/runpillaga/wrfout_d02_2023-12-16_01:00:00",
    "/mnt/scratch_lustre/duch/runpillaga/wrfout_d02_2023-12-17_01:00:00",
    "/mnt/scratch_lustre/duch/runpillaga/wrfout_d02_2023-12-18_01:00:00",
    "/mnt/scratch_lustre/duch/runpillaga/wrfout_d02_2023-12-19_01:00:00"
]

wrf_in = [Dataset(f) for f in wrf_files]

# =====================================================
# GRID LOCATION
# =====================================================

xy = ll_to_xy(
    wrf_in[0],
    LAT_SITE,
    LON_SITE
)

ix = int(xy[0])
iy = int(xy[1])

print("Grid =", ix, iy)

# =====================================================
# TIME
# =====================================================

times = getvar(
    wrf_in,
    "times",
    timeidx=ALL_TIMES,
    method="cat"
)

wrf_times = pd.to_datetime(times.values)

wrf_times = wrf_times + pd.Timedelta(hours=UTC_OFFSET)

# =====================================================
# VARIABLES
# =====================================================

theta = getvar(
    wrf_in,
    "theta",
    timeidx=ALL_TIMES,
    method="cat"
)

z = getvar(
    wrf_in,
    "height_agl",
    timeidx=ALL_TIMES,
    method="cat"
)

w = getvar(
    wrf_in,
    "wa",
    timeidx=ALL_TIMES,
    method="cat"
)

pblh = getvar(
    wrf_in,
    "PBLH",
    timeidx=ALL_TIMES,
    method="cat"
)

# =====================================================
# TKE
# =====================================================

if "QKE" in wrf_in[0].variables:

    print("Using QKE")

    qke = getvar(
        wrf_in,
        "QKE",
        timeidx=ALL_TIMES,
        method="cat"
    )

    tke = qke / 2.0

elif "TKE" in wrf_in[0].variables:

    print("Using TKE")

    tke = getvar(
        wrf_in,
        "TKE",
        timeidx=ALL_TIMES,
        method="cat"
    )

else:

    print("No TKE available")

    tke = None

# =====================================================
# COLUMN EXTRACTION
# =====================================================

theta_col = theta[:, :, iy, ix].values

z_col = z[:, :, iy, ix].values

w_col = w[:, :, iy, ix].values

pblh_col = pblh[:, iy, ix].values

if tke is not None:
    tke_col = tke[:, :, iy, ix].values
else:
    tke_col = np.full_like(theta_col, np.nan)

# =====================================================
# THETA ANOMALY
# =====================================================

theta0 = theta_col[:, 0]

theta_anom = theta_col - theta0[:, None]

# =====================================================
# TIBL DIAGNOSIS
# =====================================================

def diagnose_tibl(th, zz, tkeprof):

    n = min(
        len(th),
        len(zz),
        len(tkeprof)
    )

    th = th[:n]
    zz = zz[:n]
    tkeprof = tkeprof[:n]

    mask = (
        (zz < 1500)
        &
        (tkeprof > 0.05)
    )

    if np.sum(mask) < 5:
        return np.nan

    idx = np.where(mask)[0][-1]

    return zz[idx]

tibl = []

for t in range(theta_col.shape[0]):

    tibl.append(
        diagnose_tibl(
            theta_col[t],
            z_col[t],
            tke_col[t]
        )
    )

tibl = np.array(tibl)

# =====================================================
# CEILOMETER
# =====================================================

csv_files = sorted([
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
])

dfs = []

for f in csv_files:

    df = pd.read_csv(f)

    df["Time"] = pd.to_datetime(
        df["# Time"],
        dayfirst=True
    )

    dfs.append(df)

ceilo = pd.concat(dfs)

ceilo["bl_height"] = pd.to_numeric(
    ceilo["bl_height"],
    errors="coerce"
)

ceilo_hourly = (
    ceilo
    .set_index("Time")
    .resample("1H")
    .median(numeric_only=True)
)

# =====================================================
# 4-PANEL PLOT
# =====================================================

fig, axs = plt.subplots(
    4,
    1,
    figsize=(16,18),
    sharex=True
)

# -----------------------------------------------------
# PANEL 1
# -----------------------------------------------------

cf = axs[0].contourf(
    wrf_times,
    z_col[0],
    theta_anom.T,
    levels=np.arange(0,15,0.5),
    extend="both"
)

plt.colorbar(
    cf,
    ax=axs[0],
    label="Theta anomaly (K)"
)

axs[0].plot(
    wrf_times,
    pblh_col,
    "k",
    lw=2
)

axs[0].plot(
    wrf_times,
    tibl,
    "w--",
    lw=2
)

axs[0].set_ylim(0,2500)
axs[0].set_title("Potential Temperature Structure")

# -----------------------------------------------------
# PANEL 2
# -----------------------------------------------------

cf = axs[1].contourf(
    wrf_times,
    z_col[0],
    tke_col.T,
    levels=np.arange(0,3,0.1),
    extend="max"
)

plt.colorbar(
    cf,
    ax=axs[1],
    label="TKE"
)

axs[1].plot(
    wrf_times,
    tibl,
    "w--",
    lw=2
)

axs[1].set_ylim(0,2500)
axs[1].set_title("TKE")

# -----------------------------------------------------
# PANEL 3
# -----------------------------------------------------

cf = axs[2].contourf(
    wrf_times,
    z_col[0],
    w_col.T,
    levels=np.arange(-2,2.1,0.1),
    extend="both"
)

plt.colorbar(
    cf,
    ax=axs[2],
    label="w (m/s)"
)

axs[2].plot(
    wrf_times,
    tibl,
    "k--",
    lw=2
)

axs[2].set_ylim(0,2500)
axs[2].set_title("Vertical Velocity")

# -----------------------------------------------------
# PANEL 4
# -----------------------------------------------------

axs[3].plot(
    wrf_times,
    pblh_col,
    label="WRF PBLH",
    lw=2
)

axs[3].plot(
    wrf_times,
    tibl,
    label="Diagnosed TIBL",
    lw=2
)

axs[3].plot(
    ceilo_hourly.index,
    ceilo_hourly["bl_height"],
    label="Ceilometer",
    alpha=0.7
)

axs[3].set_ylabel("Height (m)")
axs[3].legend()
axs[3].grid()

plt.tight_layout()

plt.savefig(
    "Lidcombe_TIBL_4panel.png",
    dpi=300
)

plt.show()

