#!/usr/bin/env python3

from netCDF4 import Dataset
from wrf import getvar, ll_to_xy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ======================================================
# MERRIWA CEILOMETER LOCATION
# ======================================================

LAT_SITE = -32.14
LON_SITE = 150.36

# ======================================================
# WRF FILES
# ======================================================

wrf_files = [
    "/mnt/scratch_lustre/duch/runpillaga/wrfout_d01_2023-12-06_00:00:00",
    "/mnt/scratch_lustre/duch/runpillaga/wrfout_d01_2023-12-07_00:00:00",
    "/mnt/scratch_lustre/duch/runpillaga/wrfout_d01_2023-12-08_00:00:00",
    "/mnt/scratch_lustre/duch/runpillaga/wrfout_d01_2023-12-09_00:00:00",
    "/mnt/scratch_lustre/duch/runpillaga/wrfout_d01_2023-12-10_00:00:00",
    "/mnt/scratch_lustre/duch/runpillaga/wrfout_d01_2023-12-11_00:00:00",
    "/mnt/scratch_lustre/duch/runpillaga/wrfout_d01_2023-12-12_00:00:00",
    "/mnt/scratch_lustre/duch/runpillaga/wrfout_d01_2023-12-13_00:00:00",
    "/mnt/scratch_lustre/duch/runpillaga/wrfout_d01_2023-12-14_01:00:00",
    "/mnt/scratch_lustre/duch/runpillaga/wrfout_d01_2023-12-15_01:00:00",
    "/mnt/scratch_lustre/duch/runpillaga/wrfout_d01_2023-12-16_01:00:00",
    "/mnt/scratch_lustre/duch/runpillaga/wrfout_d01_2023-12-17_01:00:00",
    "/mnt/scratch_lustre/duch/runpillaga/wrfout_d01_2023-12-18_01:00:00",
    "/mnt/scratch_lustre/duch/runpillaga/wrfout_d01_2023-12-19_01:00:00"
]

wrf_times = []
wrf_pblh = []

for wrf_file in wrf_files:

    print("Reading", wrf_file)

    nc = Dataset(wrf_file)

    xy = ll_to_xy(
        nc,
        LAT_SITE,
        LON_SITE
    )

    ix = int(xy[0])
    iy = int(xy[1])

    pblh = getvar(
        nc,
        "PBLH",
        timeidx=None
    )

    times = getvar(
        nc,
        "times",
        timeidx=None
    )

    nt = len(times)

    for t in range(nt):

        wrf_times.append(
            pd.to_datetime(
                str(times[t].values)
            )
        )

        wrf_pblh.append(
            float(
                pblh[t, iy, ix]
            )
        )

# ======================================================
# UTC -> AEST
# ======================================================

wrf_times = (
    pd.to_datetime(wrf_times)
    + pd.Timedelta(hours=10)
)

wrf_df = pd.DataFrame({
    "time": wrf_times,
    "PBLH_WRF": wrf_pblh
})

wrf_df = wrf_df.set_index("time")

print()
print("WRF records =", len(wrf_df))

# ======================================================
# CEILOMETER FILES
# ======================================================

csv_files = [
    "ceilo_data/L3_DEFAULT_0_20231206_Merriwa.csv",
    "ceilo_data/L3_DEFAULT_0_20231207_Merriwa.csv",
    "ceilo_data/L3_DEFAULT_0_20231208_Merriwa.csv",
    "ceilo_data/L3_DEFAULT_0_20231209_Merriwa.csv",
    "ceilo_data/L3_DEFAULT_0_20231210_Merriwa.csv",
    "ceilo_data/L3_DEFAULT_0_20231211_Merriwa.csv",
    "ceilo_data/L3_DEFAULT_0_20231212_Merriwa.csv",
    "ceilo_data/L3_DEFAULT_0_20231213_Merriwa.csv",
    "ceilo_data/L3_DEFAULT_0_20231214_Merriwa.csv",
    "ceilo_data/L3_DEFAULT_0_20231215_Merriwa.csv",
    "ceilo_data/L3_DEFAULT_0_20231216_Merriwa.csv",
    "ceilo_data/L3_DEFAULT_0_20231217_Merriwa.csv",
    "ceilo_data/L3_DEFAULT_0_20231218_Merriwa.csv",
    "ceilo_data/L3_DEFAULT_0_20231219_Merriwa.csv",
    "ceilo_data/L3_DEFAULT_0_20231220_Merriwa.csv"
]

dfs = []

for f in csv_files:

    print("Reading", f)

    df = pd.read_csv(f)

    df["Time"] = pd.to_datetime(
        df["# Time"],
        dayfirst=True
    )

    df["bl_height"] = pd.to_numeric(
        df["bl_height"],
        errors="coerce"
    )

    # remove invalid values
    df.loc[
        df["bl_height"] < 0,
        "bl_height"
    ] = np.nan

    dfs.append(df)

ceilo = pd.concat(
    dfs,
    ignore_index=True
)

ceilo = ceilo.sort_values(
    "Time"
)

# ======================================================
# CEILOMETER PBLH
# ======================================================

# ------------------------------------------------------
# 20-minute median for plotting
# ------------------------------------------------------

ceilo_20min = (
    ceilo
    .set_index("Time")
    .resample("20min")
    .median(numeric_only=True)
)

# ------------------------------------------------------
# Smooth 20-minute series
# (3 x 20 min = ~1 hour smoothing)
# ------------------------------------------------------

ceilo_20min["bl_height_smooth"] = (
    ceilo_20min["bl_height"]
    .rolling(
        window=3,
        center=True,
        min_periods=1
    )
    .median()
)

# ------------------------------------------------------
# Hourly median for statistics
# ------------------------------------------------------

ceilo_hourly = (
    ceilo
    .set_index("Time")
    .resample("1H")
    .median(numeric_only=True)
)

print(
    "Ceilometer 20-min records =",
    len(ceilo_20min)
)

print(
    "Ceilometer hourly records =",
    len(ceilo_hourly)
)

# ======================================================
# MATCH TIMES
# ======================================================

compare = pd.concat(
    [
        wrf_df["PBLH_WRF"],
        ceilo_hourly["bl_height"]
    ],
    axis=1
)

compare.columns = [
    "WRF_PBLH",
    "CEILO_PBLH"
]

compare = compare.dropna()

print()
print(compare.head())
print()
print("Matched hours =", len(compare))

# ======================================================
# STATISTICS
# ======================================================

wrf = compare["WRF_PBLH"].values
obs = compare["CEILO_PBLH"].values

r = np.corrcoef(
    obs,
    wrf
)[0,1]

mb = np.mean(
    wrf - obs
)

mae = np.mean(
    np.abs(wrf - obs)
)

rmse = np.sqrt(
    np.mean(
        (wrf - obs)**2
    )
)

nmb = (
    100.0 *
    np.sum(wrf - obs)
    /
    np.sum(obs)
)

print()
print("================================")
print("PBLH MODEL EVALUATION")
print("================================")
print(f"N    = {len(obs)}")
print(f"R    = {r:.3f}")
print(f"MB   = {mb:.1f} m")
print(f"MAE  = {mae:.1f} m")
print(f"RMSE = {rmse:.1f} m")
print(f"NMB  = {nmb:.1f} %")
print("================================")

# ======================================================
# TIME SERIES
# ======================================================

plt.figure(
    figsize=(15,6)
)

plt.scatter(
    ceilo["Time"],
    ceilo["bl_height"],
    s=1,
    alpha=0.15,
    label="Ceilometer Raw"
)

plt.plot(
    ceilo_hourly.index,
    ceilo_hourly["bl_height"],
    linewidth=2,
    label="Ceilometer Hourly"
)

plt.plot(
    wrf_df.index,
    wrf_df["PBLH_WRF"],
    "-o",
    markersize=4,
    linewidth=2,
    label="WRF-Chem"
)

stats_text = (
    f"R = {r:.2f}\n"
    f"MB = {mb:.0f} m\n"
    f"RMSE = {rmse:.0f} m"
)

plt.text(
    0.02,
    0.98,
    stats_text,
    transform=plt.gca().transAxes,
    va="top",
    bbox=dict(facecolor="white")
)

plt.ylabel(
    "PBL Height (m)"
)

plt.xlabel(
    "Date / Time (AEST)"
)

plt.title(
    "Merriwa Ceilometer vs WRF-Chem PBLH"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()

# ======================================================
# SCATTER PLOT
# ======================================================

plt.figure(
    figsize=(7,7)
)

plt.scatter(
    obs,
    wrf,
    alpha=0.6
)

mx = max(
    obs.max(),
    wrf.max()
)

plt.plot(
    [0,mx],
    [0,mx],
    'k--',
    linewidth=2,
    label="1:1"
)

# regression line
m, b = np.polyfit(
    obs,
    wrf,
    1
)

x = np.linspace(
    0,
    mx,
    100
)

plt.plot(
    x,
    m*x+b,
    'r-',
    linewidth=2,
    label=f"y={m:.2f}x+{b:.0f}"
)

plt.xlabel(
    "Ceilometer PBLH (m)"
)

plt.ylabel(
    "WRF-Chem PBLH (m)"
)

plt.title(
    f"Merriwa PBLH\nR={r:.2f} MB={mb:.0f} m"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()

plt.figure(
    figsize=(15,6)
)

# ------------------------------------------------------
# Raw ceilometer retrievals
# ------------------------------------------------------

plt.scatter(
    ceilo["Time"],
    ceilo["bl_height"],
    s=1,
    alpha=0.10,
    color="lightgrey",
    label="Ceilometer Raw"
)

# ------------------------------------------------------
# Smoothed 20-minute ceilometer
# ------------------------------------------------------

plt.plot(
    ceilo_20min.index,
    ceilo_20min["bl_height_smooth"],
    linewidth=2,
    label="Ceilometer (20 min smoothed)"
)

# ------------------------------------------------------
# Hourly WRF-Chem PBLH
# ------------------------------------------------------

plt.plot(
    wrf_df.index,
    wrf_df["PBLH_WRF"],
    "-o",
    markersize=4,
    linewidth=2,
    label="WRF-Chem"
)

plt.title(
    "Merriwa Ceilometer vs WRF-Chem PBLH"
)

plt.show()


