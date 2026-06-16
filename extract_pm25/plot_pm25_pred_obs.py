#!/usr/bin/env python3

from netCDF4 import Dataset
from wrf import getvar, ll_to_xy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# USER SETTINGS
# =====================================================

STATION_NAME = "MERRIWA"

LAT_SITE = -32.14
LON_SITE = 150.36

PM25_COLUMN = "MERRIWA PM2.5"

wrf_files = [
    "/mnt/scratch_lustre/duch/runpillaga/wrfout_d01_2023-12-16_00:00:00.nc",
    "/mnt/scratch_lustre/duch/runpillaga/wrfout_d01_2023-12-17_00:00:00.nc",
    "/mnt/scratch_lustre/duch/runpillaga/wrfout_d01_2023-12-18_00:00:00.nc"
]

obs_file = "airqualpm25.csv"

# NSW EPA file appears to be AEST
UTC_OFFSET = 10

# =====================================================
# READ WRF
# =====================================================

wrf_times = []
wrf_pm25 = []

for wrf_file in wrf_files:

    print("Reading:", wrf_file)

    nc = Dataset(wrf_file)

    xy = ll_to_xy(
        nc,
        LAT_SITE,
        LON_SITE
    )

    ix = int(xy[0])
    iy = int(xy[1])

    pm25 = getvar(
        nc,
        "PM2_5_DRY",
        timeidx=None
    )

    times = getvar(
        nc,
        "times",
        timeidx=None
    )

    for t in range(len(times)):

        wrf_times.append(
            pd.to_datetime(
                str(times[t].values)
            )
        )

        wrf_pm25.append(
            float(pm25[t,0,iy,ix])
        )

# -----------------------------------------------------
# Convert WRF UTC -> AEST
# -----------------------------------------------------

wrf_times = (
    pd.to_datetime(wrf_times)
    + pd.Timedelta(hours=UTC_OFFSET)
)

wrf_df = pd.DataFrame({
    "PM25_WRF": wrf_pm25
}, index=wrf_times)

# =====================================================
# READ OBSERVATIONS
# =====================================================

obs = pd.read_csv(obs_file)

obs["datetime"] = pd.to_datetime(
    obs["Date"],
    dayfirst=True
)

obs = obs.set_index("datetime")

obs_pm25 = pd.to_numeric(
    obs[PM25_COLUMN],
    errors="coerce"
)

# Replace negatives with zero
obs_pm25 = obs_pm25.clip(lower=0)

obs_df = pd.DataFrame({
    "PM25_OBS": obs_pm25
})

# =====================================================
# MERGE
# =====================================================

compare = pd.merge(
    wrf_df,
    obs_df,
    left_index=True,
    right_index=True,
    how="inner"
)

compare = compare.dropna()

# remove impossible values
compare = compare[
    (compare["PM25_WRF"] >= 0) &
    (compare["PM25_OBS"] >= 0)
]

print()
print("Matched points =", len(compare))
print("Start =", compare.index.min())
print("End   =", compare.index.max())

# =====================================================
# STATISTICS
# =====================================================

wrf = compare["PM25_WRF"].values
obs = compare["PM25_OBS"].values

r = np.corrcoef(obs, wrf)[0,1]

mb = np.mean(wrf - obs)

mae = np.mean(np.abs(wrf - obs))

rmse = np.sqrt(
    np.mean((wrf - obs)**2)
)

nmb = (
    100.0 *
    np.sum(wrf - obs) /
    np.sum(obs)
)

print()
print("================================")
print("Station =", STATION_NAME)
print("N        =", len(obs))
print("R        =", round(r,3))
print("MB       =", round(mb,2), "ug/m3")
print("MAE      =", round(mae,2), "ug/m3")
print("RMSE     =", round(rmse,2), "ug/m3")
print("NMB      =", round(nmb,1), "%")
print("================================")

# =====================================================
# TIME SERIES
# =====================================================

plt.figure(figsize=(14,6))

plt.plot(
    compare.index,
    compare["PM25_OBS"],
    label="Observed",
    linewidth=2
)

plt.plot(
    compare.index,
    compare["PM25_WRF"],
    label="WRF-Chem",
    linewidth=2
)

stats_text = (
    f"R={r:.2f}\n"
    f"MB={mb:.1f}\n"
    f"RMSE={rmse:.1f}"
)

plt.text(
    0.02,
    0.98,
    stats_text,
    transform=plt.gca().transAxes,
    va="top",
    bbox=dict(facecolor="white")
)

plt.ylabel("PM2.5 (ug m-3)")
plt.xlabel("AEST")

plt.title(
    f"{STATION_NAME} PM2.5 Comparison"
)

plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig(
    f"{STATION_NAME}_PM25_timeseries.png",
    dpi=300
)

plt.show()

# =====================================================
# SCATTER
# =====================================================

plt.figure(figsize=(7,7))

plt.scatter(
    obs,
    wrf,
    alpha=0.6
)

mx = max(
    np.nanmax(obs),
    np.nanmax(wrf)
)

plt.plot(
    [0,mx],
    [0,mx],
    "k--"
)

plt.xlabel("Observed PM2.5")
plt.ylabel("WRF-Chem PM2.5")

plt.title(
    f"{STATION_NAME}\nR={r:.2f}  RMSE={rmse:.1f}"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    f"{STATION_NAME}_PM25_scatter.png",
    dpi=300
)

plt.show()

