#!/usr/bin/env python3

from netCDF4 import Dataset
from wrf import getvar, ll_to_xy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================================
# USER SETTINGS
# ==========================================================

obs_file = "airqualpm25.csv"

wrf_files = [
    "/mnt/scratch_lustre/duch/runpillaga/wrfout_d01_2023-12-16_00:00:00.nc",
    "/mnt/scratch_lustre/duch/runpillaga/wrfout_d01_2023-12-17_00:00:00.nc",
    "/mnt/scratch_lustre/duch/runpillaga/wrfout_d01_2023-12-18_00:00:00.nc"
]

# Observation timestamps are AEST (+10)
UTC_OFFSET = 10

# ==========================================================
# STATION INFORMATION
# Replace with official EPA coordinates if available
# ==========================================================

stations = {

    "MERRIWA":
        (-32.14, 150.36, "MERRIWA PM2.5"),

    "SINGLETON":
        (-32.56, 151.17, "SINGLETON PM2.5"),

    "MUSWELLBROOK":
        (-32.27, 150.89, "MUSWELLBROOK PM2.5"),

    "CAMBERWELL":
        (-32.21, 151.04, "CAMBERWELL PM2.5"),

    "NEWCASTLE":
        (-32.93, 151.78, "NEWCASTLE PM2.5"),

    "BERESFIELD":
        (-32.80, 151.63, "BERESFIELD PM2.5"),

    "MAYFIELD":
        (-32.89, 151.74, "MAYFIELD PM2.5"),

    "WALLSEND":
        (-32.90, 151.67, "WALLSEND PM2.5"),

    "MORISSET":
        (-33.11, 151.49, "MORISSET PM2.5"),

    "WYONG":
        (-33.28, 151.42, "WYONG PM2.5"),

    "LIVERPOOL":
        (-33.92, 150.92, "LIVERPOOL PM2.5"),

    "LIDCOMBE":
        (-33.86, 151.04, "LIDCOMBE PM2.5"),

    "PROSPECT":
        (-33.80, 150.91, "PROSPECT PM2.5"),

    "RICHMOND":
        (-33.61, 150.75, "RICHMOND PM2.5"),

    "ST_MARYS":
        (-33.76, 150.77, "ST MARYS PM2.5"),

    "PENRITH":
        (-33.75, 150.69, "PENRITH PM2.5"),

    "BRINGELLY":
        (-33.93, 150.73, "BRINGELLY PM2.5"),

    "CAMDEN":
        (-34.05, 150.70, "CAMDEN PM2.5"),

    "CAMPBELLTOWN":
        (-34.08, 150.80, "CAMPBELLTOWN WEST PM2.5"),

    "EARLWOOD":
        (-33.92, 151.13, "EARLWOOD PM2.5"),

    "RANDWICK":
        (-33.91, 151.24, "RANDWICK PM2.5"),

    "ROZELLE":
        (-33.86, 151.17, "ROZELLE PM2.5"),

    "BARGO":
        (-34.40, 150.54, "BARGO PM2.5"),

    "OAKDALE":
        (-34.08, 150.51, "OAKDALE PM2.5"),

    "WOLLONGONG":
        (-34.43, 150.89, "WOLLONGONG PM2.5"),

    "ALBION_PARK":
        (-34.58, 150.79, "ALBION PARK SOUTH PM2.5")
}

# ==========================================================
# READ OBSERVATIONS
# ==========================================================

print("Reading observations...")

obs = pd.read_csv(obs_file)

obs["datetime"] = pd.to_datetime(
    obs["Date"],
    dayfirst=True
)

obs = obs.set_index("datetime")

# ==========================================================
# PROCESS EACH STATION
# ==========================================================

results = []

os.makedirs("timeseries_plots", exist_ok=True)

for station in stations:

    lat, lon, obs_column = stations[station]

    if obs_column not in obs.columns:

        print("Missing column:", obs_column)
        continue

    print("Processing", station)

    wrf_times = []
    wrf_pm25 = []

    # ------------------------------------------
    # Extract WRF PM2.5
    # ------------------------------------------

    for wrf_file in wrf_files:

        nc = Dataset(wrf_file)

        xy = ll_to_xy(
            nc,
            lat,
            lon
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

    wrf_times = (
        pd.to_datetime(wrf_times)
        + pd.Timedelta(hours=UTC_OFFSET)
    )

    wrf_df = pd.DataFrame({
        "PM25_WRF": wrf_pm25
    }, index=wrf_times)

    # ------------------------------------------
    # Observations
    # ------------------------------------------

    obs_station = pd.to_numeric(
        obs[obs_column],
        errors="coerce"
    )

    obs_station[obs_station < 0] = 0.0

    obs_df = pd.DataFrame({
        "PM25_OBS": obs_station
    })

    # ------------------------------------------
    # Merge
    # ------------------------------------------

    compare = pd.merge(
        wrf_df,
        obs_df,
        left_index=True,
        right_index=True,
        how="inner"
    )

    compare = compare.dropna()

    if len(compare) < 10:

        print("Too few points")
        continue

    wrfv = compare["PM25_WRF"].values
    obsv = compare["PM25_OBS"].values

    r = np.corrcoef(obsv, wrfv)[0,1]

    mb = np.mean(wrfv - obsv)

    mae = np.mean(np.abs(wrfv - obsv))

    rmse = np.sqrt(
        np.mean(
            (wrfv - obsv)**2
        )
    )

    nmb = (
        100.0 *
        np.sum(wrfv - obsv) /
        np.sum(obsv)
    )

    results.append([
        station,
        len(compare),
        r,
        mb,
        mae,
        rmse,
        nmb
    ])

    # ------------------------------------------
    # Time series plot
    # ------------------------------------------

    plt.figure(figsize=(12,5))

    plt.plot(
        compare.index,
        compare["PM25_OBS"],
        label="Observed"
    )

    plt.plot(
        compare.index,
        compare["PM25_WRF"],
        label="WRF-Chem"
    )

    plt.title(
        f"{station}\n"
        f"R={r:.2f}  RMSE={rmse:.1f}"
    )

    plt.ylabel("PM2.5 (ug m-3)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        f"timeseries_plots/{station}.png",
        dpi=200
    )

    plt.close()

# ==========================================================
# SAVE SUMMARY
# ==========================================================

summary = pd.DataFrame(
    results,
    columns=[
        "Station",
        "N",
        "R",
        "MB",
        "MAE",
        "RMSE",
        "NMB"
    ]
)

summary = summary.sort_values(
    "R",
    ascending=False
)

summary.to_csv(
    "PM25_station_statistics.csv",
    index=False
)

print()
print(summary.round(2))

print()
print("Saved:")
print("  PM25_station_statistics.csv")
print("  timeseries_plots/*.png")
