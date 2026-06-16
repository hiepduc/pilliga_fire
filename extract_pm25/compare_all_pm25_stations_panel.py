#!/usr/bin/env python3

from netCDF4 import Dataset
from wrf import getvar, ll_to_xy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# USER SETTINGS
# =====================================================

obs_file = "airqualpm25.csv"

wrf_files = [
    "/mnt/scratch_lustre/duch/runpillaga/wrfout_d01_2023-12-16_00:00:00.nc",
    "/mnt/scratch_lustre/duch/runpillaga/wrfout_d01_2023-12-17_00:00:00.nc",
    "/mnt/scratch_lustre/duch/runpillaga/wrfout_d01_2023-12-18_00:00:00.nc",
    "/mnt/scratch_lustre/duch/runpillaga/wrfout_d01_2023-12-19_00:00:00.nc"
]

# WRF output UTC → AEST
UTC_OFFSET = 10

# =====================================================
# STATIONS
# =====================================================

stations = {

    "MERRIWA":
        {"lat":-32.14, "lon":150.36,
         "col":"MERRIWA PM2.5"},

    "LIVERPOOL":
        {"lat":-33.92, "lon":150.92,
         "col":"LIVERPOOL PM2.5"},

    "PROSPECT":
        {"lat":-33.80, "lon":150.92,
         "col":"PROSPECT PM2.5"},

    "RICHMOND":
        {"lat":-33.60, "lon":150.75,
         "col":"RICHMOND PM2.5"},

    "CAMDEN":
        {"lat":-34.05, "lon":150.69,
         "col":"CAMDEN PM2.5"},

    "PENRITH":
        {"lat":-33.75, "lon":150.69,
         "col":"PENRITH PM2.5"},

    "RANDWICK":
        {"lat":-33.92, "lon":151.24,
         "col":"RANDWICK PM2.5"},

    "ROZELLE":
        {"lat":-33.86, "lon":151.18,
         "col":"ROZELLE PM2.5"},

    "WOLLONGONG":
        {"lat":-34.43, "lon":150.89,
         "col":"WOLLONGONG PM2.5"}
}

# =====================================================
# READ OBS
# =====================================================

obs = pd.read_csv(obs_file)

obs["datetime"] = pd.to_datetime(
    obs["Date"],
    dayfirst=True
)

obs = obs.set_index("datetime")

# =====================================================
# PANEL PLOT
# =====================================================

fig, axes = plt.subplots(
    3, 3,
    figsize=(18,12),
    sharex=True
)

axes = axes.flatten()

# =====================================================
# LOOP STATIONS
# =====================================================

for n, (station, info) in enumerate(stations.items()):

    print()
    print("Processing", station)

    lat_site = info["lat"]
    lon_site = info["lon"]
    pm_col = info["col"]

    wrf_times = []
    wrf_pm25 = []

    # -----------------------------------------
    # WRF extraction
    # -----------------------------------------

    for wrf_file in wrf_files:

        nc = Dataset(wrf_file)

        xy = ll_to_xy(
            nc,
            lat_site,
            lon_site
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

        ntimes = len(times)

        for t in range(ntimes):

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
        "time":wrf_times,
        "PM25_WRF":wrf_pm25
    })

    wrf_df = wrf_df.set_index("time")

    # -----------------------------------------
    # OBS
    # -----------------------------------------

    if pm_col not in obs.columns:

        print("Missing column:", pm_col)
        continue

    obs_pm = pd.to_numeric(
        obs[pm_col],
        errors="coerce"
    )

    obs_pm[obs_pm < 0] = 0.0

    obs_df = pd.DataFrame({
        "PM25_OBS":obs_pm
    })

    # -----------------------------------------
    # MERGE
    # -----------------------------------------

    compare = pd.merge(
        wrf_df,
        obs_df,
        left_index=True,
        right_index=True,
        how="inner"
    )

    compare = compare.dropna()

    if len(compare) < 5:

        print("Too few data points")
        continue

    wrf = compare["PM25_WRF"].values
    obsvals = compare["PM25_OBS"].values

    r = np.corrcoef(
        obsvals,
        wrf
    )[0,1]

    rmse = np.sqrt(
        np.mean(
            (wrf-obsvals)**2
        )
    )

    mb = np.mean(
        wrf-obsvals
    )

    print(
        f"{station}: "
        f"R={r:.2f} "
        f"RMSE={rmse:.1f} "
        f"MB={mb:.1f}"
    )

    # -----------------------------------------
    # PLOT
    # -----------------------------------------

    ax = axes[n]

    ax.plot(
        compare.index,
        compare["PM25_OBS"],
        label="Obs",
        linewidth=1.5
    )

    ax.plot(
        compare.index,
        compare["PM25_WRF"],
        label="WRF",
        linewidth=1.5
    )

    ax.set_title(
        f"{station}\n"
        f"R={r:.2f}  "
        f"RMSE={rmse:.1f}"
    )

    ax.grid(True)

# =====================================================
# LEGEND
# =====================================================

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=2
)

plt.suptitle(
    "WRF-Chem vs Observed PM2.5 (AEST)",
    fontsize=16
)

plt.tight_layout(
    rect=[0,0,1,0.95]
)

plt.savefig(
    "PM25_9station_panel.png",
    dpi=300
)

plt.show()

