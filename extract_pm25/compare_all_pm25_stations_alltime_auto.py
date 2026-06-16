#!/usr/bin/env python3

from netCDF4 import Dataset
from wrf import getvar, ll_to_xy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ==========================================================
# INPUT FILES
# ==========================================================

obs_file = "airqualpm25.csv"

station_file = (
    "air-quality-monitoring-sites-summary-9-feb-2026.csv"
)

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

UTC_OFFSET = 10     # observations reported in AEST

station_results = {}

# ==========================================================
# READ OBSERVATIONS
# ==========================================================

obs = pd.read_csv(obs_file)

obs["datetime"] = pd.to_datetime(
    obs["Date"],
    dayfirst=True
)

obs = obs.set_index("datetime")

# ==========================================================
# FIND PM2.5 STATIONS
# ==========================================================

pm25_columns = []

for c in obs.columns:

    if "PM2.5" in c:
        pm25_columns.append(c)

print()
print("Stations in observation file:")
print(pm25_columns)

# ==========================================================
# READ STATION COORDINATES
# ==========================================================

#site_df = pd.read_csv(station_file)
site_df = pd.read_csv(
    station_file,
    encoding="latin1"
)

site_df["STATION"] = (
    site_df["NSW air quality monitoring (AQMN) site"]
    .astype(str)
    .str.upper()
    .str.strip()
)

coord_lookup = {}

for _, row in site_df.iterrows():

    try:

        coord_lookup[row["STATION"]] = (
            float(row["Latitude\n(South)"]),
            float(row["Longitude\n(East)"])
        )

    except:
        pass

print()
print("Coordinate stations loaded:",
      len(coord_lookup))

# ==========================================================
# BUILD WRF TIME SERIES ONCE
# ==========================================================

wrf_cache = {}

for station_col in pm25_columns:

    station = (
        station_col
        .replace(" PM2.5","")
        .replace(" 1h","")
        .replace(".1","")
        .strip()
        .upper()
    )

    if station not in coord_lookup:

        print("Skipping:", station)
        continue

    lat_site, lon_site = coord_lookup[station]

    wrf_times = []
    wrf_pm25 = []

    print("Processing:", station)

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

        nt = len(times)

        for t in range(nt):

            wrf_times.append(
                pd.to_datetime(
                    str(times[t].values)
                )
            )

            wrf_pm25.append(
                float(
                    pm25[t,0,iy,ix]
                )
            )

    wrf_times = (
        pd.to_datetime(wrf_times)
        + pd.Timedelta(hours=UTC_OFFSET)
    )

    wrf_df = pd.DataFrame({
        "PM25_WRF": wrf_pm25
    }, index=wrf_times)

    wrf_cache[station] = wrf_df

# ==========================================================
# COMPARE ALL STATIONS
# ==========================================================

results = []

for station_col in pm25_columns:

    station = (
        station_col
        .replace(" PM2.5","")
        .replace(" 1h","")
        .replace(".1","")
        .strip()
        .upper()
    )

    if station not in wrf_cache:
        continue

    obs_station = pd.to_numeric(
        obs[station_col],
        errors="coerce"
    )

    obs_station[obs_station < 0] = 0

    obs_df = pd.DataFrame({
        "PM25_OBS": obs_station
    })

    compare = pd.merge(
        wrf_cache[station],
        obs_df,
        left_index=True,
        right_index=True,
        how="inner"
    )

    compare = compare.dropna()
    
    station_results[station] = compare.copy()


    if len(compare) < 10:
        continue

    wrfv = compare["PM25_WRF"].values
    obsv = compare["PM25_OBS"].values

    r = np.corrcoef(
        obsv,
        wrfv
    )[0,1]

    mb = np.mean(wrfv - obsv)

    mae = np.mean(
        np.abs(wrfv - obsv)
    )

    rmse = np.sqrt(
        np.mean((wrfv-obsv)**2)
    )

    nmb = (
        100 *
        np.sum(wrfv-obsv)
        / np.sum(obsv)
    )

    lat, lon = coord_lookup[station]

    results.append([
        station,
        lat,
        lon,
        len(compare),
        r,
        mb,
        mae,
        rmse,
        nmb
    ])

# ==========================================================
# SAVE TABLE
# ==========================================================

summary = pd.DataFrame(
    results,
    columns=[
        "Station",
        "Lat",
        "Lon",
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
print(summary)

# =====================================================
# PANEL TIME SERIES PLOTS
# =====================================================

import matplotlib.dates as mdates

# Sydney stations

#stations_to_plot = [
#    "ROZELLE",
#    "EARLWOOD",
#    "RANDWICK",
#    "LIDCOMBE",
#    "CAMDEN",
#    "RICHMOND",
#    "OAKDALE",
#    "BARGO",
#    "PROSPECT"
#]
# Sations in Lower Hunter, Central Coast and Illawarra

stations_to_plot = [
    "MERRIWA",
    "SINGLETON",
    "NEWCASTLE",
    "WALLSEND",
    "BERESFIELD",
    "WYONG",
    "MORISSET",
    "WOLLONGONG",
    "ALBION PARK SOUTH"
]

fig, axes = plt.subplots(
    3,
    3,
    figsize=(18,12),
    sharex=True
)

axes = axes.flatten()

for n, station in enumerate(stations_to_plot):

    ax = axes[n]

    if station not in station_results:

        ax.set_title(
            station + "\nNo data"
        )

        continue

    compare = station_results[station]

    obs = compare["PM25_OBS"].values
    wrf = compare["PM25_WRF"].values

    if len(obs) < 2:

        ax.set_title(
            station + "\nInsufficient data"
        )

        continue

    r = np.corrcoef(obs, wrf)[0,1]

    mb = np.mean(wrf - obs)

    rmse = np.sqrt(
        np.mean((wrf - obs)**2)
    )

    ax.plot(
        compare.index,
        compare["PM25_OBS"],
        color="black",
        linewidth=2,
        label="Obs"
    )

    ax.plot(
        compare.index,
        compare["PM25_WRF"],
        color="red",
        linewidth=2,
        label="WRF-Chem"
    )

    ax.set_title(
        f"{station}\n"
        f"R={r:.2f} "
        f"MB={mb:.1f} "
        f"RMSE={rmse:.1f}"
    )

    ax.grid(True)

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%d-%b\n%H:%M")
    )

for ax in axes:
    ax.tick_params(
        axis="x",
        rotation=30
    )

axes[0].legend(
    loc="upper right"
)

#fig.suptitle(
#    #"WRF-Chem vs Observed PM2.5\n06-19 Dec 2023",
#    "WRF-Chem vs Observed PM2.5 06-19 Dec 2023",
#    fontsize=16
#)

plt.tight_layout()

plt.savefig(
    "PM25_panel_timeseries_gmr.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ==========================================================
# NSW MAP OF CORRELATION
# ==========================================================

fig = plt.figure(figsize=(12,10))

ax = plt.axes(
    projection=ccrs.PlateCarree()
)

ax.set_extent(
    [140,154,-38,-27]
)

ax.add_feature(
    cfeature.LAND,
    facecolor="lightgray"
)

ax.add_feature(
    cfeature.OCEAN,
    facecolor="white"
)

ax.coastlines("10m")

sc = ax.scatter(
    summary["Lon"],
    summary["Lat"],
    c=summary["R"],
    cmap="RdYlGn",
    s=120,
    edgecolor="black",
    vmin=0,
    vmax=1,
    transform=ccrs.PlateCarree()
)

for _, row in summary.iterrows():

    ax.text(
        row["Lon"] + 0.1,
        row["Lat"] + 0.05,
        row["Station"],
        fontsize=7,
        transform=ccrs.PlateCarree()
    )

plt.colorbar(
    sc,
    label="Correlation coefficient (R)"
)

plt.title(
    "WRF-Chem vs Observed PM2.5\n16-19 Dec 2023"
)

plt.tight_layout()

plt.savefig(
    "NSW_PM25_correlation_map.png",
    dpi=300
)

plt.show()

