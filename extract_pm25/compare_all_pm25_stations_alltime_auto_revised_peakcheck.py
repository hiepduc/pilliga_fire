#!/usr/bin/env python3

from netCDF4 import Dataset
from wrf import getvar, ll_to_xy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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

UTC_OFFSET = 10

# ==========================================================
# READ OBS
# ==========================================================

obs = pd.read_csv(obs_file)

obs["datetime"] = pd.to_datetime(obs["Date"], dayfirst=True)
obs = obs.set_index("datetime")

pm25_columns = [c for c in obs.columns if "PM2.5" in c]

print("\nStations:")
print(pm25_columns)

# ==========================================================
# STATION COORDS
# ==========================================================

site_df = pd.read_csv(station_file, encoding="latin1")

site_df["STATION"] = (
    site_df["NSW air quality monitoring (AQMN) site"]
    .astype(str).str.upper().str.strip()
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

print("\nStations with coords:", len(coord_lookup))

# ==========================================================
# BUILD WRF TIME SERIES
# ==========================================================

wrf_cache = {}

for station_col in pm25_columns:

    station = (
        station_col.replace(" PM2.5","")
        .replace(" 1h","")
        .replace(".1","")
        .strip().upper()
    )

    if station not in coord_lookup:
        continue

    lat_site, lon_site = coord_lookup[station]

    wrf_times = []
    wrf_vals = []

    print("Processing:", station)

    for wrf_file in wrf_files:

        nc = Dataset(wrf_file)

        xy = ll_to_xy(nc, lat_site, lon_site)
        ix, iy = int(xy[0]), int(xy[1])

        pm25 = getvar(nc, "PM2_5_DRY", timeidx=None).values
        times = getvar(nc, "times", timeidx=None).values

        nt = pm25.shape[0]

        for t in range(nt):

            wrf_times.append(
                pd.to_datetime(str(times[t]))
            )

            wrf_vals.append(
                float(pm25[t,0,iy,ix])
            )


    wrf_times = pd.to_datetime(wrf_times) + pd.Timedelta(hours=UTC_OFFSET)

    wrf_cache[station] = pd.DataFrame(
        {"WRF": wrf_vals},
        index=wrf_times
    )

# ==========================================================
# COMPARE STATIONS
# ==========================================================

results = []
station_results = {}

for station_col in pm25_columns:

    station = (
        station_col.replace(" PM2.5","")
        .replace(" 1h","")
        .replace(".1","")
        .strip().upper()
    )

    if station not in wrf_cache:
        continue

    obs_series = pd.to_numeric(obs[station_col], errors="coerce")
    obs_series = obs_series.where(obs_series >= 0)

    obs_df = pd.DataFrame({"OBS": obs_series})

    # JOIN (safe alignment)
    compare = wrf_cache[station].join(obs_df, how="outer")

    station_results[station] = compare.copy()

    valid = compare.dropna(subset=["WRF", "OBS"])

    if len(valid) < 10:
        continue

    wrf = valid["WRF"].values
    obs_v = valid["OBS"].values

    if np.std(wrf) == 0 or np.std(obs_v) == 0:
        r = np.nan
    else:
        r = np.corrcoef(obs_v, wrf)[0,1]

    mb = np.mean(wrf - obs_v)
    mae = np.mean(np.abs(wrf - obs_v))
    rmse = np.sqrt(np.mean((wrf - obs_v)**2))

    nmb = 100 * np.sum(wrf - obs_v) / np.sum(obs_v)

    lat, lon = coord_lookup[station]

    results.append([
        station, lat, lon, len(valid),
        r, mb, mae, rmse, nmb
    ])

# ==========================================================
# SUMMARY TABLE
# ==========================================================

summary = pd.DataFrame(
    results,
    columns=["Station","Lat","Lon","N","R","MB","MAE","RMSE","NMB"]
)

summary = summary.dropna(subset=["R"])

summary = summary.sort_values("R", ascending=False)

summary.to_csv("PM25_station_statistics.csv", index=False)

print("\nSUMMARY:")
print(summary)

# ==========================================================
# PANEL PLOTS
# ==========================================================

stations_to_plot = [
    "ROZELLE","EARLWOOD","RANDWICK",
    "LIDCOMBE","CAMDEN","RICHMOND",
    "OAKDALE","BARGO","PROSPECT"
]

fig, axes = plt.subplots(3,3,figsize=(18,12),sharex=True)
axes = axes.flatten()

for i, station in enumerate(stations_to_plot):

    ax = axes[i]

    if station not in station_results:
        ax.set_title(station + "\nNo data")
        continue

    df = station_results[station]

    df_plot = df.dropna(subset=["WRF","OBS"])

    if len(df_plot) < 2:
        ax.set_title(station + "\nInsufficient data")
        continue

    r = np.corrcoef(df_plot["OBS"], df_plot["WRF"])[0,1]

    ax.plot(df_plot.index, df_plot["OBS"], "k", label="Obs")
    ax.plot(df_plot.index, df_plot["WRF"], "r", label="WRF")

    ax.set_title(
        f"{station}\n"
        f"R={r:.2f} "
        f"MB={mb:.1f} "
        f"RMSE={rmse:.1f}"
    )
    #ax.set_title(f"{station}\nR={r:.2f}")
    ax.grid()

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%d-%b\n%H:%M")
    )

for ax in axes:
    ax.tick_params(axis="x", rotation=30)

axes[0].legend()
plt.tight_layout()
plt.savefig("PM25_panel.png", dpi=300)
plt.show()

# ==========================================================
# MAP
# ==========================================================

fig = plt.figure(figsize=(12,10))
ax = plt.axes(projection=ccrs.PlateCarree())

ax.set_extent([140,154,-38,-27])
ax.add_feature(cfeature.LAND, facecolor="lightgray")
ax.add_feature(cfeature.OCEAN, facecolor="white")
ax.coastlines("10m")

sc_data = summary.dropna(subset=["R"])

sc = ax.scatter(
    sc_data["Lon"],
    sc_data["Lat"],
    c=sc_data["R"],
    cmap="RdYlGn",
    s=120,
    edgecolor="black",
    vmin=0,
    vmax=1,
    transform=ccrs.PlateCarree()
)

for _, row in sc_data.iterrows():
    ax.text(row["Lon"]+0.1, row["Lat"]+0.05,
            row["Station"], fontsize=7,
            transform=ccrs.PlateCarree())

plt.colorbar(sc, label="R")
plt.title("WRF vs PM2.5 Correlation")
plt.tight_layout()
plt.savefig("PM25_map.png", dpi=300)
plt.show()

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_pm25_peak_timing(station_results, stations_to_plot):

    fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharex=True)
    axes = axes.flatten()

    for i, station in enumerate(stations_to_plot):

        ax = axes[i]

        if station not in station_results:
            ax.set_title(f"{station}\nNo data")
            ax.axis("off")
            continue

        df = station_results[station].copy()

        # clean
        df = df.dropna(subset=["WRF", "OBS"])

        if len(df) < 10:
            ax.set_title(f"{station}\nNot enough data")
            ax.axis("off")
            continue

        # ---------------------------
        # peak times
        # ---------------------------
        obs_peak_time = df["OBS"].idxmax()
        wrf_peak_time = df["WRF"].idxmax()

        obs_peak_val = df["OBS"].max()
        wrf_peak_val = df["WRF"].max()

        # ---------------------------
        # plots
        # ---------------------------
        ax.plot(df.index, df["OBS"], "k", label="OBS", linewidth=1.5)
        ax.plot(df.index, df["WRF"], "r", label="WRF", linewidth=1.5)

        # peak markers
        ax.axvline(obs_peak_time, color="black", linestyle="--", alpha=0.6)
        ax.axvline(wrf_peak_time, color="red", linestyle="--", alpha=0.6)

        ax.scatter(obs_peak_time, obs_peak_val, color="black", s=40)
        ax.scatter(wrf_peak_time, wrf_peak_val, color="red", s=40)

        # peak difference (hours)
        lag_hours = (wrf_peak_time - obs_peak_time).total_seconds() / 3600.0

        ax.set_title(
            f"{station}\n"
            f"Peak lag (WRF-OBS): {lag_hours:+.1f} h"
        )

        ax.grid(True)

        ax.xaxis.set_major_formatter(
            mdates.DateFormatter("%d-%b\n%H:%M")
        )

    # legend only once
    axes[0].legend()

    for ax in axes:
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()

    plt.savefig("PM25_peak_timing_panel.png", dpi=300)
    plt.show()

    print("Saved: PM25_peak_timing_panel.png")


plot_pm25_peak_timing(station_results, stations_to_plot)

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_all_stations_peak_day(station_results, stations_to_plot,
                               start="2023-12-18", end="2023-12-20"):

    fig, ax = plt.subplots(figsize=(14, 7))

    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    for station in stations_to_plot:

        if station not in station_results:
            continue

        df = station_results[station].copy()

        if not {"WRF", "OBS"}.issubset(df.columns):
            continue

        # ----------------------------
        # time filter (event window)
        # ----------------------------
        df = df[(df.index >= start) & (df.index <= end)]

        df = df.dropna(subset=["WRF", "OBS"])

        if len(df) < 5:
            continue

        # ----------------------------
        # normalize for comparison
        # (VERY IMPORTANT for multi-site overlay)
        # ----------------------------
        obs = df["OBS"]
        wrf = df["WRF"]

        obs_norm = (obs - obs.mean()) / obs.std()
        wrf_norm = (wrf - wrf.mean()) / wrf.std()

        # plot
        ax.plot(df.index, obs_norm, color="black", alpha=0.5)
        ax.plot(df.index, wrf_norm, color="red", alpha=0.5)

    # ----------------------------
    # styling
    # ----------------------------
    ax.set_title("PM2.5 Peak Timing Comparison (WRF vs OBS)\nNormalised across stations")
    ax.set_ylabel("Normalised PM2.5")
    ax.set_xlabel("Time")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b\n%H:%M"))

    ax.axvline(pd.to_datetime("2023-12-19 15:00"),
               color="blue", linestyle="--", label="Observed peak reference")

    ax.grid(True)
    ax.legend(["OBS", "WRF", "Observed peak time"])

    plt.tight_layout()

    plt.savefig("PM25_ALL_STATIONS_PEAK_OVERLAY.png", dpi=300)
    plt.show()

    print("Saved: PM25_ALL_STATIONS_PEAK_OVERLAY.png")

plot_all_stations_peak_day(
    station_results,
    stations_to_plot
)

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

def plot_peak_timing_raw_and_normalised(station_results,
                                        stations_to_plot,
                                        start="2023-12-18",
                                        end="2023-12-20"):

    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharex=True)

    ax_raw = axes[0]
    ax_norm = axes[1]

    for station in stations_to_plot:

        if station not in station_results:
            continue

        df = station_results[station].copy()

        if not {"WRF", "OBS"}.issubset(df.columns):
            continue

        # -----------------------
        # time window
        # -----------------------
        df = df[(df.index >= start) & (df.index <= end)]
        df = df.dropna(subset=["WRF", "OBS"])

        if len(df) < 5:
            continue

        obs = df["OBS"]
        wrf = df["WRF"]

        # =====================================================
        # RAW values (physical units)
        # =====================================================
        ax_raw.plot(df.index, obs, color="black", alpha=0.5)
        ax_raw.plot(df.index, wrf, color="red", alpha=0.5)

        # =====================================================
        # NORMALISED (shape only)
        # =====================================================
        obs_n = (obs - obs.mean()) / obs.std()
        wrf_n = (wrf - wrf.mean()) / wrf.std()

        ax_norm.plot(df.index, obs_n, color="black", alpha=0.5)
        ax_norm.plot(df.index, wrf_n, color="red", alpha=0.5)

    # -----------------------
    # formatting RAW
    # -----------------------
    ax_raw.set_title("PM2.5 RAW VALUES (WRF vs OBS)")
    ax_raw.set_ylabel("PM2.5 (µg/m³)")
    ax_raw.grid(True)

    # -----------------------
    # formatting NORM
    # -----------------------
    ax_norm.set_title("PM2.5 NORMALISED (Timing Comparison)")
    ax_norm.set_ylabel("Normalised PM2.5")
    ax_norm.grid(True)

    # -----------------------
    # shared x-axis format
    # -----------------------
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b\n%H:%M"))
        ax.axvline(pd.to_datetime("2023-12-19 15:00"),
                   color="blue", linestyle="--", linewidth=1)

    plt.suptitle("WRF-Chem PM2.5 Peak Timing Analysis (Sydney Stations)", fontsize=14)

    plt.tight_layout()

    plt.savefig("PM25_RAW_AND_NORMALISED_PEAK.png", dpi=300)
    plt.show()

    print("Saved: PM25_RAW_AND_NORMALISED_PEAK.png")

plot_peak_timing_raw_and_normalised(
    station_results,
    stations_to_plot
)

