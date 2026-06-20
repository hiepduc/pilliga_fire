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
    wrf_pm25  = []
    wrf_pblh  = []

    print("Processing:", station)

    for wrf_file in wrf_files:

        nc = Dataset(wrf_file)

        xy = ll_to_xy(nc, lat_site, lon_site)
        ix, iy = int(xy[0]), int(xy[1])

        pm25 = getvar(
            nc,
            "PM2_5_DRY",
            timeidx=None
        ).values

        pblh = getvar(
            nc,
            "PBLH",
            timeidx=None
        ).values

        times = getvar(
            nc,
            "times",
            timeidx=None
        ).values

        nt = pm25.shape[0]

        for t in range(nt):

            wrf_times.append(
                pd.to_datetime(str(times[t]))
            )

            wrf_pm25.append(
                float(pm25[t,0,iy,ix])
            )

            wrf_pblh.append(
                float(pblh[t,iy,ix])
            )

    wrf_times = (
        pd.to_datetime(wrf_times)
        + pd.Timedelta(hours=UTC_OFFSET)
    )

    wrf_cache[station] = pd.DataFrame(
        {
            "WRF_PM25": wrf_pm25,
            "WRF_PBLH": wrf_pblh
        },
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

    valid = compare.dropna(
        subset=["WRF_PM25", "OBS"]
    )

    wrf = valid["WRF_PM25"].values
    obs_v = valid["OBS"].values

    if len(valid) < 10:
        continue

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

    df_plot = df.dropna(subset=["WRF_PM25","OBS"])

    if len(df_plot) < 2:
        ax.set_title(station + "\nInsufficient data")
        continue

    r = np.corrcoef(df_plot["OBS"], df_plot["WRF_PM25"])[0,1]

    ax.plot(df_plot.index, df_plot["OBS"], "k", label="Obs")
    ax.plot(df_plot.index, df_plot["WRF_PM25"], "r", label="WRF")

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

# ##############
# Plot pm2.5 and pblh
################

############

station = "LIDCOMBE"

df = station_results[station]

fig, ax1 = plt.subplots(figsize=(15,6))

ax1.plot(
    df.index,
    df["OBS"],
    "k",
    lw=2,
    label="Observed PM2.5"
)

ax1.plot(
    df.index,
    df["WRF_PM25"],
    "r",
    lw=2,
    label="WRF PM2.5"
)

ax1.set_ylabel(
    "PM2.5 ($\\mu$g m$^{-3}$)"
)

ax2 = ax1.twinx()

ax2.plot(
    df.index,
    df["WRF_PBLH"],
    "b--",
    lw=2,
    label="WRF PBLH"
)

ax2.set_ylabel(
    "PBLH (m)"
)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc="upper left"
)

plt.title(
    f"{station}: PM2.5 and PBLH"
)

plt.grid(True)
plt.tight_layout()
plt.savefig(
    f"{station}_PM25_PBLH.png",
    dpi=300
)
plt.show()

#######################
# Lidcombe PM2.5, PBLH WRF and PBLH ceilo
#######################
# ======================================================
# CEILOMETER FILES
# ======================================================

csv_files = [
    "/mnt/scratch_lustre/duch/runpillaga/extract_pblh/ceilo_data/L3_DEFAULT_0_20231206_Lidcombe.csv",
    "/mnt/scratch_lustre/duch/runpillaga/extract_pblh/ceilo_data/L3_DEFAULT_0_20231207_Lidcombe.csv",
    "/mnt/scratch_lustre/duch/runpillaga/extract_pblh/ceilo_data/L3_DEFAULT_0_20231208_Lidcombe.csv",
    "/mnt/scratch_lustre/duch/runpillaga/extract_pblh/ceilo_data/L3_DEFAULT_0_20231209_Lidcombe.csv",
    "/mnt/scratch_lustre/duch/runpillaga/extract_pblh/ceilo_data/L3_DEFAULT_0_20231210_Lidcombe.csv",
    "/mnt/scratch_lustre/duch/runpillaga/extract_pblh/ceilo_data/L3_DEFAULT_0_20231211_Lidcombe.csv",
    "/mnt/scratch_lustre/duch/runpillaga/extract_pblh/ceilo_data/L3_DEFAULT_0_20231212_Lidcombe.csv",
    "/mnt/scratch_lustre/duch/runpillaga/extract_pblh/ceilo_data/L3_DEFAULT_0_20231213_Lidcombe.csv",
    "/mnt/scratch_lustre/duch/runpillaga/extract_pblh/ceilo_data/L3_DEFAULT_0_20231214_Lidcombe.csv",
    "/mnt/scratch_lustre/duch/runpillaga/extract_pblh/ceilo_data/L3_DEFAULT_0_20231215_Lidcombe.csv",
    "/mnt/scratch_lustre/duch/runpillaga/extract_pblh/ceilo_data/L3_DEFAULT_0_20231216_Lidcombe.csv",
    "/mnt/scratch_lustre/duch/runpillaga/extract_pblh/ceilo_data/L3_DEFAULT_0_20231217_Lidcombe.csv",
    "/mnt/scratch_lustre/duch/runpillaga/extract_pblh/ceilo_data/L3_DEFAULT_0_20231218_Lidcombe.csv",
    "/mnt/scratch_lustre/duch/runpillaga/extract_pblh/ceilo_data/L3_DEFAULT_0_20231219_Lidcombe.csv",
    "/mnt/scratch_lustre/duch/runpillaga/extract_pblh/ceilo_data/L3_DEFAULT_0_20231220_Lidcombe.csv",
    "/mnt/scratch_lustre/duch/runpillaga/extract_pblh/ceilo_data/L3_DEFAULT_0_20231221_Lidcombe.csv",
    "/mnt/scratch_lustre/duch/runpillaga/extract_pblh/ceilo_data/L3_DEFAULT_0_20231222_Lidcombe.csv",
    "/mnt/scratch_lustre/duch/runpillaga/extract_pblh/ceilo_data/L3_DEFAULT_0_20231223_Lidcombe.csv"
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
# HOURLY CEILOMETER PBLH
# ======================================================

ceilo_hourly = (
    ceilo
    .set_index("Time")
    .resample("1H")
    .median(numeric_only=True)
)

print("Ceilometer hourly records =", len(ceilo_hourly))


lidcombe_pblh = ceilo_hourly["bl_height"].copy()
lidcombe_pblh.name = "PBLH_OBS"

lidcombe_pblh.to_csv(
    "LIDCOMBE_PBLH_hourly.csv"
)

csv_files = [
    "/mnt/scratch_lustre/duch/runpillaga/extract_pblh/ceilo_data/L3_DEFAULT_0_20231206_Merriwa.csv",
    "/mnt/scratch_lustre/duch/runpillaga/extract_pblh/ceilo_data/L3_DEFAULT_0_20231207_Merriwa.csv",
    "/mnt/scratch_lustre/duch/runpillaga/extract_pblh/ceilo_data/L3_DEFAULT_0_20231208_Merriwa.csv",
    "/mnt/scratch_lustre/duch/runpillaga/extract_pblh/ceilo_data/L3_DEFAULT_0_20231209_Merriwa.csv",
    "/mnt/scratch_lustre/duch/runpillaga/extract_pblh/ceilo_data/L3_DEFAULT_0_20231210_Merriwa.csv",
    "/mnt/scratch_lustre/duch/runpillaga/extract_pblh/ceilo_data/L3_DEFAULT_0_20231211_Merriwa.csv",
    "/mnt/scratch_lustre/duch/runpillaga/extract_pblh/ceilo_data/L3_DEFAULT_0_20231212_Merriwa.csv",
    "/mnt/scratch_lustre/duch/runpillaga/extract_pblh/ceilo_data/L3_DEFAULT_0_20231213_Merriwa.csv",
    "/mnt/scratch_lustre/duch/runpillaga/extract_pblh/ceilo_data/L3_DEFAULT_0_20231214_Merriwa.csv",
    "/mnt/scratch_lustre/duch/runpillaga/extract_pblh/ceilo_data/L3_DEFAULT_0_20231215_Merriwa.csv",
    "/mnt/scratch_lustre/duch/runpillaga/extract_pblh/ceilo_data/L3_DEFAULT_0_20231216_Merriwa.csv",
    "/mnt/scratch_lustre/duch/runpillaga/extract_pblh/ceilo_data/L3_DEFAULT_0_20231217_Merriwa.csv",
    "/mnt/scratch_lustre/duch/runpillaga/extract_pblh/ceilo_data/L3_DEFAULT_0_20231218_Merriwa.csv",
    "/mnt/scratch_lustre/duch/runpillaga/extract_pblh/ceilo_data/L3_DEFAULT_0_20231219_Merriwa.csv",
    "/mnt/scratch_lustre/duch/runpillaga/extract_pblh/ceilo_data/L3_DEFAULT_0_20231220_Merriwa.csv",
    "/mnt/scratch_lustre/duch/runpillaga/extract_pblh/ceilo_data/L3_DEFAULT_0_20231221_Merriwa.csv",
    "/mnt/scratch_lustre/duch/runpillaga/extract_pblh/ceilo_data/L3_DEFAULT_0_20231222_Merriwa.csv",
    "/mnt/scratch_lustre/duch/runpillaga/extract_pblh/ceilo_data/L3_DEFAULT_0_20231223_Merriwa.csv"
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
# HOURLY CEILOMETER PBLH
# ======================================================

ceilo_hourly = (
    ceilo
    .set_index("Time")
    .resample("1H")
    .median(numeric_only=True)
)

print("Ceilometer hourly records =", len(ceilo_hourly))

merriwa_pblh = ceilo_hourly["bl_height"].copy()
merriwa_pblh.name = "PBLH_OBS"

merriwa_pblh.to_csv(
    "MERRIWA_PBLH_hourly.csv"
)

pblh_obs = {}

try:
    pblh_obs["LIDCOMBE"] = (
        pd.read_csv(
            "LIDCOMBE_PBLH_hourly.csv",
            index_col=0,
            parse_dates=True
        )
    )
except:
    pass

try:
    pblh_obs["MERRIWA"] = (
        pd.read_csv(
            "MERRIWA_PBLH_hourly.csv",
            index_col=0,
            parse_dates=True
        )
    )
except:
    pass

for station in ["LIDCOMBE","MERRIWA"]:

    if station not in station_results:
        continue

    df = station_results[station]

    fig, ax1 = plt.subplots(
        figsize=(16,7)
    )

    # ---------------------------------
    # PM2.5
    # ---------------------------------

    ax1.plot(
        df.index,
        df["OBS"],
        "k",
        lw=2,
        label="Observed PM2.5"
    )

    ax1.plot(
        df.index,
        df["WRF_PM25"],
        "r",
        lw=2,
        label="WRF PM2.5"
    )

    ax1.set_ylabel(
        "PM2.5 ($\\mu$g m$^{-3}$)"
    )

    # ---------------------------------
    # PBLH
    # ---------------------------------

    ax2 = ax1.twinx()

    ax2.plot(
        df.index,
        df["WRF_PBLH"],
        "b--",
        lw=2,
        label="WRF PBLH"
    )

    if station in pblh_obs:

        ax2.plot(
            pblh_obs[station].index,
            pblh_obs[station]["PBLH_OBS"],
            color="green",
            lw=2,
            label="Observed PBLH"
        )

    ax2.set_ylabel(
        "PBL Height (m)"
    )

    # ---------------------------------
    # Focus on smoke event
    # ---------------------------------

    ax1.set_xlim(
        pd.Timestamp("2023-12-18 00:00"),
        pd.Timestamp("2023-12-20 00:00")
    )

    # ---------------------------------
    # Combined legend
    # ---------------------------------

    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()

    ax1.legend(
        l1+l2,
        lab1+lab2,
        loc="upper left"
    )

    ax1.grid(True)

    plt.title(
        f"{station}: PM2.5 and PBLH"
    )

    plt.tight_layout()

    plt.savefig(
        f"{station}_PM25_PBLH.png",
        dpi=300
    )

    plt.show()

######################
# Peak difference
#####################

obs_peak_time = (
    df["OBS"].idxmax()
)

wrf_peak_time = (
    df["WRF_PM25"].idxmax()
)

print(
    station,
    "OBS peak:",
    obs_peak_time
)

print(
    station,
    "WRF peak:",
    wrf_peak_time
)

print(
    "Difference:",
    (obs_peak_time - wrf_peak_time)
)

