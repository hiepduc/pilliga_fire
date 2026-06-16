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
    "/mnt/scratch_lustre/duch/runpillaga/jkong/wrfout_d02_2023-12-8to19_wrfnative.nc"
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

