#!/usr/bin/env python3

import glob
import numpy as np
import pandas as pd
import xarray as xr
from wrf import getvar, latlon_coords, ALL_TIMES
from netCDF4 import Dataset
import matplotlib.pyplot as plt

# -------------------------------------------------------
# USER INPUTS
# -------------------------------------------------------

#wrf_files = sorted(glob.glob(
    #"/mnt/scratch_lustre/duch/runpillaga/wrfout_d01_2023-12*"
#    "/mnt/scratch_lustre/duch/runpillaga/wrfout_d01_2023-12-0[6-9]_00:00:00"
#))

#wrf_files = sorted(glob.glob(
#    "/mnt/scratch_lustre/duch/runpillaga/wrfout_d01_2023-12-0[6-11]_00:00:00"
#))

wrf_files = sorted(
    glob.glob("/mnt/scratch_lustre/duch/runpillaga/wrfout_d01_2023-12-0[6-9]_00:00:00")
    +
    glob.glob("/mnt/scratch_lustre/duch/runpillaga/wrfout_d01_2023-12-1[0-3]_00:00:00")
)

obs_file = "HiepNarrabriGunnedahPM25Dec2023.csv"

stations = {
    "Narrabri": {
        "lat": -30.318,
        "lon": 149.829,
        "obs_col": "NARRABRI PM2.5"
    },
    "Gunnedah": {
        "lat": -30.982,
        "lon": 150.261,
        "obs_col": "GUNNEDAH PM2.5"
    }
}

# -------------------------------------------------------
# READ WRF PM2.5
# -------------------------------------------------------

print("Reading WRF output...")

ncfiles = [Dataset(f) for f in wrf_files]

pm25_3d = getvar(ncfiles, "PM2_5_DRY", timeidx=ALL_TIMES)
pm25 = pm25_3d[:, 0, :, :]   # surface layer (level 0)

lats, lons = latlon_coords(pm25)

times = pd.to_datetime(
    np.array(pm25.Time.values).astype(str)
)

# convert UTC -> AEST
times = times + pd.Timedelta(hours=10)

wrf_df = pd.DataFrame({"Date_time": times})

# -------------------------------------------------------
# EXTRACT STATIONS
# -------------------------------------------------------

for stn, info in stations.items():

    dist2 = (
        (lats.values - info["lat"])**2 +
        (lons.values - info["lon"])**2
    )

    iy, ix = np.unravel_index(
        np.argmin(dist2),
        dist2.shape
    )

    print(
        f"{stn}: grid ({iy},{ix}) "
        f"lat={lats.values[iy,ix]:.3f} "
        f"lon={lons.values[iy,ix]:.3f}"
    )

    wrf_df[stn] = pm25[:, iy, ix].values

# -------------------------------------------------------
# READ OBSERVATIONS
# -------------------------------------------------------

obs = pd.read_csv(obs_file)

obs["Date_time"] = pd.to_datetime(
    obs["Date_time"],
    dayfirst=True,
    errors="coerce"
)

# remove negative values
for stn, info in stations.items():
    obs.loc[
        obs[info["obs_col"]] < 0,
        info["obs_col"]
    ] = np.nan

# -------------------------------------------------------
# PERIOD OF INTEREST
# -------------------------------------------------------

start = "2023-12-06 00:00"
end   = "2023-12-12 23:59"

wrf_df = wrf_df[
    (wrf_df["Date_time"] >= start) &
    (wrf_df["Date_time"] <= end)
]

obs = obs[
    (obs["Date_time"] >= start) &
    (obs["Date_time"] <= end)
]

# -------------------------------------------------------
# MERGE
# -------------------------------------------------------

merged = pd.merge(
    obs,
    wrf_df,
    on="Date_time",
    how="inner"
)

# -------------------------------------------------------
# STATISTICS
# -------------------------------------------------------

def stats(obs, mod):

    mask = (
        np.isfinite(obs) &
        np.isfinite(mod)
    )

    obs = obs[mask]
    mod = mod[mask]

    bias = np.mean(mod - obs)

    rmse = np.sqrt(
        np.mean((mod - obs)**2)
    )

    corr = np.corrcoef(obs, mod)[0,1]

    return bias, rmse, corr

for stn, info in stations.items():

    bias, rmse, corr = stats(
        merged[info["obs_col"]],
        merged[stn]
    )

    print("\n", stn)
    print("Bias =", round(bias,2))
    print("RMSE =", round(rmse,2))
    print("Corr =", round(corr,2))

# -------------------------------------------------------
# PLOT (OBS vs MODEL with wildfire highlight)
# -------------------------------------------------------

for stn, info in stations.items():

    fig, ax = plt.subplots(figsize=(14,5))

    # Observations
    ax.plot(
        merged["Date_time"],
        merged[info["obs_col"]],
        label="Observed",
        linewidth=2
    )

    # Model
    ax.plot(
        merged["Date_time"],
        merged[stn],
        label="WRF-Chem",
        linewidth=2
    )

    # Highlight wildfire peak period (9–12 Dec)
    ax.axvspan(
        pd.to_datetime("2023-12-09 00:00"),
        pd.to_datetime("2023-12-12 23:59"),
        color="red",
        alpha=0.1,
        label="Peak fire period (9–12 Dec)"
    )

    ax.set_ylabel("PM2.5 (µg m$^{-3}$)")
    ax.set_title(f"{stn}: Observed vs WRF-Chem PM2.5 (Dec 2023)")
    ax.grid(True, alpha=0.3)

    ax.legend()

    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(
        f"{stn}_PM25_OBS_vs_MODEL_Dec2023.png",
        dpi=200
    )

    plt.show()
    plt.close()

print("Plots saved.")

