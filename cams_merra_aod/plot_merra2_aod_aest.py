#!/usr/bin/env python3

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd

# ===============================
# INPUT FILES
# ===============================
files = [
    "MERRA2_400.tavg1_2d_aer_Nx.20231217.nc4",
    "MERRA2_400.tavg1_2d_aer_Nx.20231218.nc4",
    "MERRA2_400.tavg1_2d_aer_Nx.20231219.nc4",
]

# ===============================
# OPEN
# ===============================
ds = xr.open_mfdataset(files)

# Convert UTC → AEST (UTC+10)
time_aest = pd.to_datetime(ds.time.values) + pd.Timedelta(hours=10)
ds = ds.assign_coords(time=time_aest)

# ===============================
# SELECT NSW
# ===============================
ds = ds.sel(
    lat=slice(-37.5, -28),
    lon=slice(141, 154)
)

# ===============================
# COMPUTE 3-DAY MEAN AOD
# ===============================
aod = ds["TOTEXTTAU"].mean(dim="time")

# remove fill values
aod = aod.where(aod < 1e14)

# ===============================
# PLOT
# ===============================
fig = plt.figure(figsize=(8,7))

ax = plt.axes(projection=ccrs.PlateCarree())

pcm = ax.pcolormesh(
    ds.lon,
    ds.lat,
    aod,
    cmap="YlOrRd",
    vmin=0,
    vmax=1,
    transform=ccrs.PlateCarree()
)

ax.coastlines(resolution="10m", linewidth=0.8)

ax.add_feature(
    cfeature.BORDERS,
    linewidth=0.5
)

ax.set_extent([141,154,-37.5,-28])

gl = ax.gridlines(
    draw_labels=True,
    linewidth=0.3
)

gl.top_labels=False
gl.right_labels=False

# publication-style smaller colorbar
cb = plt.colorbar(
    pcm,
    shrink=0.75,
    pad=0.03,
    aspect=25
)

cb.set_label(
    "MERRA-2 AOD550",
    fontsize=12
)

plt.title(
    "Mean MERRA-2 AOD (550 nm)\n17–19 Dec 2023 (AEST)",
    fontsize=14
)

plt.tight_layout()

plt.savefig(
    "merra2_aod_nsw_mean_publication.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

