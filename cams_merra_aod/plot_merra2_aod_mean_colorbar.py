#!/usr/bin/env python3

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import glob

# ==================================
# FIND AVAILABLE FILES
# ==================================
files = sorted(
    glob.glob("MERRA2_400.tavg1_2d_aer_Nx.202312*.nc4")
)

print("\nFiles found:")
for f in files:
    print(f)

if len(files) == 0:
    raise Exception("No MERRA files found")

# ==================================
# OPEN
# ==================================
ds = xr.open_mfdataset(
    files,
    combine="by_coords"
)

# UTC → AEST (UTC+10)
ds["time"] = (
    pd.to_datetime(ds.time.values)
    + pd.Timedelta(hours=10)
)

# ==================================
# NSW DOMAIN
# ==================================
ds = ds.sel(
    lat=slice(-37.5, -28),
    lon=slice(141, 154)
)

# ==================================
# TOTAL AOD550
# ==================================
aod = ds["TOTEXTTAU"]

# remove fill values
aod = aod.where(aod < 1e14)

# mean over available times
mean_aod = aod.mean(dim="time")

# ---------------------
# Plot
# ---------------------
fig = plt.figure(figsize=(10,8))

ax = plt.axes(
    projection=ccrs.PlateCarree()
)

pcm = ax.pcolormesh(
    ds.lon,
    ds.lat,
    mean_aod,
    shading="auto",
    cmap="turbo",          # SAME AS CAMS
    transform=ccrs.PlateCarree(),
    vmin=0,
    vmax=1
)

ax.coastlines(
    resolution="10m",
    linewidth=1.2
)

ax.add_feature(
    cfeature.BORDERS,
    linewidth=0.5
)

gl = ax.gridlines(
    draw_labels=True,
    linestyle=":"
)

gl.top_labels = False
gl.right_labels = False

ax.set_extent(
    [141,154,-37,-28]
)

# identical colorbar style
cb = plt.colorbar(
    pcm,
    ax=ax,
    shrink=0.72,
    fraction=0.04,
    pad=0.03
)

cb.set_label(
    "AOD550",
    fontsize=12
)

cb.ax.tick_params(
    labelsize=10
)

plt.title(
    f"Mean MERRA-2 AOD550 (550 nm)\n17 to 19 Dec 2023, AEST)"
)

plt.tight_layout()

outfile = (
    f"merra2_aod_nsw_mean_publication2.png"
)

plt.savefig(
    outfile,
    dpi=300,
    bbox_inches="tight"
)

print("Saved:", outfile)

plt.show()


