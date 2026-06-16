#!/usr/bin/env python3

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import glob

# =================================
# TARGET TIME (AEST)
# =================================
target_time = pd.Timestamp("2023-12-19 12:30")

# =================================
# FIND FILES
# =================================
files = sorted(
    glob.glob("MERRA2_400.tavg1_2d_aer_Nx.202312*.nc4")
)

print("Files found:")
for f in files:
    print(f)

if not files:
    raise Exception("No MERRA files found")

# =================================
# OPEN
# =================================
ds = xr.open_mfdataset(
    files,
    combine="by_coords"
)

# =================================
# UTC → AEST
# MERRA time starts at xx:30 UTC
# =================================
ds = ds.assign_coords(
    time=(
        pd.to_datetime(ds.time.values)
        + pd.Timedelta(hours=10)
    )
)

# =================================
# NSW DOMAIN
# =================================
ds = ds.sel(
    lat=slice(-37.5, -28),
    lon=slice(141, 154)
)

# =================================
# SELECT AOD
# =================================
aod = ds["TOTEXTTAU"]

# remove fill values
aod = aod.where(aod < 1e14)

# =================================
# SELECT TIME
# =================================
aod_sel = aod.sel(
    time=target_time,
    method="nearest"
)

selected = pd.to_datetime(aod_sel.time.values)

print("Selected time:", selected)

# =================================
# PLOT
# =================================
fig = plt.figure(figsize=(8,7))

ax = plt.axes(
    projection=ccrs.PlateCarree()
)

pcm = ax.pcolormesh(
    ds.lon,
    ds.lat,
    aod_sel,
    cmap="YlOrRd",
    vmin=0,
    vmax=1,
    shading="auto",
    transform=ccrs.PlateCarree()
)

ax.coastlines(
    resolution="10m",
    linewidth=0.8
)

ax.add_feature(
    cfeature.BORDERS,
    linewidth=0.5
)

ax.set_extent([141,154,-37.5,-28])

gl = ax.gridlines(
    draw_labels=True,
    linewidth=0.3
)

gl.top_labels = False
gl.right_labels = False

cb = plt.colorbar(
    pcm,
    shrink=0.75,
    pad=0.03,
    aspect=25
)

cb.set_label("MERRA-2 AOD550")

plt.title(
    f"MERRA-2 AOD (550 nm)\n"
    f"{selected:%d %b %Y %H:%M AEST}"
)

plt.tight_layout()

outfile = (
    f"merra2_aod_"
    f"{selected:%Y%m%d_%H%M}_AEST.png"
)

plt.savefig(
    outfile,
    dpi=300,
    bbox_inches="tight"
)

print("Saved:", outfile)

plt.show()

