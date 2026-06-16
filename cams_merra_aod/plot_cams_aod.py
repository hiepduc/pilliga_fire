#!/usr/bin/env python3

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# -----------------------------
# INPUT
# -----------------------------
ncfile = "cams_aod550_nsw_standard.nc"

# 19 Dec 2023 12:00 Sydney (AEDT)
plot_time = "2023-12-19T00:00"

# -----------------------------
# Read
# -----------------------------
ds = xr.open_dataset(ncfile)

aod = ds["aod550"].sel(
    time=plot_time,
    method="nearest"
)

# -----------------------------
# Plot
# -----------------------------
fig = plt.figure(
    figsize=(10,8)
)

ax = plt.axes(
    projection=ccrs.PlateCarree()
)

# AOD field
im = ax.pcolormesh(
    ds.longitude,
    ds.latitude,
    aod,
    shading="auto",
    cmap="turbo",
    transform=ccrs.PlateCarree(),
    vmin=0,
    vmax=1
)

# Coastline
ax.coastlines(
    resolution="10m",
    linewidth=1.0
)

# Borders
ax.add_feature(
    cfeature.BORDERS,
    linewidth=0.5
)

# Grid
gl = ax.gridlines(
    draw_labels=True,
    linestyle=":"
)

gl.top_labels = False
gl.right_labels = False

# NSW extent
ax.set_extent(
    [141,154,-37,-28],
    crs=ccrs.PlateCarree()
)

plt.colorbar(
    im,
    label="AOD550"
)

plt.title(
    "CAMS AOD550\n19 Dec 2023 12:00 AEDT (~00 UTC)"
)

plt.tight_layout()

plt.savefig(
    "cams_aod_nsw_20231219_1200AEDT.png",
    dpi=300
)

plt.show()

