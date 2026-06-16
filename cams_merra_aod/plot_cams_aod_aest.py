#!/usr/bin/env python3

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ----------------------
# INPUT
# ----------------------
ncfile = "cams_aod550_nsw_standard.nc"

# Desired local time
#plot_time_utc = "2023-12-19T03:00"
plot_time_utc = "2023-12-19T00:00"

# ----------------------
# Read
# ----------------------
ds = xr.open_dataset(ncfile)

aod = ds["aod550"].sel(
    time=plot_time_utc,
    method="nearest"
)

actual_time = str(aod.time.values)

# ----------------------
# Plot
# ----------------------
fig = plt.figure(
    figsize=(10,8)
)

ax = plt.axes(
    projection=ccrs.PlateCarree()
)

pcm = ax.pcolormesh(
    ds.longitude,
    ds.latitude,
    aod,
    cmap="turbo",
    shading="auto",
    vmin=0,
    vmax=1,
    transform=ccrs.PlateCarree()
)

ax.coastlines(
    resolution="10m",
    linewidth=1.2
)

ax.add_feature(
    cfeature.BORDERS,
    linewidth=0.5
)

ax.gridlines(
    draw_labels=True,
    linestyle=":"
)

ax.set_extent(
    [141,154,-37,-28]
)

cb = plt.colorbar(
    pcm,
    ax=ax
)

cb.set_label("AOD550")

plt.title(
    f"CAMS AOD550\n19 Dec 2023 12:00 AEST (nearest UTC={actual_time})"
)

plt.tight_layout()

plt.savefig(
    "cams_aod550_20231219_1200_AEST.png",
    dpi=300
)

plt.show()

