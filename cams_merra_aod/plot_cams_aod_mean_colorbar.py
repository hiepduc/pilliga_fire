#!/usr/bin/env python3

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ---------------------
# Input
# ---------------------
ncfile = "cams_aod550_nsw_standard.nc"

# ---------------------
# Read
# ---------------------
ds = xr.open_dataset(ncfile)

# Mean over 3 days
aod_mean = (
    ds["aod550"]
    .sel(
        time=slice(
            "2023-12-17T00:00",
            "2023-12-19T14:00"
        )
    )
    .mean(dim="time")
)

# ---------------------
# Plot
# ---------------------
fig = plt.figure(figsize=(10,8))

ax = plt.axes(
    projection=ccrs.PlateCarree()
)

pcm = ax.pcolormesh(
    ds.longitude,
    ds.latitude,
    aod_mean,
    shading="auto",
    cmap="turbo",
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

cb = plt.colorbar(
    pcm,
    ax=ax,
    shrink=0.72,      # shorter
    fraction=0.04,    # thinner
    pad=0.03          # gap from map
)

cb.set_label(
    "Mean AOD550",
    fontsize=12
)

cb.ax.tick_params(
    labelsize=10
)
plt.title(
    "CAMS Mean AOD550\n17–19 Dec 2023 (AEST, UTC+10)"
)

plt.tight_layout()

plt.savefig(
    "cams_mean_aod550_17_19Dec_AEST.png",
    dpi=300
)

plt.show()

