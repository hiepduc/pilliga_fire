#!/usr/bin/env python3

import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import glob

# ----------------------------------
# USER INPUT (AEST)
# ----------------------------------
target_aest = pd.Timestamp("2023-12-19 12:30")

# convert to UTC
target_utc = target_aest - pd.Timedelta(hours=10)

print("Target AEST:", target_aest)
print("Selecting UTC:", target_utc)

# ----------------------------------
# OPEN FILES
# ----------------------------------
files = sorted(
    glob.glob("MERRA2_400.tavg1_2d_aer_Nx.202312*.nc4")
)

print(files)

ds = xr.open_mfdataset(
    files,
    combine="by_coords"
)

# ----------------------------------
# SELECT IN UTC
# ----------------------------------
aod = ds["TOTEXTTAU"]

aod = aod.where(aod < 1e14)

aod_sel = aod.sel(
    time=target_utc,
    method="nearest"
)

# convert selected time for display
selected_aest = (
    pd.to_datetime(aod_sel.time.values)
    + pd.Timedelta(hours=10)
)

print("Actual selected AEST:", selected_aest)

# ----------------------------------
# NSW
# ----------------------------------
aod_sel = aod_sel.sel(
    lat=slice(-37.5, -28),
    lon=slice(141, 154)
)

# ----------------------------------
# PLOT
# ----------------------------------
fig = plt.figure(figsize=(8,7))
ax = plt.axes(projection=ccrs.PlateCarree())

pcm = ax.pcolormesh(
    aod_sel.lon,
    aod_sel.lat,
    aod_sel,
    cmap="YlOrRd",
    vmin=0,
    vmax=1,
    shading="auto"
)

ax.coastlines("10m")
ax.add_feature(cfeature.BORDERS)

cb = plt.colorbar(
    pcm,
    shrink=0.75,
    pad=0.03,
    aspect=25
)

cb.set_label("MERRA-2 AOD550")

plt.title(
    selected_aest.strftime(
        "MERRA-2 AOD\n%d %b %Y %H:%M AEST"
    )
)

outfile = (
    selected_aest.strftime(
        "merra2_aod_%Y%m%d_%H%M_AEST.png"
    )
)

plt.savefig(outfile, dpi=300)

print("Saved:", outfile)

plt.show()

