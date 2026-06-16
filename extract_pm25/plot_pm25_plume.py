#!/usr/bin/env python3

from netCDF4 import Dataset
from wrf import getvar, latlon_coords
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

wrf_file = "/mnt/scratch_lustre/duch/runpillaga/wrfout_d01_2023-12-08_00:00:00"

# Hour index within file
itime = 16   # 03 UTC

nc = Dataset(wrf_file)

pm25 = getvar(nc, "PM2_5_DRY", timeidx=itime)

# surface layer
pm25_sfc = pm25[0,:,:]

lats, lons = latlon_coords(pm25_sfc)

fig = plt.figure(figsize=(12,8))

ax = plt.axes(projection=ccrs.PlateCarree())

ax.set_extent(
    [145, 155, -36, -26],
    crs=ccrs.PlateCarree()
)

#levels = [0,5,10,20,30,50,75,100,150,200]
levels = [0,5,10,15,17,20,25,27,30,35,37,40,45,50,55,60,65,70,75,80,85]

cf = ax.contourf(
    lons,
    lats,
    pm25_sfc,
    levels=levels,
    extend='max',
    transform=ccrs.PlateCarree()
)

plt.colorbar(
    cf,
    label="PM2.5 (µg m$^{-3}$)"
)

# stations
ax.plot(
    149.829,
    -30.318,
    'ko',
    markersize=8,
    transform=ccrs.PlateCarree()
)
ax.text(
    149.829,
    -30.318,
    ' Narrabri',
    transform=ccrs.PlateCarree()
)

ax.plot(
    150.261,
    -30.982,
    'ko',
    markersize=8,
    transform=ccrs.PlateCarree()
)
ax.text(
    150.261,
    -30.982,
    ' Gunnedah',
    transform=ccrs.PlateCarree()
)

ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)
ax.add_feature(cfeature.STATES)

plt.title(f"Surface PM2.5 plume on 8/12/2023 at {itime}:00 UTC")

plt.savefig(
    "PM25_plume_peak.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
