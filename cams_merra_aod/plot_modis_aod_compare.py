#!/usr/bin/env python3

import glob
import numpy as np
from pyhdf.SD import SD, SDC
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ==========================
# DOMAIN
# ==========================
lonmin, lonmax = 141,154
latmin, latmax = -37,-28

# output grid
lon_grid = np.arange(lonmin, lonmax, 0.05)
lat_grid = np.arange(latmin, latmax, 0.05)

LON, LAT = np.meshgrid(lon_grid, lat_grid)

# ==========================
# FILES
# ==========================
files = sorted(
    glob.glob(
"/mnt/scratch_lustre/duch/runpillaga/cams_merra_aod/MOD04_L2/2023/353/*.hdf"
    )
)

print("Found",len(files),"files")

all_lon=[]
all_lat=[]
all_aod=[]

for f in files:

    try:

        hdf = SD(f, SDC.READ)

        aod = hdf.select(
            "Optical_Depth_Land_And_Ocean"
        )[:]

        lat = hdf.select(
            "Latitude"
        )[:]

        lon = hdf.select(
            "Longitude"
        )[:]

        attrs = hdf.select(
            "Optical_Depth_Land_And_Ocean"
        ).attributes()

        scale = attrs.get(
            "scale_factor",
            0.001
        )

        fill = attrs.get(
            "_FillValue",
            -9999
        )

        aod = aod.astype(float)

        aod[aod==fill]=np.nan

        aod *= scale

        mask = (
            (lon>lonmin)&
            (lon<lonmax)&
            (lat>latmin)&
            (lat<latmax)&
            (aod>0)&
            (aod<5)
        )

        if np.any(mask):

            all_lon.append(
                lon[mask]
            )

            all_lat.append(
                lat[mask]
            )

            all_aod.append(
                aod[mask]
            )

    except Exception as e:

        print("skip",f,e)

# ==========================
# CHECK
# ==========================
if len(all_aod)==0:

    raise Exception(
        "No MODIS points found"
    )

lon=np.concatenate(all_lon)
lat=np.concatenate(all_lat)
aod=np.concatenate(all_aod)

print("MODIS pixels:",len(aod))

# ==========================
# GRID
# ==========================
modis = griddata(
    (lon,lat),
    aod,
    (LON,LAT),
    method="linear"
)

# ==========================
# PLOT
# ==========================
fig=plt.figure(
    figsize=(10,8)
)

ax=plt.axes(
    projection=ccrs.PlateCarree()
)

pcm=ax.pcolormesh(
    lon_grid,
    lat_grid,
    modis,
    cmap="turbo",
    vmin=0,
    vmax=1,
    shading="auto"
)

ax.coastlines(
    resolution="10m"
)

ax.add_feature(
    cfeature.BORDERS
)

ax.set_extent(
    [141,154,-37,-28]
)

cb=plt.colorbar(
    pcm,
    shrink=0.72
)

cb.set_label(
    "MODIS AOD550"
)

plt.title(
"MODIS Terra AOD\nDay 353 (19 Dec 2023)"
)

plt.tight_layout()

plt.savefig(
"modis_aod_nsw.png",
dpi=300
)

plt.show()

