#!/usr/bin/env python3

import glob
import numpy as np
from pyhdf.SD import SD, SDC
from scipy.stats import binned_statistic_2d
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ==================================================
# DOMAIN
# ==================================================
lonmin, lonmax = 141,154
latmin, latmax = -37,-28

res=0.05

lon_edges=np.arange(
    lonmin,
    lonmax+res,
    res
)

lat_edges=np.arange(
    latmin,
    latmax+res,
    res
)

lon_center=(
    lon_edges[:-1]
    +
    lon_edges[1:]
)/2

lat_center=(
    lat_edges[:-1]
    +
    lat_edges[1:]
)/2

# ==================================================
# DAYS
# ==================================================
days=[
"352",
"353"
]

files=[]

for d in days:

    files.extend(
        sorted(
        glob.glob(
f"/mnt/scratch_lustre/duch/runpillaga/cams_merra_aod/MOD04_L2/2023/{d}/*.hdf"
        )
    )

print( "Found", len(files), "files")

all_lon=[]
all_lat=[]
all_aod=[]

# ==================================================
# READ
# ==================================================
for f in files:

    try:

        hdf=SD(
            f,
            SDC.READ
        )

        if (
        "Optical_Depth_Land_And_Ocean"
        not in hdf.datasets()
        ):
            continue

        var=hdf.select(
            "Optical_Depth_Land_And_Ocean"
        )

        aod=np.array(
            var[:]
        )

        lat=np.array(
            hdf.select(
                "Latitude"
            )[:]
        )

        lon=np.array(
            hdf.select(
                "Longitude"
            )[:]
        )

        attrs=var.attributes()

        scale=attrs.get(
            "scale_factor",
            0.001
        )

        fill=attrs.get(
            "_FillValue",
            -9999
        )

        offset=attrs.get(
            "add_offset",
            0
        )

        aod=aod.astype(
            float
        )

        aod[aod==fill]=np.nan

        aod=(
            aod-offset
        )*scale

        mask=(

            (lon>=lonmin)&
            (lon<=lonmax)&
            (lat>=latmin)&
            (lat<=latmax)&
            np.isfinite(aod)&
            (aod>=0)&
            (aod<5)

        )

        if mask.sum()>0:

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

        print(
            "skip",
            f,
            e
        )

lon=np.concatenate(
all_lon
)

lat=np.concatenate(
all_lat
)

aod=np.concatenate(
all_aod
)

print(
"Pixels:",
len(aod)
)

# ==================================================
# BIN AVERAGE
# ==================================================
print(
"Creating composite..."
)

grid,_,_,_=(
binned_statistic_2d(
    lat,
    lon,
    aod,
    statistic="mean",
    bins=[
        lat_edges,
        lon_edges
    ]
)
)

# ==================================================
# PLOT
# ==================================================
fig=plt.figure(
figsize=(10,8)
)

ax=plt.axes(
projection=ccrs.PlateCarree()
)

pcm=ax.pcolormesh(
    lon_center,
    lat_center,
    grid,
    cmap="turbo",
    vmin=0,
    vmax=1,
    shading="auto"
)

ax.coastlines(
resolution="10m"
)

ax.add_feature(
cfeature.STATES
)

ax.set_extent(
[
lonmin,
lonmax,
latmin,
latmax
]
)

cb=plt.colorbar(
pcm,
shrink=0.72
)

cb.set_label(
"MODIS Terra AOD550"
)

plt.title(
"MODIS Terra AOD Composite\n18–19 Dec 2023"
)

plt.tight_layout()

plt.savefig(
"modis_aod_composite_18_19_dec.png",
dpi=300
)

plt.show()

