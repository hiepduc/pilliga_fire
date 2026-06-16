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

lon_grid=np.arange(lonmin,lonmax,0.05)
lat_grid=np.arange(latmin,latmax,0.05)

LON,LAT=np.meshgrid(
    lon_grid,
    lat_grid
)

# ==========================
# FILES
# ==========================
files=sorted(
glob.glob(
"/mnt/scratch_lustre/duch/runpillaga/cams_merra_aod/MOD04_L2/2023/353/*.hdf"
)
)

print("Found",len(files),"files")

all_lon=[]
all_lat=[]
all_aod=[]

# ==========================
# READ MODIS
# ==========================
for f in files:

    try:

        hdf=SD(
            f,
            SDC.READ
        )

        names=hdf.datasets()

        if (
        "Optical_Depth_Land_And_Ocean"
        not in names
        ):
            continue

        aod=np.array(
            hdf.select(
            "Optical_Depth_Land_And_Ocean"
            )[:]
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

        attrs=(
            hdf
            .select(
            "Optical_Depth_Land_And_Ocean"
            )
            .attributes()
        )

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

        aod=aod.astype(float)

        aod[aod==fill]=np.nan

        aod=(
            aod-offset
        )*scale

        mask=(
            (lon>lonmin)&
            (lon<lonmax)&
            (lat>latmin)&
            (lat<latmax)&
            (aod>=0)&
            (aod<5)
        )

        n=mask.sum()

        if n>0:

            print(
                "using",
                f.split("/")[-1],
                n
            )

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
            str(e)
        )

# ==========================
# CHECK
# ==========================
if len(all_aod)==0:

    raise Exception(
        "No MODIS pixels found"
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
"Total pixels:",
len(aod)
)

# ==========================
# INTERPOLATE
# ==========================
modis=griddata(
    (lon,lat),
    aod,
    (LON,LAT),
    method="nearest"
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

gl=ax.gridlines(
draw_labels=True,
linestyle=":"
)

gl.top_labels=False
gl.right_labels=False

cb=plt.colorbar(
pcm,
shrink=0.72
)

cb.set_label(
"MODIS Terra AOD550"
)

plt.title(
"MODIS Terra AOD\n19 Dec 2023"
)

plt.tight_layout()

plt.savefig(
"modis_aod_nsw.png",
dpi=300
)

plt.show()

