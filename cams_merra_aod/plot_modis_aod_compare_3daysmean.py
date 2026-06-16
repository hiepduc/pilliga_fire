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
lonmin = 141
lonmax = 154

latmin = -37
latmax = -28

# output resolution
res = 0.05

lon_edges = np.arange(
    lonmin,
    lonmax + res,
    res
)

lat_edges = np.arange(
    latmin,
    latmax + res,
    res
)

lon_center = (
    lon_edges[:-1]
    +
    lon_edges[1:]
) / 2

lat_center = (
    lat_edges[:-1]
    +
    lat_edges[1:]
) / 2


# ==================================================
# DAYS
# ==================================================
days = [
    "351",   # 17 Dec
    "352",   # 18 Dec
    "353"    # 19 Dec
]

base_dir = (
"/mnt/scratch_lustre/duch/runpillaga/"
"cams_merra_aod/MOD04_L2/2023"
)

files = []

for day in days:

    ff = sorted(
        glob.glob(
            f"{base_dir}/{day}/*.hdf"
        )
    )

    print(
        day,
        len(ff),
        "files"
    )

    files.extend(
        ff
    )

print(
    "\nTotal files:",
    len(files)
)

# ==================================================
# STORAGE
# ==================================================
all_lon = []
all_lat = []
all_aod = []

# ==================================================
# READ MODIS
# ==================================================
for f in files:

    try:

        hdf = SD(
            f,
            SDC.READ
        )

        names = hdf.datasets()

        if (
            "Optical_Depth_Land_And_Ocean"
            not in names
        ):
            continue

        var = hdf.select(
            "Optical_Depth_Land_And_Ocean"
        )

        aod = np.array(
            var[:]
        )

        lat = np.array(
            hdf.select(
                "Latitude"
            )[:]
        )

        lon = np.array(
            hdf.select(
                "Longitude"
            )[:]
        )

        attrs = (
            var.attributes()
        )

        scale = attrs.get(
            "scale_factor",
            0.001
        )

        fill = attrs.get(
            "_FillValue",
            -9999
        )

        offset = attrs.get(
            "add_offset",
            0
        )

        aod = (
            aod
            .astype(float)
        )

        aod[
            aod == fill
        ] = np.nan

        aod = (
            aod
            - offset
        ) * scale

        mask = (

            (lon >= lonmin)
            &
            (lon <= lonmax)

            &
            (lat >= latmin)
            &
            (lat <= latmax)

            &
            np.isfinite(
                aod
            )

            &
            (aod >= 0)

            &
            (aod <= 5)

        )

        n = mask.sum()

        if n > 0:

            print(
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
            e
        )

# ==================================================
# CHECK
# ==================================================
if len(all_aod) == 0:

    raise Exception(
        "No MODIS pixels found"
    )

lon = np.concatenate(
    all_lon
)

lat = np.concatenate(
    all_lat
)

aod = np.concatenate(
    all_aod
)

print(
    "\nTotal pixels:",
    len(aod)
)

# ==================================================
# GRID TO AVERAGE
# ==================================================
print(
    "Computing 3-day mean..."
)

modis_mean, _, _, _ = (
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
fig = plt.figure(
    figsize=(10,8)
)

ax = plt.axes(
    projection=
    ccrs.PlateCarree()
)

pcm = ax.pcolormesh(

    lon_center,

    lat_center,

    modis_mean,

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

gl = ax.gridlines(
    draw_labels=True,
    linestyle=":"
)

gl.top_labels = False
gl.right_labels = False

cb = plt.colorbar(
    pcm,
    shrink=0.72
)

cb.set_label(
    "MODIS Terra AOD550"
)

plt.title(
"MODIS Terra AOD (3-day mean)\n17–19 Dec 2023"
)

plt.tight_layout()

outfile = (
"modis_aod_3day_mean_20231217_20231219.png"
)

plt.savefig(
outfile,
dpi=300
)

print(
"\nSaved:",
outfile
)

plt.show()

