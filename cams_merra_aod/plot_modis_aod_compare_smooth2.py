#!/usr/bin/env python3

import glob
import numpy as np
from pyhdf.SD import SD, SDC
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ==================================================
# DOMAIN
# ==================================================
lonmin, lonmax = 141, 154
latmin, latmax = -37, -28

# finer output grid
res = 0.02

lon_grid = np.arange(
    lonmin,
    lonmax,
    res
)

lat_grid = np.arange(
    latmin,
    latmax,
    res
)

LON, LAT = np.meshgrid(
    lon_grid,
    lat_grid
)

# ==================================================
# FILES
# ==================================================
files = sorted(
    glob.glob(
"/mnt/scratch_lustre/duch/runpillaga/cams_merra_aod/MOD04_L2/2023/353/*.hdf"
    )
)

print("Found", len(files), "files")

all_lon = []
all_lat = []
all_aod = []

# ==================================================
# READ
# ==================================================
for f in files:

    try:

        hdf = SD(
            f,
            SDC.READ
        )

        if (
            "Optical_Depth_Land_And_Ocean"
            not in hdf.datasets()
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

        attrs = var.attributes()

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

        aod = aod.astype(
            float
        )

        aod[aod == fill] = np.nan

        aod = (
            aod - offset
        ) * scale

        mask = (
            (lon > lonmin)
            &
            (lon < lonmax)
            &
            (lat > latmin)
            &
            (lat < latmax)
            &
            np.isfinite(aod)
            &
            (aod >= 0)
            &
            (aod < 5)
        )

        if mask.sum() > 0:

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
    "Total pixels:",
    len(aod)
)

# ==================================================
# INTERPOLATE
# ==================================================
print(
    "Interpolating..."
)

modis = griddata(
    (
        lon,
        lat
    ),
    aod,
    (
        LON,
        LAT
    ),
    method="linear"
)

# fill only holes
holes = np.isnan(
    modis
)

fill = griddata(
    (
        lon,
        lat
    ),
    aod,
    (
        LON,
        LAT
    ),
    method="nearest"
)

modis[holes] = fill[
    holes
]

# ==================================================
# LIGHT SMOOTHING
# ==================================================
print(
    "Smoothing..."
)

modis = gaussian_filter(
    modis,
    sigma=0.7
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
    lon_grid,
    lat_grid,
    modis,
    cmap="turbo",
    vmin=0,
    vmax=1,
    shading="gouraud"
)

ax.coastlines(
    resolution="10m",
    linewidth=0.8
)

ax.add_feature(
    cfeature.BORDERS,
    linewidth=0.4
)

ax.add_feature(
    cfeature.STATES,
    linewidth=0.5
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
    shrink=0.72,
    pad=0.02,
    aspect=35
)

cb.set_label(
    "MODIS Terra AOD550"
)

plt.title(
    "MODIS Terra AOD\n19 Dec 2023"
)

plt.tight_layout()

plt.savefig(
    "modis_aod_nsw_smooth.png",
    dpi=300
)

plt.show()

