#!/usr/bin/env python3

import glob
import numpy as np
from pyhdf.SD import SD, SDC
from scipy.stats import binned_statistic_2d
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# =====================================================
# DOMAIN
# =====================================================

lonmin, lonmax = 141, 154
latmin, latmax = -37, -28

# finer grid
res = 0.05

lon_edges = np.arange(lonmin, lonmax + res, res)
lat_edges = np.arange(latmin, latmax + res, res)

lon_grid = (
    lon_edges[:-1]
    + lon_edges[1:]
) / 2

lat_grid = (
    lat_edges[:-1]
    + lat_edges[1:]
) / 2

# =====================================================
# INPUT DAYS
# =====================================================

base = (
"/mnt/scratch_lustre/duch/runpillaga/"
"cams_merra_aod/MOD04_L2/2023"
)

days = [
    "351",   # 17 Dec
    "352",   # 18 Dec
    "353"    # 19 Dec
]

files = []

for d in days:

    files += sorted(
        glob.glob(
            f"{base}/{d}/*.hdf"
        )
    )

print("Found", len(files), "files")

# =====================================================
# READ MODIS
# =====================================================

all_lon = []
all_lat = []
all_aod = []

for f in files:

    try:

        hdf = SD(f, SDC.READ)

        if (
        "Optical_Depth_Land_And_Ocean"
        not in hdf.datasets()
        ):
            continue

        aod = hdf.select(
            "Optical_Depth_Land_And_Ocean"
        )[:]

        lat = hdf.select(
            "Latitude"
        )[:]

        lon = hdf.select(
            "Longitude"
        )[:]

        attrs = (
            hdf
            .select(
                "Optical_Depth_Land_And_Ocean"
            )
            .attributes()
        )

        scale = attrs.get(
            "scale_factor",
            0.001
        )

        offset = attrs.get(
            "add_offset",
            0
        )

        fill = attrs.get(
            "_FillValue",
            -9999
        )

        aod = aod.astype(float)

        aod[aod == fill] = np.nan

        aod = (
            aod - offset
        ) * scale

        mask = (
            (lon >= lonmin) &
            (lon <= lonmax) &
            (lat >= latmin) &
            (lat <= latmax) &
            (aod > 0) &
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

            print(
                f.split("/")[-1],
                mask.sum()
            )

    except Exception as e:

        print(
            "skip",
            f,
            e
        )

# =====================================================
# MERGE
# =====================================================

lon = np.concatenate(all_lon)
lat = np.concatenate(all_lat)
aod = np.concatenate(all_aod)

print(
    "Total pixels:",
    len(aod)
)

# =====================================================
# GRID BY MEAN
# =====================================================

modis, _, _, _ = (
    binned_statistic_2d(
        lon,
        lat,
        aod,
        statistic="mean",
        bins=[
            lon_edges,
            lat_edges
        ]
    )
)

modis = modis.T

# =====================================================
# LIGHT SMOOTH
# =====================================================

mask = np.isfinite(modis)

filled = np.nan_to_num(
    modis,
    nan=0
)

smooth = gaussian_filter(
    filled,
    sigma=0.8
)

weight = gaussian_filter(
    mask.astype(float),
    sigma=0.8
)

modis = smooth / np.maximum(
    weight,
    0.001
)

modis[weight < 0.2] = np.nan

# =====================================================
# PLOT
# =====================================================

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
    shading="auto"
)

ax.coastlines(
    resolution="10m"
)

ax.add_feature(
    cfeature.BORDERS
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
    "3-day mean MODIS AOD550"
)

plt.title(
"MODIS Terra AOD\n17–19 Dec 2023 mean"
)

plt.tight_layout()

plt.savefig(
    "modis_aod_3day_mean.png",
    dpi=300
)

plt.show()

