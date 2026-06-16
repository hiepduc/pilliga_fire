#!/usr/bin/env python3

from netCDF4 import Dataset
from wrf import getvar, latlon_coords
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------
# USER SETTINGS
# -------------------------------------------------------

stations = {
    "Narrabri": {
        "lat": -30.318,
        "lon": 149.829,
        "file": "/mnt/scratch_lustre/duch/runpillaga/wrfout_d01_2023-12-08_00:00:00",
        "itime": 18,
        "title": "Narrabri Peak (9 Dec 2023 04:00 AEST)"
    },

    "Gunnedah": {
        "lat": -30.982,
        "lon": 150.261,
        "file": "/mnt/scratch_lustre/duch/runpillaga/wrfout_d01_2023-12-09_00:00:00",
        "itime": 5,
        "title": "Gunnedah Peak (9 Dec 2023 15:00 AEST)"
    }
}

# -------------------------------------------------------
# FIGURE
# -------------------------------------------------------

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12,8),
    sharey=True
)

for ax, (site, info) in zip(axes, stations.items()):

    nc = Dataset(info["file"])

    pm25 = getvar(
        nc,
        "PM2_5_DRY",
        timeidx=info["itime"]
    )

    lats, lons = latlon_coords(pm25[0,:,:])

    dist2 = (
        (lats.values - info["lat"])**2 +
        (lons.values - info["lon"])**2
    )

    iy, ix = np.unravel_index(
        np.argmin(dist2),
        dist2.shape
    )

    profile = pm25[:, iy, ix].values

    levels = np.arange(len(profile))

    print("\n", site)
    print(
        f"Grid point: ({iy},{ix}) "
        f"lat={lats.values[iy,ix]:.3f} "
        f"lon={lons.values[iy,ix]:.3f}"
    )

    print(
        "Surface PM2.5 =",
        profile[0]
    )

    ax.plot(
        profile,
        levels,
        marker="o"
    )

    ax.invert_yaxis()

    ax.grid(True)

    ax.set_xlabel(
        "PM2.5 (µg m$^{-3}$)"
    )

    ax.set_title(info["title"])

axes[0].set_ylabel(
    "WRF Vertical Level"
)

plt.suptitle(
    "WRF-Chem PM2.5 Vertical Profiles During Observed Peaks",
    fontsize=14
)

plt.tight_layout()

plt.savefig(
    "PM25_vertical_profiles_peak.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

