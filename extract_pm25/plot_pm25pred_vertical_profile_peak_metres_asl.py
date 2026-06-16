#!/usr/bin/env python3

import glob
import numpy as np
import pandas as pd

from netCDF4 import Dataset
from wrf import (
    getvar,
    latlon_coords,
    ALL_TIMES
)

import matplotlib.pyplot as plt

# --------------------------------------------------
# files
# --------------------------------------------------

files = sorted(glob.glob(
    "/mnt/scratch_lustre/duch/runpillaga/wrfout_d01_2023-12-0[6-9]_00:00:00"
))

ncfiles = [Dataset(f) for f in files]

# --------------------------------------------------
# PM25
# --------------------------------------------------

pm25 = getvar(
    ncfiles,
    "PM2_5_DRY",
    timeidx=ALL_TIMES
)

z = getvar(
    ncfiles,
    "height_agl",
    timeidx=ALL_TIMES
)

times = pd.to_datetime(
    np.array(pm25.Time.values).astype(str)
)

times = times + pd.Timedelta(hours=10)

# --------------------------------------------------
# stations
# --------------------------------------------------

stations = {
    "Narrabri": (-30.318,149.829),
    "Gunnedah": (-30.982,150.261)
}

lats, lons = latlon_coords(pm25[0,0,:,:])

# --------------------------------------------------
# plot
# --------------------------------------------------

plt.figure(figsize=(8,8))

for site,(slat,slon) in stations.items():

    # find nearest grid point
    dist2 = (
        (lats.values - slat)**2 +
        (lons.values - slon)**2
    )

    iy, ix = np.unravel_index(
        np.argmin(dist2),
        dist2.shape
    )

    surface = pm25[:,0,iy,ix].values

    imax = np.argmax(surface)

    print()
    print(site)
    print("Peak PM2.5 =", surface[imax])
    print("Time =", times[imax])

    # vertical profile at peak time
    profile = pm25[imax,:,iy,ix].values
    height  = z[imax,:,iy,ix].values

    plt.plot(
        profile,
        height,
        marker='o',
        label=f"{site}"
    )

# --------------------------------------------------
# formatting
# --------------------------------------------------

plt.xlabel("PM2.5 (µg m$^{-3}$)")
plt.ylabel("Height AGL (m)")
plt.title("Vertical PM2.5 profile at model peak (AGL)")
plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig(
    "PM25_profile_model_peak_AGL.png",
    dpi=300
)

plt.show()

