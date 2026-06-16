#!/usr/bin/env python3

import glob
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from wrf import getvar, latlon_coords, ALL_TIMES

# -------------------------------------------------------
# INPUT
# -------------------------------------------------------

file = "/mnt/scratch_lustre/duch/runpillaga/wrfout_d01_2023-12-09_00:00:00"

itime = 5  # example time index (change as needed)

stations = {
    "Narrabri": (-30.318, 149.829),
    "Gunnedah": (-30.982, 150.261)
}

# -------------------------------------------------------
# OPEN FILE
# -------------------------------------------------------

nc = Dataset(file)

pm25 = getvar(nc, "PM2_5_DRY", timeidx=itime)
z = getvar(nc, "height_agl", timeidx=itime)

lats, lons = latlon_coords(pm25[0, :, :])

# -------------------------------------------------------
# PLOT
# -------------------------------------------------------

plt.figure(figsize=(6, 8))

for name, (slat, slon) in stations.items():

    dist2 = (lats.values - slat)**2 + (lons.values - slon)**2
    iy, ix = np.unravel_index(np.argmin(dist2), dist2.shape)

    profile = pm25[:, iy, ix].values
    height = z[:, iy, ix].values

    print(f"\n{name}")
    print("Max PM2.5:", np.max(profile))
    print("Height of max:", height[np.argmax(profile)])

    plt.plot(profile, height, marker="o", label=name)

plt.xlabel("PM2.5 (µg m$^{-3}$)")
plt.ylabel("Height AGL (m)")
plt.title("Vertical PM2.5 Profiles (WRF-Chem)")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("PM25_profile_heightAGL.png", dpi=300)
plt.show()

