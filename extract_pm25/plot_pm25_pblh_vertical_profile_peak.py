#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from wrf import getvar, latlon_coords

file = "/mnt/scratch_lustre/duch/runpillaga/wrfout_d01_2023-12-09_00:00:00"
itime = 5

stations = {
    "Narrabri": (-30.318, 149.829),
    "Gunnedah": (-30.982, 150.261)
}

nc = Dataset(file)

pm25 = getvar(nc, "PM2_5_DRY", timeidx=itime)
z = getvar(nc, "height_agl", timeidx=itime)
pblh = getvar(nc, "PBLH", timeidx=itime)

lats, lons = latlon_coords(pm25[0, :, :])

plt.figure(figsize=(6, 8))

for name, (slat, slon) in stations.items():

    dist2 = (lats.values - slat)**2 + (lons.values - slon)**2
    iy, ix = np.unravel_index(np.argmin(dist2), dist2.shape)

    profile = pm25[:, iy, ix].values
    height = z[:, iy, ix].values

    pblh_val = float(pblh[iy, ix])

    max_i = np.argmax(profile)

    print(f"\n{name}")
    print("Max PM2.5:", profile[max_i])
    print("Height of max (m):", height[max_i])
    print("PBLH (m):", pblh_val)

    plt.plot(profile, height, marker="o", label=name)

    # PBLH line
    plt.axhline(
        pblh_val,
        linestyle="--",
        linewidth=2,
        label=f"{name} PBLH"
    )

plt.xlabel("PM2.5 (µg m$^{-3}$)")
plt.ylabel("Height AGL (m)")
plt.title("PM2.5 Profiles with PBLH Overlay")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("PM25_profile_with_PBLH.png", dpi=300)
plt.show()

