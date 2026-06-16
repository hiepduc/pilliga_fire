#!/usr/bin/env python3

import glob
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from wrf import getvar, latlon_coords, ALL_TIMES
import matplotlib.pyplot as plt

files = sorted(glob.glob(
    "/mnt/scratch_lustre/duch/runpillaga/wrfout_d01_2023-12-0[6-9]_00:00:00"
))

stations = {
    "Narrabri": (-30.318,149.829),
    "Gunnedah": (-30.982,150.261)
}

ncfiles = [Dataset(f) for f in files]

pblh = getvar(
    ncfiles,
    "PBLH",
    timeidx=ALL_TIMES
)

lats, lons = latlon_coords(pblh)

times = pd.to_datetime(
    np.array(pblh.Time.values).astype(str)
)

times = times + pd.Timedelta(hours=10)

plt.figure(figsize=(14,5))

for stn,(slat,slon) in stations.items():

    dist2 = (
        (lats.values-slat)**2 +
        (lons.values-slon)**2
    )

    iy,ix = np.unravel_index(
        np.argmin(dist2),
        dist2.shape
    )

    pblh_station = pblh[:,iy,ix]

    plt.plot(
        times,
        pblh_station,
        label=stn
    )

plt.ylabel("PBLH (m)")
plt.title("PBLH at Narrabri and Gunnedah")
plt.grid()
plt.legend()

plt.tight_layout()

plt.savefig(
    "PBLH_Narrabri_Gunnedah.png",
    dpi=300
)

plt.show()

