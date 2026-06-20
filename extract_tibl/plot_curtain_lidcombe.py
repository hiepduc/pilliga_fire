#!/usr/bin/env python3

from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

f = Dataset(
"/mnt/scratch_lustre/duch/runpillaga/extract_tibl/ceilodata/L3_DEFAULT__202312190000_1_360_1_3120_10_30_4000_3_0_1_500_1000_4000_60.nc"
)

time_sec = f.variables["time"][:]

time = pd.to_datetime(
time_sec,
unit="s",
origin="unix"
)

bs = f.variables["Bs_profile_data"][:]

rng = f.variables["range"][:]

bs = np.array(bs,dtype=float)

bs[bs < 0] = np.nan

plt.figure(figsize=(14,6))

plt.pcolormesh(
time,
rng,
np.log10(bs.T),
shading="auto"
)

plt.colorbar(label="log10(backscatter)")

plt.ylabel("Height (m)")
plt.title("Lidcombe Ceilometer Backscatter")

plt.ylim(0,4000)

plt.show()

