#!/usr/bin/env python3

from netCDF4 import Dataset
from wrf import getvar, ll_to_xy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# SITE
# --------------------------------------------------

LAT_SITE = -32.14
LON_SITE = 150.36

# --------------------------------------------------
# WRF FILES
# --------------------------------------------------

wrf_files = [
    "/mnt/scratch_lustre/duch/runpillaga/wrfout_d01_2023-12-16_00:00:00.nc",
    "/mnt/scratch_lustre/duch/runpillaga/wrfout_d01_2023-12-17_00:00:00.nc",
    "/mnt/scratch_lustre/duch/runpillaga/wrfout_d01_2023-12-18_00:00:00.nc"
]

wrf_times = []
wrf_pblh = []

for wrf_file in wrf_files:

    nc = Dataset(wrf_file)

    xy = ll_to_xy(nc, LAT_SITE, LON_SITE)
    ix = int(xy[0])
    iy = int(xy[1])

    pblh = getvar(nc, "PBLH", timeidx=None)

    times = getvar(nc, "times", timeidx=None)

    for t in range(len(times)):

        wrf_times.append(
            pd.to_datetime(str(times[t].values))
        )

        wrf_pblh.append(
            float(pblh[t, iy, ix])
        )

# UTC → AEST
wrf_times = pd.to_datetime(wrf_times) + pd.Timedelta(hours=10)

wrf_df = pd.DataFrame({
    "time": wrf_times,
    "PBLH_WRF": wrf_pblh
})

# --------------------------------------------------
# CEILOMETER FILES
# --------------------------------------------------

csv_files = [
    "L3_DEFAULT_0_20231216_Merriwa.csv",
    "L3_DEFAULT_0_20231217_Merriwa.csv",
    "L3_DEFAULT_0_20231218_Merriwa.csv",
    "L3_DEFAULT_0_20231219_Merriwa.csv"
]

dfs = []

for f in csv_files:

    df = pd.read_csv(f)

    df["Time"] = pd.to_datetime(
        df["# Time"],
        dayfirst=True
    )

    df["bl_height"] = pd.to_numeric(
        df["bl_height"],
        errors="coerce"
    )

    df.loc[df["bl_height"] < 0, "bl_height"] = np.nan

    dfs.append(df)

ceilo = pd.concat(dfs)

ceilo = ceilo.sort_values("Time")

# hourly average
ceilo_hourly = (
    ceilo
    .set_index("Time")
    .resample("1H")
    .median()
)

# --------------------------------------------------
# PLOT
# --------------------------------------------------

plt.figure(figsize=(14,6))

# raw ceilometer
plt.scatter(
    ceilo["Time"],
    ceilo["bl_height"],
    s=1,
    alpha=0.2,
    label="Ceilometer raw"
)

# hourly ceilometer
plt.plot(
    ceilo_hourly.index,
    ceilo_hourly["bl_height"],
    linewidth=2,
    label="Ceilometer hourly"
)

# WRF
plt.plot(
    wrf_df["time"],
    wrf_df["PBLH_WRF"],
    "-o",
    linewidth=2,
    label="WRF PBLH"
)

plt.ylabel("Boundary Layer Height (m)")
plt.xlabel("Date Time AEST")

plt.title("Merriwa PBL Height Comparison")

plt.grid(True)
plt.legend()

plt.tight_layout()

plt.show()


