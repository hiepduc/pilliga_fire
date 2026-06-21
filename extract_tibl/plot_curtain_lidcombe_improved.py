#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from netCDF4 import Dataset

# ==========================================================
# SETTINGS
# ==========================================================

LAT_SITE = -33.89
LON_SITE = 151.05

UTC_OFFSET = 11

WRF_FILE = (
"/mnt/scratch_lustre/duch/runpillaga/"
"wrfout_d02_2023-12-19_01:00:00"
)

CEILO_FILE = (
"/mnt/scratch_lustre/duch/runpillaga/extract_tibl/ceilodata/"
"L3_DEFAULT__202312190000_1_360_1_3120_10_30_4000_3_0_1_500_1000_4000_60.nc"
)

# ==========================================================
# WRF
# ==========================================================

f = Dataset(WRF_FILE)

# ----------------------------------------------------------
# grid point
# ----------------------------------------------------------

xlat = f.variables["XLAT"]

if len(xlat.shape) == 3:
    lat = xlat[0,:,:]
    lon = f.variables["XLONG"][0,:,:]
else:
    lat = xlat[:,:]
    lon = f.variables["XLONG"][:,:]

dist = np.sqrt(
    (lat-LAT_SITE)**2 +
    (lon-LON_SITE)**2
)

iy, ix = np.unravel_index(
    np.argmin(dist),
    dist.shape
)

print("Grid =", ix, iy)

# ----------------------------------------------------------
# time
# ----------------------------------------------------------

times = []

for row in f.variables["Times"][:]:

    s = b"".join(row).decode("utf-8")

    s = s.replace("_"," ")

    times.append(pd.to_datetime(s))

times = pd.to_datetime(times)

times = times + pd.Timedelta(hours=UTC_OFFSET)

# ----------------------------------------------------------
# heights
# ----------------------------------------------------------

g = 9.81

PH = f.variables["PH"][:,:,iy,ix]
PHB = f.variables["PHB"][:,:,iy,ix]

z_stag = (PH+PHB)/g

z = 0.5*(
    z_stag[:,:-1] +
    z_stag[:,1:]
)

# ----------------------------------------------------------
# theta
# ----------------------------------------------------------

theta = (
    f.variables["T"][:,:,iy,ix]
    + 300.0
)

# ----------------------------------------------------------
# PBLH
# ----------------------------------------------------------

PBLH = f.variables["PBLH"][:,iy,ix]

# ==========================================================
# TIBL ESTIMATE
# ==========================================================

TIBL = np.full(len(times), np.nan)

for t in range(len(times)):

    zz = z[t]
    th = theta[t]

    mask = (
        np.isfinite(zz)
        &
        np.isfinite(th)
        &
        (zz < 2500)
    )

    zz = zz[mask]
    th = th[mask]

    if len(zz) < 5:
        continue

    grad = np.gradient(th, zz)

    idx = np.argmax(grad)

    TIBL[t] = zz[idx]

# ==========================================================
# CEILOMETER
# ==========================================================

c = Dataset(CEILO_FILE)

ctime = pd.to_datetime(
    c.variables["time"][:],
    unit="s",
    origin="unix"
)

ctime = ctime + pd.Timedelta(hours=UTC_OFFSET)

rng = c.variables["range"][:]

# ----------------------------------------------------------
# backscatter
# ----------------------------------------------------------

bs = np.array(
    c.variables["Bs_profile_data"][:],
    dtype=float
)

bs[bs <= 0] = np.nan

bs_log = np.log10(bs)

# ----------------------------------------------------------
# boundary layers
# ----------------------------------------------------------

bl = np.array(
    c.variables["bl_height"][:],
    dtype=float
)

bl[bl < 0] = np.nan

BL1 = bl[:,0]
BL2 = bl[:,1]

# ==========================================================
# PLOT
# ==========================================================

plt.figure(figsize=(16,8))

pcm = plt.pcolormesh(
    ctime,
    rng,
    bs_log.T,
    shading="auto"
)

plt.colorbar(
    pcm,
    label="log10(backscatter)"
)

# ----------------------------------------------------------
# Ceilo layers
# ----------------------------------------------------------

plt.plot(
    ctime,
    BL1,
    color="white",
    lw=2,
    label="Ceilo BL1"
)

plt.plot(
    ctime,
    BL2,
    color="cyan",
    lw=2,
    label="Ceilo BL2"
)

# ----------------------------------------------------------
# WRF
# ----------------------------------------------------------

plt.plot(
    times,
    PBLH,
    color="red",
    lw=2,
    label="WRF PBLH"
)

plt.plot(
    times,
    TIBL,
    color="yellow",
    lw=2,
    label="WRF TIBL"
)

plt.ylim(0,4000)

plt.ylabel("Height (m AGL)")

plt.title(
    "Lidcombe CL51 Backscatter vs WRF PBLH/TIBL\n"
    "19 Dec 2023"
)

plt.legend(loc="upper right")

plt.grid()

plt.tight_layout()

plt.savefig(
    "Lidcombe_WRF_vs_Ceilometer_19Dec2023.png",
    dpi=300
)

plt.show()


