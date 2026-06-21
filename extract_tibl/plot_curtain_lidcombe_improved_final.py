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

CEILO_FILES = [

"/mnt/scratch_lustre/duch/runpillaga/extract_tibl/ceilodata/"
"L3_DEFAULT__202312190000_1_360_1_3120_10_30_4000_3_0_1_500_1000_4000_60.nc",

"/mnt/scratch_lustre/duch/runpillaga/extract_tibl/ceilodata/"
"L3_DEFAULT__202312200000_1_360_1_3120_10_30_4000_3_0_1_500_1000_4000_60.nc",

"/mnt/scratch_lustre/duch/runpillaga/extract_tibl/ceilodata/"
"L3_DEFAULT__202312210000_1_360_1_3120_10_30_4000_3_0_1_500_1000_4000_60.nc"

]

MAX_HEIGHT = 4000

# ==========================================================
# OPEN WRF
# ==========================================================

f = Dataset(WRF_FILE)

# ==========================================================
# FIND LIDCOMBE GRID POINT
# ==========================================================

lat = f.variables["XLAT"][0,:,:]
lon = f.variables["XLONG"][0,:,:]

dist = np.sqrt(
    (lat - LAT_SITE)**2 +
    (lon - LON_SITE)**2
)

iy, ix = np.unravel_index(
    np.argmin(dist),
    dist.shape
)

print("Grid =", iy, ix)

# ==========================================================
# WRF TIMES
# ==========================================================

times = []

for row in f.variables["Times"][:]:

    s = b"".join(row).decode("utf-8")

    s = s.replace("_"," ")

    times.append(pd.to_datetime(s))

times = pd.to_datetime(times)

times = times + pd.Timedelta(hours=UTC_OFFSET)

# ==========================================================
# HEIGHT
# ==========================================================

g = 9.81

PH  = f.variables["PH"][:,:,iy,ix]
PHB = f.variables["PHB"][:,:,iy,ix]

z_stag = (PH + PHB)/g

z = 0.5 * (
    z_stag[:,:-1] +
    z_stag[:,1:]
)

# ==========================================================
# THETA
# ==========================================================

theta = (
    f.variables["T"][:,:,iy,ix]
    + 300.0
)

# ==========================================================
# PBLH
# ==========================================================

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
# READ CEILOMETER FILES
# ==========================================================

ctime_all = []
bs_all = []
bl1_all = []
bl2_all = []

for fname in CEILO_FILES:

    print("Reading", fname)

    c = Dataset(fname)

    ct = pd.to_datetime(
        c.variables["time"][:],
        unit="s",
        origin="unix"
    )

    ct = ct + pd.Timedelta(hours=UTC_OFFSET)

    bs = np.array(
        c.variables["Bs_profile_data"][:],
        dtype=float
    )

    bs[bs <= 0] = np.nan

    bl = np.array(
        c.variables["bl_height"][:],
        dtype=float
    )

    bl[bl < 0] = np.nan

    ctime_all.append(ct)
    bs_all.append(bs)

    bl1_all.append(bl[:,0])
    bl2_all.append(bl[:,1])

    rng = c.variables["range"][:]

# ==========================================================
# CONCATENATE
# ==========================================================

ctime = pd.DatetimeIndex(
    np.concatenate(ctime_all)
)

bs = np.vstack(bs_all)

bl1 = np.concatenate(bl1_all)

bl2 = np.concatenate(bl2_all)

bs_log = np.log10(bs)

# ==========================================================
# PLOT
# ==========================================================

fig, ax = plt.subplots(
    figsize=(18,8)
)

pcm = ax.pcolormesh(
    ctime,
    rng,
    bs_log.T,
    shading="auto",
    vmin=1,
    vmax=5
)

# ----------------------------------------------------------
# CEILOMETER LAYERS
# ----------------------------------------------------------

ax.plot(
    ctime,
    bl1,
    color="white",
    lw=2,
    label="Ceilo BL1"
)

ax.plot(
    ctime,
    bl2,
    color="cyan",
    lw=2,
    label="Ceilo BL2"
)

# ----------------------------------------------------------
# WRF
# ----------------------------------------------------------

ax.plot(
    times,
    PBLH,
    color="red",
    lw=2.5,
    label="WRF PBLH"
)

ax.plot(
    times,
    TIBL,
    color="yellow",
    lw=2.5,
    label="WRF TIBL"
)

# ----------------------------------------------------------
# FORMAT
# ----------------------------------------------------------

ax.set_ylim(0, MAX_HEIGHT)

#ax.set_xlim(
#    pd.Timestamp("2023-12-19 12:00"),
#    pd.Timestamp("2023-12-20 11:00")
#)

ax.set_xlim(
    ctime.min(),
    ctime.max()
)

ax.set_ylabel("Height (m AGL)")

ax.set_title(
    "Lidcombe CL51 Backscatter vs WRF PBLH/TIBL\n"
    "19 Dec 2023 Smoke Event"
)

ax.grid(True)

ax.legend(
    loc="upper right",
    fontsize=11
)

cbar = plt.colorbar(
    pcm,
    ax=ax
)

cbar.set_label(
    "log10(backscatter)"
)

plt.tight_layout()

plt.savefig(
    "Lidcombe_Backscatter_WRF_PBLH_TIBL.png",
    dpi=300
)

plt.show()

