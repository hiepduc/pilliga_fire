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
"L3_DEFAULT__202312200000_1_360_1_3120_10_30_4000_3_0_1_500_1000_4000_60.nc"

]

MAX_HEIGHT = 4000

# ==========================================================
# OPEN WRF
# ==========================================================

f = Dataset(WRF_FILE)

# ==========================================================
# FIND GRID
# ==========================================================

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
# W
# ==========================================================

W_stag = f.variables["W"][:,:,iy,ix]

W = 0.5 * (
    W_stag[:,:-1] +
    W_stag[:,1:]
)

# ==========================================================
# PBLH
# ==========================================================

PBLH = f.variables["PBLH"][:,iy,ix]

# ==========================================================
# dtheta/dz
# ==========================================================

dtdz = np.full_like(theta, np.nan)

for t in range(theta.shape[0]):

    dtdz[t,:] = np.gradient(
        theta[t,:],
        z[t,:]
    )

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
# READ CEILOMETER
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

ctime = pd.DatetimeIndex(
    np.concatenate(ctime_all)
)

bs = np.vstack(bs_all)

bl1 = np.concatenate(bl1_all)

bl2 = np.concatenate(bl2_all)

bs_log = np.log10(bs)

# ==========================================================
# FIGURE
# ==========================================================

fig, ax = plt.subplots(
    4,
    1,
    figsize=(18,18),
    sharex=True
)

# ==========================================================
# PANEL 1
# ==========================================================

pcm = ax[0].pcolormesh(
    ctime,
    rng,
    bs_log.T,
    shading="auto",
    vmin=1,
    vmax=5
)

ax[0].plot(
    ctime,
    bl1,
    color="white",
    lw=2,
    label="Ceilo BL1"
)

ax[0].plot(
    ctime,
    bl2,
    color="cyan",
    lw=2,
    label="Ceilo BL2"
)

ax[0].plot(
    times,
    PBLH,
    color="red",
    lw=2,
    label="WRF PBLH"
)

ax[0].plot(
    times,
    TIBL,
    color="yellow",
    lw=2,
    label="WRF TIBL"
)

ax[0].set_ylim(0,MAX_HEIGHT)

ax[0].set_title(
    "Lidcombe CL51 Backscatter + Ceilo BL1/BL2 + WRF PBLH/TIBL"
)

ax[0].legend(loc="upper right")

plt.colorbar(
    pcm,
    ax=ax[0],
    label="log10(backscatter)"
)

# ==========================================================
# PANEL 2
# ==========================================================

pcm = ax[1].pcolormesh(
    times,
    z[0,:],
    theta.T,
    shading="auto"
)

ax[1].plot(times,PBLH,'k',lw=2)
ax[1].plot(times,TIBL,'w',lw=2)

ax[1].set_ylim(0,MAX_HEIGHT)

ax[1].set_title(
    "WRF Potential Temperature"
)

plt.colorbar(
    pcm,
    ax=ax[1],
    label="K"
)

# ==========================================================
# PANEL 3
# ==========================================================

pcm = ax[2].pcolormesh(
    times,
    z[0,:],
    dtdz.T,
    shading="auto"
)

ax[2].plot(times,PBLH,'k',lw=2)
ax[2].plot(times,TIBL,'w',lw=2)

ax[2].set_ylim(0,MAX_HEIGHT)

ax[2].set_title(
    "WRF Stability dθ/dz"
)

plt.colorbar(
    pcm,
    ax=ax[2],
    label="K m$^{-1}$"
)

# ==========================================================
# PANEL 4
# ==========================================================

pcm = ax[3].pcolormesh(
    times,
    z[0,:],
    W.T,
    shading="auto",
    vmin=-2,
    vmax=2
)

ax[3].plot(times,PBLH,'k',lw=2)
ax[3].plot(times,TIBL,'w',lw=2)

ax[3].set_ylim(0,MAX_HEIGHT)

ax[3].set_title(
    "WRF Vertical Velocity"
)

plt.colorbar(
    pcm,
    ax=ax[3],
    label="m s$^{-1}$"
)

# ==========================================================
# ZOOM TO SMOKE EVENT
# ==========================================================

for a in ax:

    a.set_xlim(
        pd.Timestamp("2023-12-19 12:00"),
        pd.Timestamp("2023-12-20 11:00")
    )

    a.grid(True)

    a.set_ylabel("Height (m)")

# ==========================================================
# SAVE
# ==========================================================

plt.tight_layout()

plt.savefig(
    "Lidcombe_4Panel_WRF_Ceilometer_19Dec2023.png",
    dpi=300
)

plt.show()

