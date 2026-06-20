#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from netCDF4 import Dataset

# =====================================================
# SETTINGS
# =====================================================

LAT_SITE = -33.89
LON_SITE = 151.05

UTC_OFFSET = 11   # AEDT in Dec

WRF_FILE = (
"/mnt/scratch_lustre/duch/runpillaga/"
"wrfout_d02_2023-12-19_01:00:00"
)

CEILO_FILE = (
"/mnt/scratch_lustre/duch/runpillaga/extract_tibl/ceilodata/"
"L3_DEFAULT__202312190000_1_360_1_3120_10_30_4000_3_0_1_500_1000_4000_60.nc"
)

MAX_HEIGHT = 4000

# =====================================================
# OPEN WRF
# =====================================================

f = Dataset(WRF_FILE)

# -----------------------------------------------------
# LAT/LON
# -----------------------------------------------------

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

# =====================================================
# TIME
# =====================================================

times_raw = f.variables["Times"][:]

print("Times shape =", times_raw.shape)
print("First raw time =", times_raw[0])

times = []

for row in times_raw:

    try:

        s = "".join(
            row.astype(str)
        )

    except:

        s = "".join(
            [str(x) for x in row]
        )

    print("Decoded:", repr(s))

    times.append(
        pd.to_datetime(
            s,
            format="%Y-%m-%d_%H:%M:%S"
        )
    )

times = pd.DatetimeIndex(times)

times = times + pd.Timedelta(hours=UTC_OFFSET)

# =====================================================
# HEIGHT
# =====================================================

g = 9.81

PH = f.variables["PH"][:,:,iy,ix]
PHB = f.variables["PHB"][:,:,iy,ix]

z_stag = (PH + PHB)/g

z = 0.5*(z_stag[:,:-1] + z_stag[:,1:])

# =====================================================
# THETA
# =====================================================

theta = f.variables["T"][:,:,iy,ix] + 300

# =====================================================
# VERTICAL VELOCITY
# =====================================================

W = f.variables["W"][:,:,iy,ix]

W = 0.5*(W[:,:-1] + W[:,1:])

# =====================================================
# PBLH
# =====================================================

PBLH = f.variables["PBLH"][:,iy,ix]

# =====================================================
# dtheta/dz
# =====================================================

dtdz = np.full_like(theta,np.nan)

for t in range(theta.shape[0]):

    dtdz[t,:] = np.gradient(
        theta[t,:],
        z[t,:]
    )

# =====================================================
# TIBL ESTIMATE
# =====================================================

TIBL = np.full(theta.shape[0],np.nan)

for t in range(theta.shape[0]):

    zz = z[t,:]
    th = theta[t,:]

    mask = zz < 2000

    zz = zz[mask]
    th = th[mask]

    grad = np.gradient(th,zz)

    idx = np.argmax(grad)

    TIBL[t] = zz[idx]

# =====================================================
# CEILOMETER
# =====================================================

c = Dataset(CEILO_FILE)

ctime = pd.to_datetime(
    c.variables["time"][:],
    unit="s",
    origin="unix"
)

ctime = ctime + pd.Timedelta(hours=UTC_OFFSET)

rng = c.variables["range"][:]

bs = np.array(
    c.variables["Bs_profile_data"][:],
    dtype=float
)

bs[bs <= 0] = np.nan

bs_log = np.log10(bs)

bl = np.array(
    c.variables["bl_height"][:],
    dtype=float
)

bl[bl < 0] = np.nan

bl1 = bl[:,0]
bl2 = bl[:,1]

# =====================================================
# FIGURE
# =====================================================

fig, ax = plt.subplots(
    4,
    1,
    figsize=(16,16),
    sharex=True
)

# =====================================================
# PANEL 1
# =====================================================

pcm = ax[0].pcolormesh(
    ctime,
    rng,
    bs_log.T,
    shading="auto"
)

ax[0].plot(
    ctime,
    bl1,
    'w',
    lw=2,
    label="Ceilo BL1"
)

ax[0].plot(
    ctime,
    bl2,
    'c',
    lw=2,
    label="Ceilo BL2"
)

ax[0].plot(
    times,
    PBLH,
    'r',
    lw=2,
    label="WRF PBLH"
)

ax[0].plot(
    times,
    TIBL,
    'y',
    lw=2,
    label="WRF TIBL"
)

ax[0].set_ylim(0,MAX_HEIGHT)

ax[0].set_title(
    "Ceilometer Backscatter"
)

ax[0].legend()

plt.colorbar(
    pcm,
    ax=ax[0],
    label="log10(backscatter)"
)

# =====================================================
# PANEL 2
# =====================================================

pcm = ax[1].pcolormesh(
    times,
    z[0,:],
    theta.T,
    shading="auto"
)

ax[1].plot(times,PBLH,'k',lw=2)
ax[1].plot(times,TIBL,'w',lw=2)

ax[1].set_ylim(0,MAX_HEIGHT)

ax[1].set_title("WRF Potential Temperature")

plt.colorbar(
    pcm,
    ax=ax[1],
    label="K"
)

# =====================================================
# PANEL 3
# =====================================================

pcm = ax[2].pcolormesh(
    times,
    z[0,:],
    W.T,
    shading="auto"
)

ax[2].plot(times,PBLH,'k',lw=2)
ax[2].plot(times,TIBL,'w',lw=2)

ax[2].set_ylim(0,MAX_HEIGHT)

ax[2].set_title("WRF Vertical Velocity")

plt.colorbar(
    pcm,
    ax=ax[2],
    label="m/s"
)

# =====================================================
# PANEL 4
# =====================================================

pcm = ax[3].pcolormesh(
    times,
    z[0,:],
    dtdz.T,
    shading="auto"
)

ax[3].plot(times,PBLH,'k',lw=2)
ax[3].plot(times,TIBL,'w',lw=2)

ax[3].set_ylim(0,MAX_HEIGHT)

ax[3].set_title("WRF Stability dθ/dz")

plt.colorbar(
    pcm,
    ax=ax[3],
    label="K/m"
)

# =====================================================
# FORMAT
# =====================================================

for a in ax:
    a.set_ylabel("Height (m)")
    a.grid()

plt.tight_layout()

plt.savefig(
    "Lidcombe_19Dec2023_4Panel_TIBL.png",
    dpi=300
)

plt.show()

