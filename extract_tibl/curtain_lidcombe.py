#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from netCDF4 import Dataset

# =====================================================
# SETTINGS
# =====================================================

WRF_FILE = "/mnt/scratch_lustre/duch/runpillaga/wrfout_d02_2023-12-08_00:00:00"

LAT_SITE = -33.89
LON_SITE = 151.05

UTC_OFFSET = 10

# =====================================================
# OPEN FILE
# =====================================================

ds = Dataset(WRF_FILE)

# =====================================================
# GRID LOCATION
# =====================================================

xlat = ds.variables["XLAT"][:]
xlon = ds.variables["XLONG"][:]

if xlat.ndim == 3:
    lat = xlat[0]
    lon = xlon[0]
else:
    lat = xlat
    lon = xlon

dist = (lat - LAT_SITE)**2 + (lon - LON_SITE)**2

iy, ix = np.unravel_index(
    np.argmin(dist),
    dist.shape
)

print("Grid =", ix, iy)

# =====================================================
# TIMES
# =====================================================

times_raw = ds.variables["Times"][:]

times = []

for row in times_raw:

    s = "".join(
        c.decode("utf-8")
        for c in row
    )

    times.append(
        pd.to_datetime(
            s,
            format="%Y-%m-%d_%H:%M:%S"
        )
    )

times = pd.DatetimeIndex(times)
times = times + pd.Timedelta(hours=UTC_OFFSET)

nt = len(times)

# =====================================================
# WRF VARIABLES
# =====================================================

T = ds.variables["T"][:,:,iy,ix] + 300.0

W = ds.variables["W"][:,:,iy,ix]

PH  = ds.variables["PH"][:,:,iy,ix]
PHB = ds.variables["PHB"][:,:,iy,ix]

PBLH = ds.variables["PBLH"][:,iy,ix]

# =====================================================
# HEIGHT
# =====================================================

z_stag = (PH + PHB)/9.81

z = 0.5*(z_stag[:,:-1] + z_stag[:,1:])

print("T shape =", T.shape)
print("W shape =", W.shape)
print("z shape =", z.shape)

# =====================================================
# TIBL ESTIMATION
# =====================================================

def estimate_tibl(theta, height):

    if len(theta) != len(height):
        n = min(len(theta), len(height))
        theta = theta[:n]
        height = height[:n]

    mask = np.isfinite(theta) & np.isfinite(height)

    theta = theta[mask]
    height = height[mask]

    if len(theta) < 5:
        return np.nan

    # low-level inversion only
    keep = height < 1500

    theta = theta[keep]
    height = height[keep]

    if len(theta) < 5:
        return np.nan

    dtheta = np.gradient(theta, height)

    idx = np.argmax(dtheta)

    return height[idx]

# =====================================================
# TIBL TIME SERIES
# =====================================================

tibl = np.zeros(nt)

for t in range(nt):

    tibl[t] = estimate_tibl(
        T[t],
        z[t]
    )

# =====================================================
# CURTAIN PLOT
# =====================================================

hours = np.arange(nt)

X, Y = np.meshgrid(
    hours,
    z[0]
)

fig, ax = plt.subplots(
    2,
    1,
    figsize=(14,10),
    sharex=True
)

# =====================================================
# THETA
# =====================================================

cf1 = ax[0].contourf(
    X,
    Y,
    T.T,
    levels=30,
    cmap="turbo"
)

ax[0].plot(
    hours,
    PBLH,
    "k",
    lw=2,
    label="WRF PBLH"
)

ax[0].plot(
    hours,
    tibl,
    "w--",
    lw=2,
    label="TIBL"
)

ax[0].set_ylabel("Height (m)")
ax[0].set_ylim(0,3000)

ax[0].set_title(
    "Potential Temperature Curtain"
)

ax[0].legend()

plt.colorbar(
    cf1,
    ax=ax[0],
    label="Theta (K)"
)

# =====================================================
# W
# =====================================================

cf2 = ax[1].contourf(
    X,
    Y,
    W[:,:34].T,
    levels=np.arange(-2,2.1,0.1),
    cmap="RdBu_r",
    extend="both"
)

ax[1].plot(
    hours,
    PBLH,
    "k",
    lw=2
)

ax[1].plot(
    hours,
    tibl,
    "w--",
    lw=2
)

ax[1].set_ylim(0,3000)

ax[1].set_ylabel("Height (m)")
ax[1].set_xlabel("Hour (AEST)")

ax[1].set_title(
    "Vertical Velocity Curtain"
)

ax[1].set_xticks(hours)
ax[1].set_xticklabels(
    [t.strftime("%H") for t in times],
    rotation=45
)

plt.colorbar(
    cf2,
    ax=ax[1],
    label="W (m/s)"
)

plt.tight_layout()

plt.savefig(
    "Lidcombe_TIBL_curtain.png",
    dpi=300
)

plt.show()

