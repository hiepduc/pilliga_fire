#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from netCDF4 import Dataset

# ==========================================================
# SETTINGS
# ==========================================================

WRF_FILE = "mnt/scratch_lustre/duch/runpillaga/wrfout_d02_2023-12-19_00:00:00"

LAT_SITE = -33.89
LON_SITE = 151.05

UTC_OFFSET = 11      # AEDT in Dec 2023

PROFILE_HOUR = 15    # Local time for profile plot

# ==========================================================
# OPEN FILE
# ==========================================================

ds = Dataset(WRF_FILE)

# ==========================================================
# GRID
# ==========================================================

xlat = ds.variables["XLAT"][:]
xlon = ds.variables["XLONG"][:]

if xlat.ndim == 3:
    lat = xlat[0]
    lon = xlon[0]
else:
    lat = xlat
    lon = xlon

dist = np.sqrt(
    (lat - LAT_SITE)**2 +
    (lon - LON_SITE)**2
)

iy, ix = np.unravel_index(
    np.argmin(dist),
    dist.shape
)

print("Grid =", ix, iy)

# ==========================================================
# TIMES
# ==========================================================

times = []

for row in ds.variables["Times"][:]:

    tstr = "".join(
        c.decode() if isinstance(c, bytes) else str(c)
        for c in row
    )

    times.append(pd.to_datetime(
        tstr,
        format="%Y-%m-%d_%H:%M:%S"
    ))

times = pd.DatetimeIndex(times)

times = times + pd.Timedelta(hours=UTC_OFFSET)

nt = len(times)

# ==========================================================
# VARIABLES
# ==========================================================

T  = ds.variables["T"][:]
P  = ds.variables["P"][:]
PB = ds.variables["PB"][:]

U = ds.variables["U"][:]
V = ds.variables["V"][:]

W = ds.variables["W"][:]

PH  = ds.variables["PH"][:]
PHB = ds.variables["PHB"][:]

PBLH = ds.variables["PBLH"][:]

# ==========================================================
# HEIGHT
# ==========================================================

z_stag = (PH + PHB) / 9.81

z_mass = 0.5 * (
    z_stag[:, :-1, :, :] +
    z_stag[:, 1:, :, :]
)

# ==========================================================
# THETA
# ==========================================================

theta = T + 300.0

# ==========================================================
# EXTRACT COLUMN
# ==========================================================

theta_col = theta[:, :, iy, ix]

z_col = z_mass[:, :, iy, ix]

pblh_col = PBLH[:, iy, ix]

# U/V unstagger

u_col = 0.5 * (
    U[:, :, iy, ix] +
    U[:, :, iy, ix+1]
)

v_col = 0.5 * (
    V[:, :, iy, ix] +
    V[:, :, iy+1, ix]
)

wind_col = np.sqrt(
    u_col**2 +
    v_col**2
)

# W unstagger

w_col = 0.5 * (
    W[:, :-1, iy, ix] +
    W[:, 1:, iy, ix]
)

# ==========================================================
# COASTAL TIBL DETECTOR
# ==========================================================

def estimate_tibl(theta_prof,
                  z_prof,
                  threshold=0.75):

    theta0 = theta_prof[0]

    dtheta = theta_prof - theta0

    idx = np.where(
        dtheta > threshold
    )[0]

    if len(idx) == 0:
        return np.nan

    return z_prof[idx[0]]

# ==========================================================
# COMPUTE TIBL
# ==========================================================

tibl = []

for t in range(nt):

    tibl.append(
        estimate_tibl(
            theta_col[t],
            z_col[t]
        )
    )

tibl = np.array(tibl)

# ==========================================================
# PROFILE HOUR
# ==========================================================

profile_idx = np.argmin(
    np.abs(
        times.hour - PROFILE_HOUR
    )
)

print(
    "Profile time =",
    times[profile_idx]
)

# ==========================================================
# 4 PANEL FIGURE
# ==========================================================

fig, axs = plt.subplots(
    2,
    2,
    figsize=(14,10)
)

# ----------------------------------------------------------
# PANEL 1
# ----------------------------------------------------------

axs[0,0].plot(
    times,
    pblh_col,
    lw=2,
    label="WRF PBLH"
)

axs[0,0].plot(
    times,
    tibl,
    lw=2,
    label="TIBL"
)

axs[0,0].set_ylabel("Height (m)")
axs[0,0].set_title("PBLH vs TIBL")
axs[0,0].grid()
axs[0,0].legend()

# ----------------------------------------------------------
# PANEL 2
# ----------------------------------------------------------

axs[0,1].plot(
    theta_col[profile_idx],
    z_col[profile_idx],
    lw=2
)

axs[0,1].set_xlabel("Theta (K)")
axs[0,1].set_ylabel("Height (m)")
axs[0,1].set_title(
    f"Theta Profile\n{times[profile_idx]}"
)
axs[0,1].grid()

# ----------------------------------------------------------
# PANEL 3
# ----------------------------------------------------------

axs[1,0].plot(
    wind_col[profile_idx],
    z_col[profile_idx],
    lw=2
)

axs[1,0].set_xlabel("Wind speed (m/s)")
axs[1,0].set_ylabel("Height (m)")
axs[1,0].set_title(
    f"Wind Profile\n{times[profile_idx]}"
)
axs[1,0].grid()

# ----------------------------------------------------------
# PANEL 4
# ----------------------------------------------------------

axs[1,1].plot(
    w_col[profile_idx],
    z_col[profile_idx],
    lw=2
)

axs[1,1].axvline(
    0,
    color="k"
)

axs[1,1].set_xlabel("W (m/s)")
axs[1,1].set_ylabel("Height (m)")
axs[1,1].set_title(
    f"Vertical Velocity\n{times[profile_idx]}"
)
axs[1,1].grid()

plt.tight_layout()

plt.savefig(
    "TIBL_4panel_WRF.png",
    dpi=300
)

plt.show()

ds.close()

