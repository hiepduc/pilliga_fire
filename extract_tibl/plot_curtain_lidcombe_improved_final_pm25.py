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

MAX_HEIGHT = 4000

WRF_FILE = (
"/mnt/scratch_lustre/duch/runpillaga/"
"wrfout_d02_2023-12-19_01:00:00"
)

PM25_FILE = (
"/mnt/scratch_lustre/duch/runpillaga/"
"extract_pm25/airqualpm25.csv"
)

CEILO_FILES = [

"/mnt/scratch_lustre/duch/runpillaga/extract_tibl/ceilodata/"
"L3_DEFAULT__202312180000_1_360_1_3120_10_30_4000_3_0_1_500_1000_4000_60.nc",

"/mnt/scratch_lustre/duch/runpillaga/extract_tibl/ceilodata/"
"L3_DEFAULT__202312190000_1_360_1_3120_10_30_4000_3_0_1_500_1000_4000_60.nc",

"/mnt/scratch_lustre/duch/runpillaga/extract_tibl/ceilodata/"
"L3_DEFAULT__202312200000_1_360_1_3120_10_30_4000_3_0_1_500_1000_4000_60.nc"

]

# ==========================================================
# OPEN WRF
# ==========================================================

f = Dataset(WRF_FILE)

lat = f.variables["XLAT"][0,:,:]
lon = f.variables["XLONG"][0,:,:]

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
# HEIGHTS
# ==========================================================

g = 9.81

PH  = f.variables["PH"][:,:,iy,ix]
PHB = f.variables["PHB"][:,:,iy,ix]

z_stag = (PH + PHB) / g

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

PBLH = np.array(
    f.variables["PBLH"][:,iy,ix],
    dtype=float
)

# ==========================================================
# WRF PM2.5
# ==========================================================

PM25_WRF = np.array(
    f.variables["PM2_5_DRY"][:,0,iy,ix],
    dtype=float
)

PM25_WRF = pd.Series(
    PM25_WRF
).rolling(
    3,
    center=True,
    min_periods=1
).mean()

# ==========================================================
# TIBL
# ==========================================================

TIBL = np.full(
    len(times),
    np.nan
)

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

    grad = np.gradient(
        th,
        zz
    )

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

    ct = ct + pd.Timedelta(
        hours=UTC_OFFSET
    )

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
# CONCATENATE CEILOMETER
# ==========================================================

ctime = pd.DatetimeIndex(
    np.concatenate(ctime_all)
)

bs = np.vstack(bs_all)

bl1 = np.concatenate(bl1_all)

bl2 = np.concatenate(bl2_all)

bs_log = np.log10(bs)

# ==========================================================
# OBSERVED PM2.5
# ==========================================================

obs = pd.read_csv(PM25_FILE)

obs_time = pd.to_datetime(
    obs["Date"],
    dayfirst=True,
    errors="coerce"
)

obs_pm25 = pd.to_numeric(
    obs["LIDCOMBE PM2.5"],
    errors="coerce"
)

obs_pm25[obs_pm25 < 0] = np.nan

obs_df = pd.DataFrame({
    "time": obs_time,
    "pm25": obs_pm25
})

mask = (
    (obs_df.time >= "2023-12-19") &
    (obs_df.time <  "2023-12-22")
)

obs_df = obs_df.loc[mask]

# ==========================================================
# PLOT
# ==========================================================

fig, ax = plt.subplots(
    figsize=(20,9)
)

ax2 = ax.twinx()
ax2.spines["right"].set_position(("outward", 60))

# ==========================================================
# BACKSCATTER
# ==========================================================

pcm = ax.pcolormesh(
    ctime,
    rng,
    bs_log.T,
    shading="auto",
    vmin=1,
    vmax=5
)

# ==========================================================
# BL1 / BL2
# ==========================================================

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

# ==========================================================
# WRF PBLH / TIBL
# ==========================================================

ax.plot(
    times,
    PBLH,
    color="red",
    lw=3,
    label="WRF PBLH"
)

ax.plot(
    times,
    TIBL,
    color="yellow",
    lw=3,
    label="WRF TIBL"
)

# ==========================================================
# PM2.5
# ==========================================================

ax2.plot(
    obs_df.time,
    obs_df.pm25,
    color="magenta",
    lw=3,
    label="Observed PM2.5"
)

ax2.plot(
    times,
    PM25_WRF,
    color="lime",
    lw=3,
    label="WRF PM2.5"
)

ax2.set_ylabel("PM2.5 (µg m$^{-3}$)", color="magenta", fontsize=12)

ax2.tick_params(axis="y", colors="magenta")
ax2.spines["right"].set_color("magenta")

# ==========================================================
# FORMAT
# ==========================================================

ax.set_ylim(
    0,
    MAX_HEIGHT
)

start = pd.Timestamp("2023-12-18 00:00")
end   = pd.Timestamp("2023-12-21 23:59")

ax.set_xlim(start, end)
ax2.set_xlim(start, end)


ax.set_ylabel(
    "Height (m AGL)"
)

ax2.set_ylabel(
    "PM2.5 (µg m$^{-3}$)"
)

ax2.set_ylim(
    0,
    400
)

ax.set_title(
    "Lidcombe CL51 Backscatter + BL1/BL2 + "
    "WRF PBLH/TIBL + PM2.5"
)

ax.grid(True)

from mpl_toolkits.axes_grid1 import make_axes_locatable

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2.5%", pad=0.10)

from mpl_toolkits.axes_grid1 import make_axes_locatable

divider = make_axes_locatable(ax)

# reserve space on RIGHT for BOTH: colorbar + PM2.5 axis
cax = divider.append_axes("right", size="2.5%", pad=0.15)

cbar = fig.colorbar(pcm, cax=cax)
cbar.set_label("log10(backscatter)")

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc="upper right",
    fontsize=10
)

plt.tight_layout()

plt.savefig(
    "Lidcombe_Backscatter_PM25_WRF_Ceilo.png",
    dpi=300
)

plt.show()

