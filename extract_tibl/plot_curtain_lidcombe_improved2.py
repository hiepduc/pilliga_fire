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

#CEILO_FILE = (
#"/mnt/scratch_lustre/duch/runpillaga/extract_tibl/ceilodata/"
#"L3_DEFAULT__202312190000_1_360_1_3120_10_30_4000_3_0_1_500_1000_4000_60.nc"
#)

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

# =====================================================
# CEILOMETER
# =====================================================

ceilo_files = [

"/mnt/scratch_lustre/duch/runpillaga/extract_tibl/ceilodata/"
"L3_DEFAULT__202312190000_1_360_1_3120_10_30_4000_3_0_1_500_1000_4000_60.nc",

"/mnt/scratch_lustre/duch/runpillaga/extract_tibl/ceilodata/"
"L3_DEFAULT__202312200000_1_360_1_3120_10_30_4000_3_0_1_500_1000_4000_60.nc"

]

ctime_all = []
bs_all = []
bl1_all = []
bl2_all = []

for fname in ceilo_files:

    print("Reading", fname)

    c = Dataset(fname)

    t = pd.to_datetime(
        c.variables["time"][:],
        unit="s",
        origin="unix"
    )

    t = t + pd.Timedelta(hours=UTC_OFFSET)

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

    ctime_all.append(t)
    bs_all.append(bs)

    bl1_all.append(bl[:,0])
    bl2_all.append(bl[:,1])

    rng = c.variables["range"][:]

# ---------------------------------------------------
# concatenate
# ---------------------------------------------------

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

plt.xlim(
    pd.Timestamp("2023-12-19 12:00"),
    pd.Timestamp("2023-12-20 11:00")
)

# ----------------------------------------------------------
# Ceilo layers
# ----------------------------------------------------------

plt.plot(
    ctime,
    bl1,
    color="white",
    lw=2,
    label="Ceilo BL1"
)

plt.plot(
    ctime,
    bl2,
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

####################
# Check ceilometer data
#######################

bl1 = np.concatenate(bl1_all)
bl2 = np.concatenate(bl2_all)

print("\nBL1 statistics")
print("Total:", len(bl1))
print("Valid:", np.sum(np.isfinite(bl1)))
print("Min:", np.nanmin(bl1))
print("Max:", np.nanmax(bl1))

print("\nBL2 statistics")
print("Total:", len(bl2))
print("Valid:", np.sum(np.isfinite(bl2)))
print("Min:", np.nanmin(bl2))
print("Max:", np.nanmax(bl2))

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


mask = (
    (ctime >= pd.Timestamp("2023-12-20 00:00"))
    &
    (ctime <= pd.Timestamp("2023-12-20 11:00"))
)

plt.figure(figsize=(12,4))

plt.scatter(
    ctime[mask],
    bl1[mask],
    s=4
)

plt.ylim(0,2000)

plt.title("BL1 only")
plt.grid()

plt.show()

plt.show()

mask20 = ctime >= pd.Timestamp("2023-12-20 00:00")

print("\n20 Dec only")
print("BL1 valid:", np.sum(np.isfinite(bl1[mask20])))
print("BL2 valid:", np.sum(np.isfinite(bl2[mask20])))

print("\nCeilo time range")
print("Start =", ctime.min())
print("End   =", ctime.max())

print("\nWRF time range")
print("Start =", times.min())
print("End   =", times.max())


print("\nBL1 after midnight")

mask = (
    (ctime >= pd.Timestamp("2023-12-20 00:00"))
    &
    (ctime <= pd.Timestamp("2023-12-20 03:00"))
)

print(bl1[mask][:20])
print(bl2[mask][:20])

