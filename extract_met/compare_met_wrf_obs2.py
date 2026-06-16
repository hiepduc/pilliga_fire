#!/usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from netCDF4 import Dataset
from wrf import getvar, ll_to_xy, ALL_TIMES
import pandas as pd
import numpy as np

# ==========================================================
# INPUT FILES
# ==========================================================

obs_file = "/mnt/scratch_lustre/duch/lake_macquarie_ps/HiepMetDataAllSites2023.csv"

station_file = "air-quality-monitoring-sites-summary-9-feb-2026.csv"

wrf_files = [
    "/mnt/scratch_lustre/duch/runpillaga/jkong/wrfout_d02_2023-12-8to19_wrfnative.nc"
]

UTC_OFFSET = 10

# ==========================================================
# STATIONS TO VALIDATE
# ==========================================================

stations_to_use = [
    "LIVERPOOL",
    "LIDCOMBE",
    "PROSPECT",
    "BRINGELLY",
    "NEWCASTLE",
    "SINGLETON",
    "MUSWELLBROOK",
    "RICHMOND",
    "ST MARYS",
    "BARGO",
    "ROZELLE",
    "RANDWICK",
    "MERRIWA",
    "BERESFIELD",
    "MAYFIELD",
    "ALBION PARK SOUTH",
    "WALLSEND",
    "MORISSET",
    "WYONG",
    "CAMDEN",
    "CAMPBELLTOWN WEST",
    "CAMBERWELL",
    "EARLWOOD",
    "PENRITH",
    "OAKDALE",
    "WOLLONGONG"
]

# ==========================================================
# LOAD OBS DATA
# ==========================================================

obs = pd.read_csv(obs_file)

obs["datetime"] = pd.to_datetime(
    obs["Date_time"],
    dayfirst=True,
    errors="coerce"
)

obs = obs.set_index("datetime")

obs.columns = (
    obs.columns
    .str.upper()
    .str.strip()
)

print("\nObservation file loaded")
print("Rows:", len(obs))

# ==========================================================
# COLUMN MATCHING
# ==========================================================

def find_col(station, var):

    station = station.upper()

    for c in obs.columns:

        if station in c and var in c:
            return c

    return None

# ==========================================================
# STATION COORDS
# ==========================================================

site_df = pd.read_csv(
    station_file,
    encoding="latin1"
)

site_df["STATION"] = (
    site_df["NSW air quality monitoring (AQMN) site"]
    .astype(str)
    .str.upper()
    .str.strip()
)

coord_lookup = {}

for _, r in site_df.iterrows():

    try:

        coord_lookup[r["STATION"]] = (
            float(r["Latitude\n(South)"]),
            float(r["Longitude\n(East)"])
        )

    except:
        pass

print("Stations with coordinates:", len(coord_lookup))

# ==========================================================
# READ WRF ONCE
# ==========================================================

print("\nLoading WRF data...")

nc = Dataset(wrf_files[0])

t2 = getvar(nc, "T2", timeidx=ALL_TIMES)
q2 = getvar(nc, "Q2", timeidx=ALL_TIMES)
u10 = getvar(nc, "U10", timeidx=ALL_TIMES)
v10 = getvar(nc, "V10", timeidx=ALL_TIMES)
psfc = getvar(nc, "PSFC", timeidx=ALL_TIMES)

times = getvar(nc, "times", timeidx=ALL_TIMES)

wrf_times = pd.to_datetime(
    [str(t.values) for t in times]
) + pd.Timedelta(hours=UTC_OFFSET)

ny = t2.shape[1]
nx = t2.shape[2]

print("WRF dimensions:", ny, nx)
print("WRF timesteps :", len(wrf_times))

# ==========================================================
# EXTRACT WRF AT STATIONS
# ==========================================================

wrf_cache = {}

for station in stations_to_use:

    station = station.upper()

    if station not in coord_lookup:

        print(f"{station}: no coordinates")
        continue

    lat, lon = coord_lookup[station]

    try:

        xy = ll_to_xy(nc, lat, lon)

        ix = int(xy[0])
        iy = int(xy[1])

    except Exception as e:

        print(f"{station}: ll_to_xy failed")
        print(e)
        continue

    if (
        ix < 0 or ix >= nx or
        iy < 0 or iy >= ny
    ):

        print(
            f"{station}: outside WRF domain "
            f"(ix={ix}, iy={iy})"
        )
        continue

    print(
        f"Processing {station} "
        f"(ix={ix}, iy={iy})"
    )

    T2_all = []
    RH_all = []
    WS_all = []
    WD_all = []

    nt = len(wrf_times)

    for t in range(nt):

        temp_c = float(
            t2[t, iy, ix]
        ) - 273.15

        T2_all.append(temp_c)

        u = float(u10[t, iy, ix])
        v = float(v10[t, iy, ix])

        ws = np.sqrt(u*u + v*v)

        wd = (
            270 -
            np.degrees(
                np.arctan2(v, u)
            )
        ) % 360

        WS_all.append(ws)
        WD_all.append(wd)

        es = (
            6.112 *
            np.exp(
                (17.67 * temp_c) /
                (temp_c + 243.5)
            )
        )

        e = (
            float(q2[t, iy, ix]) *
            float(psfc[t, iy, ix]) /
            0.622
        ) / 100.0

        rh = 100.0 * e / es

        rh = max(0, min(100, rh))

        RH_all.append(rh)

    wrf_cache[station] = pd.DataFrame(
        {
            "T_WRF": T2_all,
            "RH_WRF": RH_all,
            "WS_WRF": WS_all,
            "WD_WRF": WD_all
        },
        index=wrf_times
    )

# ==========================================================
# STATS
# ==========================================================

def calc_stats(model, obs):

    mask = (
        np.isfinite(model) &
        np.isfinite(obs)
    )

    model = np.asarray(model)[mask]
    obs = np.asarray(obs)[mask]

    if len(model) < 10:
        return np.nan, np.nan, np.nan

    r = np.corrcoef(
        model,
        obs
    )[0, 1]

    mb = np.mean(model - obs)

    rmse = np.sqrt(
        np.mean(
            (model - obs) ** 2
        )
    )

    return r, mb, rmse

# ==========================================================
# VALIDATION
# ==========================================================

results = []

for station in wrf_cache:

    temp_col = find_col(station, "TEMP")
    rh_col   = find_col(station, "HUMID")
    ws_col   = find_col(station, "WSP")
    wd_col   = find_col(station, "WDR")

    if temp_col is None:

        print(
            f"{station}: "
            "temperature column not found"
        )

        continue

    obs_df = pd.DataFrame(
        index=obs.index
    )

    obs_df["T_OBS"] = pd.to_numeric(
        obs[temp_col],
        errors="coerce"
    )

    if rh_col:
        obs_df["RH_OBS"] = pd.to_numeric(
            obs[rh_col],
            errors="coerce"
        )

    if ws_col:
        obs_df["WS_OBS"] = pd.to_numeric(
            obs[ws_col],
            errors="coerce"
        )

    if wd_col:
        obs_df["WD_OBS"] = pd.to_numeric(
            obs[wd_col],
            errors="coerce"
        )

    compare = pd.merge(
        wrf_cache[station],
        obs_df,
        left_index=True,
        right_index=True,
        how="inner"
    )

    if len(compare) < 10:

        print(
            f"{station}: "
            f"only {len(compare)} matched records"
        )

        continue

    rT, mbT, rmseT = calc_stats(
        compare["T_WRF"],
        compare["T_OBS"]
    )

    rW, mbW, rmseW = calc_stats(
        compare["WS_WRF"],
        compare["WS_OBS"]
    )

    rH, mbH, rmseH = calc_stats(
        compare["RH_WRF"],
        compare["RH_OBS"]
    )

    results.append([
        station,
        len(compare),
        rT, mbT, rmseT,
        rW, mbW, rmseW,
        rH, mbH, rmseH
    ])

# ==========================================================
# OUTPUT
# ==========================================================

df = pd.DataFrame(
    results,
    columns=[
        "Station",
        "N",
        "R_T",
        "MB_T",
        "RMSE_T",
        "R_WS",
        "MB_WS",
        "RMSE_WS",
        "R_RH",
        "MB_RH",
        "RMSE_RH"
    ]
)

df = df.sort_values(
    "Station"
)

df.to_csv(
    "WRF_MET_validation.csv",
    index=False
)

print("\nValidation Summary")
print(df)

print(
    "\nSaved: "
    "WRF_MET_validation.csv"
)

# ==========================================================
# PANEL PLOT FOR ONE STATION
# ==========================================================

def plot_station(station):

    temp_col = find_col(station, "TEMP")
    rh_col   = find_col(station, "HUMID")
    ws_col   = find_col(station, "WSP")
    wd_col   = find_col(station, "WDR")

    if temp_col is None:
        print("No temperature column for", station)
        return

    obs_df = pd.DataFrame(index=obs.index)

    obs_df["T_OBS"] = pd.to_numeric(obs[temp_col], errors="coerce")

    if rh_col:
        obs_df["RH_OBS"] = pd.to_numeric(obs[rh_col], errors="coerce")

    if ws_col:
        obs_df["WS_OBS"] = pd.to_numeric(obs[ws_col], errors="coerce")

    if wd_col:
        obs_df["WD_OBS"] = pd.to_numeric(obs[wd_col], errors="coerce")

    compare = pd.merge(
        wrf_cache[station],
        obs_df,
        left_index=True,
        right_index=True,
        how="inner"
    )

    compare = compare.sort_index()

    if len(compare) == 0:
        print("No matched data for", station)
        return

    fig, ax = plt.subplots(
        2, 2,
        figsize=(16,10),
        sharex=True
    )

    # --------------------------------------------------
    # Temperature
    # --------------------------------------------------

    ax[0,0].plot(
        compare.index,
        compare["T_OBS"],
        label="Observed"
    )

    ax[0,0].plot(
        compare.index,
        compare["T_WRF"],
        label="WRF"
    )

    ax[0,0].set_title("Temperature (°C)")
    ax[0,0].legend()
    ax[0,0].grid(True)

    # --------------------------------------------------
    # Relative Humidity
    # --------------------------------------------------

    if "RH_OBS" in compare:

        ax[0,1].plot(
            compare.index,
            compare["RH_OBS"],
            label="Observed"
        )

        ax[0,1].plot(
            compare.index,
            compare["RH_WRF"],
            label="WRF"
        )

        ax[0,1].set_title("Relative Humidity (%)")
        ax[0,1].legend()
        ax[0,1].grid(True)

    # --------------------------------------------------
    # Wind Speed
    # --------------------------------------------------

    if "WS_OBS" in compare:

        ax[1,0].plot(
            compare.index,
            compare["WS_OBS"],
            label="Observed"
        )

        ax[1,0].plot(
            compare.index,
            compare["WS_WRF"],
            label="WRF"
        )

        ax[1,0].set_title("Wind Speed (m/s)")
        ax[1,0].legend()
        ax[1,0].grid(True)

    # --------------------------------------------------
    # Wind Direction
    # --------------------------------------------------

    if "WD_OBS" in compare:

        ax[1,1].plot(
            compare.index,
            compare["WD_OBS"],
            ".",
            markersize=2,
            label="Observed"
        )

        ax[1,1].plot(
            compare.index,
            compare["WD_WRF"],
            ".",
            markersize=2,
            label="WRF"
        )

        ax[1,1].set_ylim(0,360)
        ax[1,1].set_title("Wind Direction (deg)")
        ax[1,1].legend()
        ax[1,1].grid(True)

    plt.suptitle(station, fontsize=16)

    plt.tight_layout()

    outfile = f"panel_{station.replace(' ','_')}.png"

    plt.savefig(
        outfile,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()

    print("Saved:", outfile)

selected_stations = [
    "LIVERPOOL",
    "LIDCOMBE",
    "PROSPECT",
    "CAMDEN",
    "MUSWELLBROOK",
    "NEWCASTLE"
]

for station in selected_stations:

    if station in wrf_cache:

        plot_station(station)

