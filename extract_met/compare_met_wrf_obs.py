#!/usr/bin/env python3

from netCDF4 import Dataset
from wrf import getvar, ll_to_xy, ALL_TIMES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

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
# LOAD OBS MET DATA
# ==========================================================

obs = pd.read_csv(obs_file)

obs["datetime"] = pd.to_datetime(obs["Date_time"], dayfirst=True, errors="coerce")
obs = obs.set_index("datetime")

print(obs.columns[:20])

for c in obs.columns:
    if "LIVERPOOL" in c.upper():
        print(c)

# clean column names (VERY IMPORTANT)
obs.columns = obs.columns.str.strip().str.upper()

# ==========================================================
# FUNCTION: find matching column
# ==========================================================

def find_col(station, var):
    """
    station: 'LIVERPOOL'
    var: 'TEMP', 'HUMID', 'WSP', 'WDR'
    """
    for c in obs.columns:
        if station in c and var in c:
            return c
    return None

# ==========================================================
# LOAD STATION COORDS
# ==========================================================

site_df = pd.read_csv(station_file, encoding="latin1")

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

print("Stations with coords:", len(coord_lookup))

# ==========================================================
# WRF CACHE
# ==========================================================

wrf_cache = {}

# ==========================================================
# LOOP STATIONS
# ==========================================================

stations = [c.replace(" TEMP 1H", "").replace(" HUMID 1H", "")
            .replace(" WSP 1H", "").replace(" WDR 1H", "")
            .strip()
            for c in obs.columns if "TEMP" in c]

stations = sorted(set(stations))

for station in stations:

    station_clean = station.upper()

    if station_clean not in coord_lookup:
        continue

    lat, lon = coord_lookup[station_clean]

    print("Processing:", station_clean)

    wrf_times = []
    T2_all = []
    RH_all = []
    WS_all = []
    WD_all = []

    for f in wrf_files:

        nc = Dataset(f)

        xy = ll_to_xy(nc, lat, lon)
        ix, iy = int(xy[0]), int(xy[1])

        # WRF variables
        t2 = getvar(nc, "T2", timeidx=ALL_TIMES)
        q2 = getvar(nc, "Q2", timeidx=ALL_TIMES)
        u10 = getvar(nc, "U10", timeidx=ALL_TIMES)
        v10 = getvar(nc, "V10", timeidx=ALL_TIMES)
        times = getvar(nc, "times", timeidx=ALL_TIMES)

        # pressure for RH conversion
        psfc = getvar(nc, "PSFC", timeidx=ALL_TIMES)

        nt = len(times)

        for t in range(nt):

            # time
            wrf_times.append(pd.to_datetime(str(times[t].values)) + pd.Timedelta(hours=UTC_OFFSET))

            # temperature (K → C)
            T2_all.append(float(t2[t, iy, ix] - 273.15))

            # wind speed
            ws = np.sqrt(float(u10[t, iy, ix])**2 + float(v10[t, iy, ix])**2)
            WS_all.append(ws)

            # wind direction (meteorological)
            wd = (270 - np.degrees(np.arctan2(float(v10[t, iy, ix]), float(u10[t, iy, ix])))) % 360
            WD_all.append(wd)

            # relative humidity (approx from q2 + T2 + PSFC)
            es = 6.112 * np.exp((17.67*(float(t2[t, iy, ix]-273.15)))/(float(t2[t, iy, ix]-29.65)))
            e = float(q2[t, iy, ix]) * float(psfc[t, iy, ix]) / 0.622
            rh = 100 * e / es
            RH_all.append(rh)

    wrf_df = pd.DataFrame({
        "T_WRF": T2_all,
        "RH_WRF": RH_all,
        "WS_WRF": WS_all,
        "WD_WRF": WD_all
    }, index=wrf_times)

    wrf_cache[station_clean] = wrf_df

# ==========================================================
# COMPARE
# ==========================================================

results = []

for station in wrf_cache.keys():

    temp_col = find_col(station, "TEMP")
    rh_col   = find_col(station, "HUMID")
    ws_col   = find_col(station, "WSP")
    wd_col   = find_col(station, "WDR")

    if temp_col is None:
        continue

    obs_df = pd.DataFrame(index=obs.index)

    obs_df["T_OBS"]  = pd.to_numeric(obs[temp_col], errors="coerce")
    obs_df["RH_OBS"] = pd.to_numeric(obs[rh_col], errors="coerce") if rh_col else np.nan
    obs_df["WS_OBS"] = pd.to_numeric(obs[ws_col], errors="coerce") if ws_col else np.nan
    obs_df["WD_OBS"] = pd.to_numeric(obs[wd_col], errors="coerce") if wd_col else np.nan

    compare = pd.merge(wrf_cache[station], obs_df, left_index=True, right_index=True, how="inner")
    compare = compare.dropna()

    if len(compare) < 10:
        continue

    def stats(a, b):
        r = np.corrcoef(a, b)[0, 1]
        mb = np.mean(a - b)
        rmse = np.sqrt(np.mean((a - b)**2))
        return r, mb, rmse

    rT, mbT, rmseT = stats(compare["T_WRF"], compare["T_OBS"])
    rW, mbW, rmseW = stats(compare["WS_WRF"], compare["WS_OBS"]) if "WS_OBS" in compare else (np.nan,np.nan,np.nan)
    rH, mbH, rmseH = stats(compare["RH_WRF"], compare["RH_OBS"]) if "RH_OBS" in compare else (np.nan,np.nan,np.nan)

    results.append([
        station,
        len(compare),
        rT, mbT, rmseT,
        rW, mbW, rmseW,
        rH, mbH, rmseH
    ])

# ==========================================================
# SAVE OUTPUT
# ==========================================================

df = pd.DataFrame(results, columns=[
    "Station","N",
    "R_T","MB_T","RMSE_T",
    "R_WS","MB_WS","RMSE_WS",
    "R_RH","MB_RH","RMSE_RH"
])

df.to_csv("WRF_MET_validation.csv", index=False)

print(df)

