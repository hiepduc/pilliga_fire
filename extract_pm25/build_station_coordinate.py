#!/usr/bin/env python3

import pandas as pd
import numpy as np

# =====================================================
# INPUTS
# =====================================================

site_file = "air-quality-monitoring-sites-summary-9-feb-2026.csv"

# =====================================================
# READ NSW SITE DATABASE
# =====================================================

sites = pd.read_csv(
    site_file,
    encoding="latin1"
)

print(sites.columns.tolist())

# =====================================================
# BUILD LOOKUP TABLE
# =====================================================

station_lookup = {}

for _, row in sites.iterrows():

    try:

        name = str(
            row["NSW air quality monitoring (AQMN) site"]
        ).upper().strip()

        lat = float(
            row["Latitude\n(South)"]
        )

        lon = float(
            row["Longitude\n(East)"]
        )

        station_lookup[name] = {
            "lat": lat,
            "lon": lon
        }

    except Exception:
        pass

# =====================================================
# SHOW STATIONS
# =====================================================

for k in sorted(station_lookup.keys()):

    print(
        k,
        station_lookup[k]["lat"],
        station_lookup[k]["lon"]
    )

print()
print("Stations found =", len(station_lookup))

