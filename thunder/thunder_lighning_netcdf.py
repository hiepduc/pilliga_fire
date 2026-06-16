#!/usr/bin/env python3

import xarray as xr

# ==========================================================
# INPUT
# ==========================================================

infile = "WWLLN_th_2023.nc"
outfile = "WWLLN_th_2023_standard.nc"

# ==========================================================
# OPEN FILE
# ==========================================================

ds = xr.open_dataset(infile)

print(ds)

# ==========================================================
# TRANSPOSE VARIABLE
# ==========================================================

th = ds["thunder_hours"].transpose("nlat", "nlon", "nmon")

# ==========================================================
# CREATE NEW DATASET
# ==========================================================

newds = xr.Dataset(
    {
        "thunder_hours": (
            ("lat", "lon", "mon"),
            th.values
        )
    },
    coords={
        "lat": ds["lat"].values,
        "lon": ds["lon"].values,
        "mon": ds["mon"].values
    }
)

# ==========================================================
# ADD ATTRIBUTES
# ==========================================================

newds["thunder_hours"].attrs["long_name"] = "Thunder hours per month"
newds["thunder_hours"].attrs["units"] = "Hours"

# ==========================================================
# SAVE
# ==========================================================

newds.to_netcdf(outfile)

print(f"Saved: {outfile}")

