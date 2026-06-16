#!/usr/bin/env python3

import xarray as xr
import numpy as np
import pandas as pd

infile = "cams_aod550_nsw_20231217_20231220.nc"
outfile = "cams_aod550_nsw_standard.nc"

# --------------------------------------------------
# Read
# --------------------------------------------------
ds = xr.open_dataset(infile)

aod = ds["aod550"].values
valid = ds["valid_time"].values

# Dimensions
nlead = aod.shape[0]
nref  = aod.shape[1]
nlat  = aod.shape[2]
nlon  = aod.shape[3]

# --------------------------------------------------
# Convert from:
# (forecast_period, forecast_reference_time, lat, lon)
#
# to:
# (time, lat, lon)
# --------------------------------------------------

aod_flat = aod.reshape(
    nlead * nref,
    nlat,
    nlon
)

time_flat = valid.reshape(
    nlead * nref
)

# Convert timestamps
time_flat = pd.to_datetime(time_flat)

# Remove duplicate valid times
time_unique, idx = np.unique(
    time_flat,
    return_index=True
)

aod_flat = aod_flat[idx]

# Sort chronologically
order = np.argsort(time_unique)

time_unique = time_unique[order]
aod_flat = aod_flat[order]

# --------------------------------------------------
# Output
# --------------------------------------------------

out = xr.Dataset(
    {
        "aod550": (
            ["time", "latitude", "longitude"],
            aod_flat
        )
    },
    coords={
        "time": time_unique,
        "latitude": ds.latitude.values,
        "longitude": ds.longitude.values
    }
)

out["aod550"].attrs = ds["aod550"].attrs

out.to_netcdf(outfile)

print(out)
print(f"\nSaved: {outfile}")

