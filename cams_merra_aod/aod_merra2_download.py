#!/usr/bin/env python3

import xarray as xr

url = (
"https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/"
"MERRA2/M2T1NXAER.5.12.4/"
"2023/12/"
"MERRA2_400.tavg1_2d_aer_Nx.20231217.nc4"
)

ds = xr.open_dataset(
    url,
    engine="netcdf4"
)

print(ds)

aod = ds["TOTEXTTAU"]

aod.to_netcdf(
    "merra2_aod_20231217.nc"
)
