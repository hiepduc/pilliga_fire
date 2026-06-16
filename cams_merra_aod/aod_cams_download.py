import cdsapi

c = cdsapi.Client()

dataset = "cams-global-atmospheric-composition-forecasts"

request = {
    "date": "2023-12-17/2023-12-20",
    "type": ["forecast"],
    "variable": [
        "total_aerosol_optical_depth_550nm"
    ],
    "time": [
        "00:00","12:00"
    ],
    "leadtime_hour": [
        "0","3","6","9","12","15","18","21",
        "24","27","30","33","36","39","42","45",
        "48","51","54","57","60","63","66","69",
        "72"
    ],
    "area": [-28,141,-37,154],   # N,W,S,E (NSW)
    "data_format": "netcdf"
}

c.retrieve(
    dataset,
    request,
    "cams_aod550_nsw_20231217_20231220.nc"
)
