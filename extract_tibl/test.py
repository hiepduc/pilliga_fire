from netCDF4 import Dataset

f = "/mnt/scratch_lustre/duch/runpillaga/wrfout_d02_2023-12-08_00:00:00"

ds = Dataset(f)

print(ds.variables.keys())
