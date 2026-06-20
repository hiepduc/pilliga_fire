#!/usr/bin/env python3

from netCDF4 import Dataset

f = Dataset("../wrfout_d02_2023-12-08_00:00:00")

for v in ["XLAT", "XLONG"]:
    print(v)
    print("shape =", f.variables[v].shape)
    print("dims  =", f.variables[v].dimensions)
    print()

