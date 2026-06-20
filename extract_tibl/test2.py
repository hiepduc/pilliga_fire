python3 - << EOF
from netCDF4 import Dataset

f = Dataset("../wrfout_d02_2023-12-08_00:00:00")

for v in ["U","V","PH","PHB","T","P","PB","PBLH"]:
    print(v, f.variables[v].shape)
EOF
