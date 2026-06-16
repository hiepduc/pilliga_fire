#!/usr/bin/env python3

import os
import re
import requests

TOKEN = "eyJ0eXAiOiJKV1QiLCJvcmlnaW4iOiJFYXJ0aGRhdGEgTG9naW4iLCJzaWciOiJlZGxqd3RwdWJrZXlfb3BzIiwiYWxnIjoiUlMyNTYifQ.eyJ0eXBlIjoiVXNlciIsInVpZCI6ImhkdWMiLCJleHAiOjE3ODE4NzU3NzgsImlhdCI6MTc3NjY5MTc3OCwiaXNzIjoiaHR0cHM6Ly91cnMuZWFydGhkYXRhLm5hc2EuZ292IiwiaWRlbnRpdHlfcHJvdmlkZXIiOiJlZGxfb3BzIiwiYWNyIjoiZWRsIiwiYXNzdXJhbmNlX2xldmVsIjozfQ.hjb68bCv_jJzO7kQjNF2caAzLLvY4znedX2HZH9_Gzd7NWgJtmbtR7YZ5xGCRkhoW5_Ip9hICLcp9zPLICQsAC3Cyuy6cDF0L-znpjtV4JRFKhvuaxTNu1Sol-KIAhk7ZeVAXLyHgWaQrN9dJLNxkAL992KMB5EbfXpJ-83sp7cpuIN8txwCPoplDWqsM7oJ5amX0X01A8YL8eNoVLYj42VgXAknmyJK41XSJMJ7Jlcq9aZ4Fx_0BXL1GUnnkh7vvFCsSJpZe1fhXApfBp3Fe1KYvpKUTI8Wz3yjJf5aMNvjwDUE5RGCw4ioa17UM9_BHKDpkzDs89bozBQSwVZTbA"

YEAR=2023
JDAY=351

OUTDIR=f"MOD04_L2/{YEAR}/{JDAY:03d}"

os.makedirs(
    OUTDIR,
    exist_ok=True
)

BASE=(
"https://ladsweb.modaps.eosdis.nasa.gov/"
f"archive/allData/61/MOD04_L2/{YEAR}/{JDAY:03d}/"
)

headers={
    "Authorization":f"Bearer {TOKEN}"
}

print("Reading directory...")

r=requests.get(
    BASE,
    headers=headers
)

print("HTTP",r.status_code)

if r.status_code!=200:
    raise Exception("Cannot access directory")

# extract filenames
files=sorted(
    set(
        re.findall(
            r'MOD04_L2\.A\d+\.\d+\.061\.\d+\.hdf',
            r.text
        )
    )
)

print("Found",len(files),"granules")

count=0

for fn in files:

    outfile=f"{OUTDIR}/{fn}"

    if os.path.exists(outfile):

        print("exists",fn)
        continue

    url=BASE+fn

    print("download",fn)

    rr=requests.get(
        url,
        headers=headers,
        stream=True
    )

    if rr.status_code==200:

        with open(
            outfile,
            "wb"
        ) as f:

            for chunk in rr.iter_content(
                1024*1024
            ):
                f.write(chunk)

        count+=1

print()
print("Downloaded",count)

