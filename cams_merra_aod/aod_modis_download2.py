#!/usr/bin/env python3

import requests
import os

# ====================================
# TOKEN
# ====================================
TOKEN = "eyJ0eXAiOiJKV1QiLCJvcmlnaW4iOiJFYXJ0aGRhdGEgTG9naW4iLCJzaWciOiJlZGxqd3RwdWJrZXlfb3BzIiwiYWxnIjoiUlMyNTYifQ.eyJ0eXBlIjoiVXNlciIsInVpZCI6ImhkdWMiLCJleHAiOjE3ODE4NzU3NzgsImlhdCI6MTc3NjY5MTc3OCwiaXNzIjoiaHR0cHM6Ly91cnMuZWFydGhkYXRhLm5hc2EuZ292IiwiaWRlbnRpdHlfcHJvdmlkZXIiOiJlZGxfb3BzIiwiYWNyIjoiZWRsIiwiYXNzdXJhbmNlX2xldmVsIjozfQ.hjb68bCv_jJzO7kQjNF2caAzLLvY4znedX2HZH9_Gzd7NWgJtmbtR7YZ5xGCRkhoW5_Ip9hICLcp9zPLICQsAC3Cyuy6cDF0L-znpjtV4JRFKhvuaxTNu1Sol-KIAhk7ZeVAXLyHgWaQrN9dJLNxkAL992KMB5EbfXpJ-83sp7cpuIN8txwCPoplDWqsM7oJ5amX0X01A8YL8eNoVLYj42VgXAknmyJK41XSJMJ7Jlcq9aZ4Fx_0BXL1GUnnkh7vvFCsSJpZe1fhXApfBp3Fe1KYvpKUTI8Wz3yjJf5aMNvjwDUE5RGCw4ioa17UM9_BHKDpkzDs89bozBQSwVZTbA"

YEAR = 2023
JDAY = 353

OUTDIR = f"MOD04_L2/{YEAR}/{JDAY:03d}"

os.makedirs(
    OUTDIR,
    exist_ok=True
)

headers = {
    "Authorization": f"Bearer {TOKEN}"
}

base = (
"https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/"
f"61/MOD04_L2/{YEAR}/{JDAY:03d}/"
)

downloaded = 0

# MODIS granules every 5 minutes
for hh in range(24):

    for mm in range(0,60,5):

        t = f"{hh:02d}{mm:02d}"

        url = (
            base +
            f"MOD04_L2.A{YEAR}{JDAY:03d}.{t}.061.hdf"
        )

        outfile = (
            f"{OUTDIR}/"
            f"MOD04_L2.A{YEAR}{JDAY:03d}.{t}.061.hdf"
        )

        if os.path.exists(outfile):

            print("exists", t)
            continue

        try:

            print("download", t)

            r = requests.get(
                url,
                headers=headers,
                stream=True,
                timeout=120
            )

            if r.status_code == 200:

                with open(
                    outfile,
                    "wb"
                ) as f:

                    for chunk in r.iter_content(
                        1024*1024
                    ):
                        f.write(chunk)

                downloaded += 1

            else:

                print(
                    "missing",
                    t,
                    r.status_code
                )

        except Exception as e:

            print(
                "failed",
                t,
                e
            )

print()
print(
    "Downloaded",
    downloaded,
    "files"
)

