"""Fetch the frozen Phase A inputs without opening the holdout archives."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from urllib.request import urlopen

RAW = Path("data/raw")
BTS = "https://www.transtats.bts.gov/PREZIP/"
DOWNLOADS = {
    **{
        f"On_Time_Reporting_Carrier_On_Time_Performance_1987_present_2025_{month}.zip":
        BTS + f"On_Time_Reporting_Carrier_On_Time_Performance_1987_present_2025_{month}.zip"
        for month in range(1, 13)
    },
    "airports.csv": "https://raw.githubusercontent.com/davidmegginson/ourairports-data/d1cf45405deea3f617af99914b9d05e3175910dc/airports.csv",
}


def fetch(item):
    name, url = item
    target = RAW / name
    receipt = target.with_suffix(target.suffix + ".json")
    if not (target.exists() and receipt.exists() and json.loads(receipt.read_text())["url"] == url):
        partial = target.with_suffix(target.suffix + ".part")
        try:
            with urlopen(url, timeout=90) as response, partial.open("wb") as out:
                shutil.copyfileobj(response, out)
            with partial.open("rb") as source:
                sha256 = hashlib.file_digest(source, "sha256").hexdigest()
            partial.replace(target)
            receipt.write_text(json.dumps({
                "url": url, "sha256": sha256,
                "downloaded_utc": datetime.now(timezone.utc).isoformat(),
            }, indent=2) + "\n")
        finally:
            partial.unlink(missing_ok=True)
    print(f"Ready: {name}", flush=True)


if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, DOWNLOADS.items()))
