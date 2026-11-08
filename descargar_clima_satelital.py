from __future__ import annotations

import io
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
LAT, LON = 8.75, -75.88
START, END = "2014-04-01", "2025-12-31"
UA = {"User-Agent": "DelitosMonteria-research/1.0"}


def get(url: str, timeout: int = 180) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read()


def nasa_power() -> pd.DataFrame:
    q = urllib.parse.urlencode({
        "parameters": "T2M_MAX,T2M,T2M_MIN,RH2M,QV2M,PS,PRECTOTCORR", "community": "RE",
        "longitude": LON, "latitude": LAT, "start": START.replace("-", ""), "end": END.replace("-", ""),
        "format": "CSV"})
    txt = get(f"https://power.larc.nasa.gov/api/temporal/daily/point?{q}").decode()
    lines = txt.splitlines()
    k = next(i for i, l in enumerate(lines) if l.startswith("YEAR,"))
    d = pd.read_csv(io.StringIO("\n".join(lines[k:])))
    d["fecha"] = pd.to_datetime(dict(year=d.YEAR, month=d.MO, day=d.DY))
    d = d.drop(columns=["YEAR", "MO", "DY"]).replace(-999.0, float("nan"))
    return d.set_index("fecha")


def chirps() -> pd.DataFrame:
    geom = json.dumps({"type": "Polygon", "coordinates": [[[-75.93, 8.70], [-75.83, 8.70], [-75.83, 8.80],
                                                           [-75.93, 8.80], [-75.93, 8.70]]]})
    s, e = pd.Timestamp(START), pd.Timestamp(END)
    q = urllib.parse.urlencode({"datatype": 0, "begintime": s.strftime("%m/%d/%Y"), "endtime": e.strftime("%m/%d/%Y"),
                                "intervaltype": 0, "operationtype": 5, "geometry": geom})
    base = "https://climateserv.servirglobal.net/api"
    rid = json.loads(get(f"{base}/submitDataRequest/?{q}"))[0]
    for _ in range(120):
        prog = json.loads(get(f"{base}/getDataRequestProgress/?id={rid}"))[0]
        if prog >= 100:
            break
        time.sleep(5)
    data = json.loads(get(f"{base}/getDataFromRequest/?id={rid}"))["data"]
    d = pd.DataFrame({"fecha": pd.to_datetime([r["date"] for r in data], format="%m/%d/%Y"),
                      "precip_chirps_mm": [r["value"]["avg"] for r in data]})
    d.loc[d.precip_chirps_mm < 0, "precip_chirps_mm"] = float("nan")  # -9999 = sin dato
    return d.set_index("fecha").sort_index()


def main() -> None:
    n = nasa_power()
    n.to_csv(RAW / "clima_nasa_power_monteria.csv")
    print("NASA POWER:", n.shape, n.index.min().date(), n.index.max().date(), "nulos:", int(n.isna().sum().sum()))
    c = chirps()
    c.to_csv(RAW / "clima_chirps_monteria.csv")
    print("CHIRPS:", c.shape, c.index.min().date(), c.index.max().date(), "nulos:", int(c.isna().sum().sum()))


if __name__ == "__main__":
    main()
