import csv
import io
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

EST = "0013035501"
RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
QUERIES = {  # (dataset, sensor, select). Solo el sensor original: desde 2024 hay un sensor GPRS
    # duplicado (0071 / 0257) que, si se suma, duplica la precipitación.
    "temp": ("sbwg-7ju4", "0068", "date_trunc_ymd(fechaobservacion) as fecha, avg(valorobservado) as t_media, "
                                  "max(valorobservado) as t_max, min(valorobservado) as t_min, count(*) as n_temp"),
    "prec": ("s54a-sgyg", "0240", "date_trunc_ymd(fechaobservacion) as fecha, sum(valorobservado) as precip_mm, "
                                  "count(*) as n_prec"),
    # humedad relativa horaria (sensor original 0027; el 0028 GPRS es un duplicado desde 2023-09)
    "hum": ("uext-mhny", "0027", "date_trunc_ymd(fechaobservacion) as fecha, avg(valorobservado) as rh_media, "
                                "min(valorobservado) as rh_min, count(*) as n_rh"),
}
# Nota: se revisaron las estaciones automáticas cercanas (MONTERIA-AUT 13067020, INCODER-AUT 13055040,
# C.E. LA VICTORIA 1303500339, TREMENTINO-AUT 25020430) para rellenar la precipitación 2018-2022:
# ninguna tiene registros válidos en ese periodo, y TURIPANÁ (13075030) es convencional y no está publicada.


def fetch_year(ds, sensor, select, year):
    where = (f"codigoestacion='{EST}' AND codigosensor='{sensor}' "
             f"AND fechaobservacion >= '{year}-01-01T00:00:00' "
             f"AND fechaobservacion < '{year + 1}-01-01T00:00:00'")
    q = urllib.parse.urlencode({"$select": select, "$where": where,
                                "$group": "fecha", "$order": "fecha", "$limit": 1000})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(f"https://www.datos.gov.co/resource/{ds}.csv?{q}", timeout=600) as r:
                return list(csv.DictReader(io.StringIO(r.read().decode("utf-8"))))
        except Exception as e:  # reintentos ante timeouts del servidor
            print(f"  {ds} {year} intento {attempt + 1}: {e}", file=sys.stderr)
            time.sleep(10)
    raise RuntimeError(f"No se pudo descargar {ds} {year}")


if __name__ == "__main__":
    out = {}
    for name, (ds, sensor, select) in QUERIES.items():
        rows = []
        for y in range(2015, 2027):
            chunk = fetch_year(ds, sensor, select, y)
            print(f"{name} {y}: {len(chunk)} días")
            rows += chunk
        df = pd.DataFrame(rows)
        df["fecha"] = pd.to_datetime(df["fecha"]).dt.normalize()
        out[name] = df.set_index("fecha").apply(pd.to_numeric)
    clima = pd.concat(out.values(), axis=1, join="outer").sort_index()
    clima.to_csv(RAW / "clima_ideam_garzones_diario.csv")
    print(clima.describe().T)
