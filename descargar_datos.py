import csv
import io
import time
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://www.datos.gov.co/resource/{id}.csv"
RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

MINDEFENSA = {
    "homicidio": "m8fd-ahd9",
    "hurto_personas": "4rxi-8m8d",
    "lesiones_personales": "jr6v-i33g",
    "violencia_intrafamiliar": "gepp-dxcs",
    "hurto_residencias": "7mn7-vzqp",
    "hurto_comercio": "7i2x-h5vp",
    "delitos_informaticos": "4v6r-wu98",
    "hurto_vehiculos": "csb4-y6v2",
    "delitos_sexuales": "bz43-8ahq",
    "homicidio_transito": "uav5-b85g",
    "extorsion": "q2ib-t9am",
    "delitos_ambientales": "9zck-qfvc",
    "abigeato": "p88b-5ac7",
}
MEDLEGAL = {"medlegal_homicidios_victimas": ("vtub-3de2", "codigo_dane_municipio like '23001%'")}


def fetch(ds_id: str, where: str) -> list[dict]:
    rows, offset, page = [], 0, 50000
    while True:
        q = urllib.parse.urlencode({"$where": where, "$limit": page, "$offset": offset, "$order": ":id"})
        with urllib.request.urlopen(BASE.format(id=ds_id) + "?" + q, timeout=120) as r:
            chunk = list(csv.DictReader(io.StringIO(r.read().decode("utf-8"))))
        rows += chunk
        if len(chunk) < page:
            return rows
        offset += page
        time.sleep(0.5)


def save(name: str, rows: list[dict]) -> None:
    path = RAW / f"{name}.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"{name}: {len(rows)} filas -> {path.name}")


if __name__ == "__main__":
    for name, ds in MINDEFENSA.items():
        save(name, fetch(ds, "cod_muni='23001'"))
    for name, (ds, where) in MEDLEGAL.items():
        save(name, fetch(ds, where))
