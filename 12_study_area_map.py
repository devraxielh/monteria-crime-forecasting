from __future__ import annotations

import importlib
import json
import sys
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
U = importlib.import_module("00_config_and_utils")

GEO = U.DATA_RAW / "geo"
SRC = {"ADM1": "https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/COL/ADM1/"
               "geoBoundaries-COL-ADM1_simplified.geojson",
       "ADM2": "https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/COL/ADM2/"
               "geoBoundaries-COL-ADM2_simplified.geojson"}
STATION = (-75.825, 8.826)
NASA = (-75.88, 8.75)
CHIRPS_BOX = (-75.93, 8.70, 0.10, 0.10)  # lon0, lat0, ancho, alto


def load(level: str) -> dict:
    GEO.mkdir(parents=True, exist_ok=True)
    f = GEO / f"geoBoundaries-COL-{level}_simplified.geojson"
    if not f.exists():
        req = urllib.request.Request(SRC[level], headers={"User-Agent": "DelitosMonteria-research/1.0"})
        with urllib.request.urlopen(req, timeout=180) as r:
            f.write_bytes(r.read())
    return json.loads(f.read_text(encoding="utf-8"))


def rings(geom: dict):
    if geom["type"] == "Polygon":
        yield from geom["coordinates"][:1]
    elif geom["type"] == "MultiPolygon":
        for poly in geom["coordinates"]:
            yield poly[0]


def draw(ax, features, **kw):
    for ft in features:
        for ring in rings(ft["geometry"]):
            ax.add_patch(Polygon(np.asarray(ring), closed=True, **kw))


def find(features, name: str):
    norm = lambda s: s.lower().replace("í", "i").replace("ó", "o").replace("á", "a")  # noqa: E731
    return [f for f in features if norm(f["properties"].get("shapeName", "")) == norm(name)]


def fig_14():
    U.set_style()
    adm1, adm2 = load("ADM1")["features"], load("ADM2")["features"]
    cordoba = find(adm1, "Córdoba")
    monteria = find(adm2, "Montería")
    assert cordoba and monteria, "Córdoba o Montería no encontrados en geoBoundaries"
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.2), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.12})
    # (a) Colombia, Córdoba resaltado
    ax = axes[0]
    draw(ax, adm1, facecolor="0.93", edgecolor="0.55", lw=0.3)
    draw(ax, cordoba, facecolor=U.OKABE_ITO[1], edgecolor="0.25", lw=0.6, alpha=0.85)
    draw(ax, monteria, facecolor=U.OKABE_ITO[6], edgecolor="black", lw=0.6)
    ax.set_xlim(-79.5, -66.5); ax.set_ylim(-4.6, 13.0)
    ax.set_aspect("equal")
    ax.annotate("Montería", xy=(-75.9, 8.6), xytext=(-73.6, 10.9), fontsize=7,
                arrowprops=dict(arrowstyle="-", lw=0.6, color="black"))
    ax.text(-74.4, 7.4, "Córdoba", fontsize=6.5, color="0.15")
    ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")
    ax.tick_params(labelsize=7)
    # (b) Montería y fuentes climáticas
    ax = axes[1]
    lon0, lon1, lat0, lat1 = -76.35, -75.45, 8.25, 9.15
    near = [f for f in adm2 if any(lon0 - 0.5 < np.mean(np.asarray(r)[:, 0]) < lon1 + 0.5 and
                                   lat0 - 0.5 < np.mean(np.asarray(r)[:, 1]) < lat1 + 0.5
                                   for r in rings(f["geometry"]))]
    draw(ax, near, facecolor="0.95", edgecolor="0.6", lw=0.4)
    draw(ax, monteria, facecolor="#FBE3D3", edgecolor="black", lw=0.8)
    x0, y0, w, h = CHIRPS_BOX
    ax.add_patch(Rectangle((x0, y0), w, h, fill=False, edgecolor=U.OKABE_ITO[5], lw=1.2, ls="--",
                           label="CHIRPS averaging box (urban area)"))
    ax.plot(*STATION, marker="^", ms=7, color=U.OKABE_ITO[6], mec="black", mew=0.5, ls="none",
            label="IDEAM station 0013035501")
    ax.plot(*NASA, marker="s", ms=6, color=U.OKABE_ITO[3], mec="black", mew=0.5, ls="none",
            label="NASA POWER point")
    ax.set_xlim(lon0, lon1); ax.set_ylim(lat0, lat1)
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude (°)")
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc="lower left", frameon=True, framealpha=0.9)
    ax.text(-75.98, 8.52, "Montería\nmunicipality", ha="center", va="center", fontsize=7, style="italic")
    for a in axes:
        a.grid(False)
    fig.canvas.draw()
    top = max(a.get_position().y1 for a in axes)
    for a, lab in zip(axes, "ab"):
        fig.text(a.get_position().x0 - 0.045, top + 0.02, f"({lab})", fontweight="bold", fontsize=11, va="bottom")
    U.save_figure(fig, "fig_14_study_area")


if __name__ == "__main__":
    fig_14()
