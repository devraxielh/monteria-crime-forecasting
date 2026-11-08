from __future__ import annotations

import holidays
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

U = __import__("importlib").import_module("00_config_and_utils") if __name__ != "__main__" else None
if U is None:
    import importlib
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    U = importlib.import_module("00_config_and_utils")

# ------------------------------------------------------------------ grupos de variables
LAG_FEATURES = ["lag7", "lag14", "lag21", "lag28", "lag364", "sn4", "rm7", "rm28", "rm91", "rm365",
                "rsd28", "zero28", "total_rm28"]
CAL_CORE = ["dow", "month", "dom", "doy_s1", "doy_c1", "doy_s2", "doy_c2", "weekend", "trend", "covid_lockdown"]
HOLIDAY = ["festivo", "puente", "vispera_festivo", "temporada_decembrina", "semana_santa"]
PAYDAY = ["quincena"]
CLIMATE = ["t_max", "t_media", "t_min", "precip_mm", "rain_day", "precip_rm7", "tmax_rm7"]
# La humedad (y por tanto el índice de calor) solo es confiable 2015-06..2019-09: después el sensor 0027
# marca 3 % / 100 % (oct-2019 a mar-2024) o queda casi constante (desde abr-2024). No se usa para pronóstico;
# heat_index se conserva en el panel solo para el análisis de sensibilidad del DLNM (script 07).
RH_VALID = (pd.Timestamp("2015-06-01"), pd.Timestamp("2019-09-30"))
CATEGORICAL = ["series_id", "dow", "month"]
FEATURE_GROUPS = {"Lags": LAG_FEATURES, "Calendar": CAL_CORE, "Holidays": HOLIDAY,
                  "Payday": PAYDAY, "Climate": CLIMATE}
ALL_FEATURES = ["series_id"] + LAG_FEATURES + CAL_CORE + HOLIDAY + PAYDAY + CLIMATE
COVID_LOCKDOWN = (pd.Timestamp("2020-03-25"), pd.Timestamp("2020-08-31"))  # aislamiento obligatorio nacional
PRECIP_FAILURE = (pd.Timestamp("2017-10-01"), pd.Timestamp("2023-06-30"))  # pluviómetro registra 0 mm


def load_climate() -> pd.DataFrame:
    c = pd.read_csv(U.DATA_RAW / "clima_ideam_garzones_diario.csv", parse_dates=["fecha"]).set_index("fecha")
    c.loc[c["n_temp"] < 0.7 * 24, ["t_media", "t_max", "t_min"]] = np.nan      # < 70 % de horas
    c.loc[c["n_prec"] < 0.7 * 144, "precip_mm"] = np.nan                       # < 70 % de registros 10-min
    c.loc[PRECIP_FAILURE[0]:PRECIP_FAILURE[1], "precip_mm"] = np.nan           # falla del sensor
    c.loc[c["n_rh"] < 0.7 * 24, ["rh_media", "rh_min"]] = np.nan
    c.loc[(c.index < RH_VALID[0]) | (c.index > RH_VALID[1]), ["rh_media", "rh_min"]] = np.nan
    # índice de calor diario máximo aproximado: Tmax con la HR mínima del día (tarde)
    c["heat_index"] = U.heat_index_c(c["t_max"], c["rh_min"])
    c = c[["t_media", "t_max", "t_min", "precip_mm", "rh_media", "heat_index"]]
    idx = pd.date_range(U.TRAIN_START - pd.Timedelta(days=400), U.TEST_END, freq="D")
    c = c.reindex(idx)
    c["rain_day"] = (c["precip_mm"] >= 1.0).astype(float).where(c["precip_mm"].notna())
    c["precip_rm7"] = c["precip_mm"].rolling(7, min_periods=4).mean()
    c["tmax_rm7"] = c["t_max"].rolling(7, min_periods=4).mean()
    return c


def calendar_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
    co = holidays.Colombia(years=range(idx.year.min(), idx.year.max() + 1))
    fest = pd.Series(idx.normalize().isin(pd.to_datetime(list(co.keys()))).astype(int), index=idx)
    cal = pd.DataFrame(index=idx)
    cal["dow"], cal["month"], cal["dom"] = idx.day_of_week, idx.month, idx.day
    doy = idx.day_of_year / 365.25
    for k in (1, 2):
        cal[f"doy_s{k}"], cal[f"doy_c{k}"] = np.sin(2 * np.pi * k * doy), np.cos(2 * np.pi * k * doy)
    cal["weekend"] = (idx.day_of_week >= 5).astype(int)
    cal["trend"] = (idx - U.TRAIN_START).days / 365.25
    cal["covid_lockdown"] = ((idx >= COVID_LOCKDOWN[0]) & (idx <= COVID_LOCKDOWN[1])).astype(int)
    cal["festivo"] = fest.values
    # puente: sábado-domingo-lunes cuando el lunes es festivo (Ley Emiliani)
    mon_fest = (fest & (idx.day_of_week == 0)).astype(int)
    puente = mon_fest | mon_fest.shift(-1, fill_value=0) | mon_fest.shift(-2, fill_value=0)
    cal["puente"] = puente.astype(int).values
    cal["vispera_festivo"] = fest.shift(-1, fill_value=0).values
    cal["temporada_decembrina"] = (((idx.month == 12) & (idx.day >= 7)) | ((idx.month == 1) & (idx.day <= 6))).astype(int)
    sem_santa = np.zeros(len(idx), int)
    for d, name in co.items():
        if "Viernes Santo" in name or "Good Friday" in name:
            gf = pd.Timestamp(d)
            sem_santa |= ((idx >= gf - pd.Timedelta(days=5)) & (idx <= gf + pd.Timedelta(days=2))).astype(int)
    cal["semana_santa"] = sem_santa
    last_dom = idx.days_in_month
    cal["quincena"] = (np.isin(idx.day, [1, 15, 16]) | (idx.day == last_dom) | (idx.day == last_dom - 1)).astype(int)
    return cal


def build_panel() -> pd.DataFrame:
    d = pd.read_csv(U.DATA_PROCESSED / "panel_diario.csv", parse_dates=["fecha"]).set_index("fecha")
    d = d.loc[:U.TEST_END, U.SERIES]
    total = d.sum(axis=1)
    idx = d.index
    cal, clim = calendar_features(idx), load_climate().reindex(idx)
    frames = []
    for sid, s in enumerate(U.SERIES):
        y = d[s].astype(float)
        f = pd.DataFrame({"date": idx, "series": s, "series_id": sid, "y": y.values}, index=idx)
        for L in (7, 14, 21, 28, 364):
            f[f"lag{L}"] = y.shift(L)
        f["sn4"] = f[["lag7", "lag14", "lag21", "lag28"]].mean(axis=1)
        y7 = y.shift(U.MIN_LAG)  # toda ventana móvil termina en t-7 (sin fuga para h<=7)
        for w in (7, 28, 91, 365):
            f[f"rm{w}"] = y7.rolling(w, min_periods=int(0.8 * w)).mean()
        f["rsd28"] = y7.rolling(28, min_periods=22).std()
        f["zero28"] = (y7 == 0).astype(float).where(y7.notna()).rolling(28, min_periods=22).mean()
        f["total_rm28"] = total.shift(U.MIN_LAG).rolling(28, min_periods=22).mean()
        frames.append(f.join(cal).join(clim))
    p = pd.concat(frames, ignore_index=True)
    p = p[(p["date"] >= U.TRAIN_START) & (p["date"] <= U.TEST_END)]
    p = p.dropna(subset=["y"]).reset_index(drop=True)  # delitos_informaticos y faltantes declarados
    return p


def make_folds() -> tuple[pd.DataFrame, pd.DataFrame]:
    starts = pd.date_range(U.CAL_START, U.TEST_END, freq="QS")
    folds, origins = [], []
    for k, fs in enumerate(starts):
        fe = min(fs + pd.offsets.QuarterEnd(0), U.TEST_END)
        folds.append({"fold": k, "train_end": fs - pd.Timedelta(days=1), "test_start": fs, "test_end": fe,
                      "role": "test" if fs >= U.TEST_START else "calibration"})
        o = fs - pd.Timedelta(days=1)
        while o < fe:  # orígenes semanales: cada pronóstico usa datos hasta o y cubre o+1..o+7
            for h in range(1, U.HORIZON + 1):
                t = o + pd.Timedelta(days=h)
                if t <= fe:
                    origins.append({"fold": k, "origin": o, "date": t, "h": h})
            o += pd.Timedelta(days=U.HORIZON)
    return pd.DataFrame(folds), pd.DataFrame(origins)


def cohort_table(p: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s in U.SERIES:
        q = p[p.series == s]
        tr, te = q[q.date < U.TEST_START], q[q.date >= U.TEST_START]
        rows.append({
            "Series": U.SERIES_LABEL[s], "Train days": len(tr), "Test days": len(te),
            "Train mean": tr.y.mean(), "Test mean": te.y.mean(), "Var/mean (train)": tr.y.var() / tr.y.mean(),
            "Zero days % (train)": 100 * (tr.y == 0).mean(), "Zero days % (test)": 100 * (te.y == 0).mean(),
            "Max": q.y.max(), "Total cases": q.y.sum(),
        })
    return pd.DataFrame(rows)


def fig_01(p: pd.DataFrame, clim: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(U.COL2, 5.0))
    # (a) proporción de días con cero vs media diaria
    ax = axes[0, 0]
    tr = p[p.date < U.TEST_START]
    g = tr.groupby("series").y.agg(mean="mean", zeros=lambda s: (s == 0).mean())
    g = g.sort_values("zeros")
    ypos = np.arange(len(g))
    ax.barh(ypos, 100 * g["zeros"], color=U.OKABE_ITO[5], height=0.65)
    for yp, (s, r) in zip(ypos, g.iterrows()):
        ax.text(100 * r["zeros"] + 1.5, yp, f"{r['mean']:.2f}/day", va="center", fontsize=6.5)
    ax.set_yticks(ypos, [U.SERIES_LABEL[s] for s in g.index], fontsize=7.5)
    ax.set_xlim(0, 105); ax.set_xlabel("Zero-count days in training period (%)")
    ax.grid(axis="y", visible=False)
    U.panel_label(ax, "a", x=-0.42)
    # (b) serie semanal de 4 delitos representativos
    ax = axes[0, 1]
    sel = ["hurto_personas", "violencia_intrafamiliar", "delitos_informaticos", "homicidio"]
    styles = ["-", "--", "-.", ":"]
    for s, ls, c in zip(sel, styles, U.OKABE_ITO[1:5]):
        w = p[p.series == s].set_index("date").y.resample("W").sum()
        ax.plot(w.index, w.values, ls, lw=0.8, color=c if c != "#F0E442" else "#CC79A7", label=U.SERIES_LABEL[s])
    ax.axvspan(U.TEST_START, U.TEST_END, color="0.85", alpha=0.5, lw=0)
    ax.set_ylabel("Weekly count"); ax.set_ylim(0, 105)
    ax.legend(loc="upper left", ncol=2, fontsize=6.5)
    U.panel_label(ax, "b")
    # (c) recuentos diarios por día de la semana (hurto a personas, lesiones)
    ax = axes[1, 0]
    dd = tr[tr.series.isin(["hurto_personas", "lesiones_personales", "homicidio_transito"])].copy()
    dd["rel"] = dd.y / dd.groupby("series").y.transform("mean")
    m = dd.groupby(["series", "dow"]).rel.mean().reset_index()
    for (s, grp), mk, c in zip(m.groupby("series"), ["o", "s", "^"], U.OKABE_ITO[5:8]):
        ax.plot(grp.dow, grp.rel, marker=mk, ms=4, lw=1, color=c, label=U.SERIES_LABEL[s])
    ax.set_xticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.axhline(1, color="0.4", lw=0.6); ax.set_ylabel("Relative mean count"); ax.legend(fontsize=6.5)
    U.panel_label(ax, "c")
    # (d) clima oficial IDEAM: Tmax mensual y precipitación
    ax = axes[1, 1]
    mclim = clim.loc[U.TRAIN_START:U.TEST_END].resample("MS").agg({"t_max": "mean", "precip_mm": lambda s: s.sum(min_count=20)})
    ax.bar(mclim.index, mclim["precip_mm"], width=25, color=U.OKABE_ITO[2], label="Precipitation")
    ax.set_ylabel("Monthly precipitation (mm)"); ax.set_ylim(0, 600)
    ax.axvspan(PRECIP_FAILURE[0], PRECIP_FAILURE[1], color="0.9", lw=0, label="Rain gauge failure")
    ax2 = ax.twinx(); ax2.plot(mclim.index, mclim["t_max"], color=U.OKABE_ITO[6], lw=1, label="Mean $T_{max}$")
    ax2.set_ylabel(r"Mean daily $T_{max}$ ($^\circ$C)"); ax2.grid(False); ax2.spines["right"].set_visible(True)
    ax2.set_ylim(26, 38)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", ncol=3, fontsize=6.5)
    U.panel_label(ax, "d")
    fig.tight_layout()
    U.save_figure(fig, "fig_01_data_distribution")


def main() -> None:
    p = build_panel()
    folds, origins = make_folds()
    p.to_parquet(U.RESULTS / "panel_features.parquet", index=False)
    p.to_csv(U.DATA_PROCESSED / "dataset_ml_ready.csv", index=False)
    folds.to_csv(U.RESULTS / "folds.csv", index=False)
    origins.to_csv(U.RESULTS / "origins.csv", index=False)
    t = cohort_table(p, folds)
    tex = t.copy()
    for c in tex.columns[1:]:
        tex[c] = tex[c].map(lambda v: f"{v:,.0f}" if float(v).is_integer() and abs(v) >= 10 else U.fmt(v, 2))
    U.save_table(t, "table_01_cohort_summary", tex)
    fig_01(p, load_climate())
    clim_cov = p.drop_duplicates("date").set_index("date")[["t_max", "precip_mm"]].notna()
    print(f"panel: {p.shape}, folds: {len(folds)} ({(folds.role == 'test').sum()} test), "
          f"origins: {origins.origin.nunique()}, days: {origins.date.nunique()}")
    print("cobertura clima train/test:\n",
          clim_cov.groupby(clim_cov.index >= U.TEST_START).mean().round(3))


if __name__ == "__main__":
    main()
