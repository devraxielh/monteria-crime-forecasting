from __future__ import annotations

import importlib
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
U = importlib.import_module("00_config_and_utils")
P = importlib.import_module("01_data_splitting_and_prep")
M = importlib.import_module("02_baseline_and_sota_models")
warnings.filterwarnings("ignore")

QCOLS = M.QCOLS
HW = 4  # horizonte semanal
W_MODELS = ["SN-Poisson-W", "GLM-NB-W", "LGBM-NB-W", "Chronos-2-W", "Ensemble-W", "Ensemble+CQR-W"]
W_COLORS = {"SN-Poisson-W": U.MODEL_COLORS["SN-Poisson"], "GLM-NB-W": U.MODEL_COLORS["GLM-NB"],
            "LGBM-NB-W": U.MODEL_COLORS["LGBM-NB"], "Chronos-2-W": U.MODEL_COLORS["Chronos-2"],
            "Ensemble-W": U.MODEL_COLORS["Ensemble"], "Ensemble+CQR-W": U.MODEL_COLORS["Ensemble+CQR"]}
W_MARK = dict(zip(W_MODELS, ["o", "D", "P", "v", "h", "*"]))
DAILY_SHOW = ["SN-Poisson", "GLM-NB", "LGBM-NB", "Chronos-Bolt", "Chronos-2", "Chronos-2-FT", "TimesFM-2.5",
              "TiRex", U.PROPOSED]


# ------------------------------------------------------------------ A. horizonte diario
def daily_by_horizon() -> pd.DataFrame:
    preds = pd.read_parquet(U.RESULTS / "oof_predictions.parquet")
    origins = pd.read_csv(U.RESULTS / "origins.csv", parse_dates=["origin", "date"])
    s = pd.read_csv(U.RESULTS / "train_scales.csv", index_col=0).iloc[:, 0]
    scales = {(int(k.split("|")[0]), k.split("|")[1]): v for k, v in s.items()}
    d = preds.merge(origins[["fold", "date", "h"]], on=["fold", "date"], how="left")
    d["crps"] = U.crps_from_quantiles(d.y.values, d[QCOLS].values)
    d["scale"] = [scales[(f, s_)] for f, s_ in zip(d.fold, d.series)]
    d["scrps"] = d.crps / d.scale
    d["covered"] = (d.y >= d["q0.1"]) & (d.y <= d["q0.9"])
    out = d.groupby(["model", "h", "series", "fold"]).agg(sCRPS=("scrps", "mean"), Coverage80=("covered", "mean"))
    out = out.reset_index()
    out.to_csv(U.RESULTS / "horizon_scores_daily.csv", index=False)
    return out


# ------------------------------------------------------------------ B. escala semanal
def weekly_panel() -> pd.DataFrame:
    p = pd.read_parquet(U.RESULTS / "panel_features.parquet")
    p["week"] = p.date.dt.to_period("W-SUN").dt.start_time
    clim_ok = lambda s: s.mean() if s.notna().sum() >= 5 else np.nan  # noqa: E731
    g = p.groupby(["series", "week"])
    w = g.agg(y=("y", "sum"), n_days=("y", "size"), series_id=("series_id", "first"),
              n_festivos=("festivo", "sum"), puente=("puente", "max"), n_quincena=("quincena", "sum"),
              decembrina=("temporada_decembrina", "mean"), semana_santa=("semana_santa", "max"),
              covid=("covid_lockdown", "mean"), tmax_mean=("t_max", clim_ok), tmax_max=("t_max", "max"),
              hi_mean=("heat_index", clim_ok), rh_mean=("rh_media", clim_ok),
              precip_sum=("precip_mm", lambda s: s.sum() if s.notna().sum() >= 5 else np.nan),
              rain_days=("rain_day", lambda s: s.sum() if s.notna().sum() >= 5 else np.nan)).reset_index()
    w = w[w.n_days == 7].drop(columns="n_days")  # solo semanas completas
    woy = w.week.dt.isocalendar().week.astype(int).values / 52.18
    w["woy_s1"], w["woy_c1"] = np.sin(2 * np.pi * woy), np.cos(2 * np.pi * woy)
    w["month"] = w.week.dt.month
    w = w.sort_values(["series", "week"]).reset_index(drop=True)
    return w


W_EXO = ["n_festivos", "puente", "n_quincena", "decembrina", "semana_santa", "covid", "tmax_mean", "tmax_max",
         "hi_mean", "rh_mean", "precip_sum", "rain_days", "woy_s1", "woy_c1", "month"]


def lag_feats(w: pd.DataFrame, h: int) -> pd.DataFrame:
    """Rezagos y medias móviles que terminan en t-h (pronóstico directo a h semanas)."""
    out = w.copy()
    g = out.groupby("series").y
    for L in range(h, h + 4):
        out[f"lag_{L - h}"] = g.shift(L)
    out["lag52"] = g.shift(52)
    yh = g.shift(h)
    for k in (4, 13, 52):
        out[f"rm{k}"] = yh.groupby(out.series).transform(lambda s: s.rolling(k, min_periods=int(0.75 * k)).mean())
    out["zero13"] = (yh == 0).astype(float).where(yh.notna()).groupby(out.series).transform(
        lambda s: s.rolling(13, min_periods=10).mean())
    return out


W_LAGS = ["lag_0", "lag_1", "lag_2", "lag_3", "lag52", "rm4", "rm13", "rm52", "zero13"]


def w_origins(w: pd.DataFrame) -> pd.DataFrame:
    weeks = np.array(sorted(w.week.unique()))
    first = weeks[np.searchsorted(weeks, U.CAL_START)] - pd.Timedelta(days=7)  # última semana conocida
    rows = []
    o = first
    while True:
        fut = [o + pd.Timedelta(days=7 * h) for h in range(1, HW + 1)]
        if fut[-1] > weeks[-1]:
            break
        role = "test" if fut[0] >= U.TEST_START else "calibration"
        for h, t in enumerate(fut, 1):
            rows.append({"origin": o, "week": t, "h": h, "role": role})
        o += pd.Timedelta(days=7 * HW)
    return pd.DataFrame(rows)


def run_weekly(w: pd.DataFrame, orgs: pd.DataFrame) -> pd.DataFrame:
    import torch
    from chronos import BaseChronosPipeline

    c2 = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map=U.torch_device(), dtype=torch.float32)
    feats_by_h = {h: lag_feats(w, h) for h in range(1, HW + 1)}
    params = {**M.LGBM_PARAMS, "n_estimators": 400, "min_child_samples": 30}
    rows = []
    t0 = time.time()
    for o, og in orgs.groupby("origin"):
        tr_mask_week = w.week <= o
        for h in range(1, HW + 1):
            t = o + pd.Timedelta(days=7 * h)
            fh = feats_by_h[h]
            tr, te = fh[fh.week <= o].dropna(subset=["lag_3"]), fh[fh.week == t]
            feats = ["series_id"] + W_LAGS + W_EXO
            # LGBM-NB-W directo para h, dispersión NB por serie en validación interna (últimas 26 semanas)
            cut = o - pd.Timedelta(days=7 * 26)
            a, b = tr[tr.week <= cut], tr[tr.week > cut]
            mi = M._lgbm_fit(M._prep_X(a, feats), a.y, params)  # noqa: SLF001
            mva = mi.predict(M._prep_X(b, feats))  # noqa: SLF001
            alphas = {s: U.nb_alpha_moments(b.y[b.series == s], mva[(b.series == s).values]) for s in U.SERIES}
            mf = M._lgbm_fit(M._prep_X(tr, feats), tr.y, params)  # noqa: SLF001
            mu = mf.predict(M._prep_X(te, feats))  # noqa: SLF001
            q = M.nb_quantiles_by_series(te.series, mu, alphas)
            for i, (_, r) in enumerate(te.iterrows()):
                rows.append(("LGBM-NB-W", r.series, t, o, h, r.y, mu[i], *q[i]))
            # SN-Poisson-W
            for _, r in te.iterrows():
                m_ = r.rm4 if pd.notna(r.rm4) else r.rm13
                rows.append(("SN-Poisson-W", r.series, t, o, h, r.y, m_, *U.count_quantiles([max(m_, 1e-3)], 0.0)[0]))
            # GLM-NB-W (Pipeline sklearn ajustado en cada origen)
            gfeats = W_LAGS + [f for f in W_EXO if f != "month"]
            for s in U.SERIES:
                a_s, b_s, t_s = a[a.series == s], b[b.series == s], te[te.series == s]
                if t_s.empty:
                    continue
                pin = _wglm().fit(a_s[gfeats], a_s.y)
                al = U.nb_alpha_moments(b_s.y, pin.predict(b_s[gfeats]))
                pf = _wglm().fit(tr[tr.series == s][gfeats], tr[tr.series == s].y)
                m_ = pf.predict(t_s[gfeats])
                qq = U.count_quantiles(m_, al)
                rows.append(("GLM-NB-W", s, t, o, h, t_s.y.iloc[0], m_[0], *qq[0]))
        # Chronos-2-W (una llamada por origen, h = 1..4)
        fut_weeks = [o + pd.Timedelta(days=7 * h) for h in range(1, HW + 1)]
        cov = w.drop_duplicates("week").set_index("week")[W_EXO].astype(float)
        ctx = []
        for s in U.SERIES:
            ws = w[(w.series == s) & tr_mask_week].set_index("week")
            c = cov.loc[ws.index].copy(); c["target"] = ws.y.astype(float); c["id"] = s; c["timestamp"] = c.index
            ctx.append(c.iloc[-260:])
        ctx = pd.concat(ctx, ignore_index=True)
        fut = pd.concat([cov.loc[fut_weeks].assign(id=s, timestamp=fut_weeks) for s in U.SERIES], ignore_index=True)
        out = c2.predict_df(ctx, future_df=fut, id_column="id", timestamp_column="timestamp", target="target",
                            prediction_length=HW, quantile_levels=list(U.QUANTILES), freq="W-MON")
        yv = w.set_index(["series", "week"]).y
        for _, r in out.iterrows():
            t = pd.Timestamp(r["timestamp"]); h = fut_weeks.index(t) + 1
            qq = np.sort(np.clip([r[str(x)] for x in U.QUANTILES], 0, None))
            rows.append(("Chronos-2-W", r["id"], t, o, h, yv.loc[(r["id"], t)], qq.mean(), *qq))
        print(f"origen {o.date()} ({time.time() - t0:.0f}s)", flush=True)
    d = pd.DataFrame(rows, columns=["model", "series", "week", "origin", "h", "y", "mean"] + QCOLS)
    d[QCOLS] = np.sort(np.clip(d[QCOLS].values, 0, None), axis=1)
    # ensamble y CQR secuencial (calibración = orígenes anteriores, misma serie y horizonte)
    a = d[d.model == "LGBM-NB-W"].set_index(["series", "week", "origin", "h"]).sort_index()
    b = d[d.model == "Chronos-2-W"].set_index(["series", "week", "origin", "h"]).reindex(a.index)
    ens = a[["y"]].copy(); ens["mean"] = (a["mean"] + b["mean"]) / 2
    ens[QCOLS] = (a[QCOLS].values + b[QCOLS].values) / 2; ens["model"] = "Ensemble-W"
    ens = ens.reset_index()
    cq = []
    for (s, h), g in ens.groupby(["series", "h"]):
        g = g.sort_values("origin").copy()
        qv = g[QCOLS].values.copy()
        for i, o in enumerate(g.origin.values):
            calm = g.week.values <= o  # solo semanas ya observadas en el origen (sin fuga)
            qv[i] = U.conformalize(g[QCOLS].values[i:i + 1], g[QCOLS].values[calm], g.y.values[calm])[0] \
                if calm.sum() >= 8 else g[QCOLS].values[i]
        g[QCOLS] = qv; g["model"] = "Ensemble+CQR-W"; cq.append(g)
    d = pd.concat([d, ens, pd.concat(cq)], ignore_index=True)
    d = d.merge(orgs[["origin", "role"]].drop_duplicates(), on="origin")
    return d


def _wglm():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import PoissonRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler()),
                     ("glm", PoissonRegressor(alpha=1e-2, max_iter=3000))])


def weekly_scores(d: pd.DataFrame, w: pd.DataFrame) -> pd.DataFrame:
    d = d[d.role == "test"].copy()
    d["quarter"] = d.origin.dt.to_period("Q").astype(str)
    rows = []
    for (m, s, qtr, h), g in d.groupby(["model", "series", "quarter", "h"]):
        o = g.origin.min()
        ytr = w[(w.series == s) & (w.week <= o)].y.values
        r = U.evaluate_block(g.y.values, g[QCOLS].values, g["mean"].values, U.snaive_scale(ytr, 1))
        rows.append({"model": m, "series": s, "quarter": qtr, "h": h, **r})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ tabla y figura
def table_06(hd: pd.DataFrame, ws: pd.DataFrame) -> None:
    daily_scores = pd.read_csv(U.RESULTS / "fold_scores.csv")
    rows = []
    for m in U.MODEL_ORDER:
        r = {"Scale": "Daily", "Model": m}
        for h in (1, 7):
            v = hd[(hd.model == m) & (hd.h == h)].sCRPS
            r[f"h{h}_mean"], r[f"h{h}_sd"] = v.mean(), v.std()
        r["all_mean"] = daily_scores[daily_scores.model == m].sCRPS.mean()
        r["cov80"] = daily_scores[daily_scores.model == m].Coverage80.mean()
        rows.append(r)
    for m in W_MODELS:
        r = {"Scale": "Weekly", "Model": m}
        for h in (1, 4):
            v = ws[(ws.model == m) & (ws.h == h)].sCRPS
            r[f"h{h}_mean"], r[f"h{h}_sd"] = v.mean(), v.std()
        r["all_mean"] = ws[ws.model == m].sCRPS.mean()
        r["cov80"] = ws[ws.model == m].Coverage80.mean()
        rows.append(r)
    t = pd.DataFrame(rows)
    f = lambda a, b: "--" if pd.isna(a) else f"{a:.3f} $\\pm$ {b:.3f}"  # noqa: E731
    tex = pd.DataFrame({"Scale": t.Scale, "Model": t.Model,
                        "sCRPS h=1": [f(a, b) for a, b in zip(t.h1_mean, t.h1_sd)],
                        "sCRPS h=7 d": [f(a, b) for a, b in zip(t.h7_mean, t.h7_sd)],
                        "sCRPS h=4 wk": [f(a, b) for a, b in zip(t.h4_mean, t.h4_sd)],
                        "sCRPS (all h)": [f"{x:.3f}" for x in t.all_mean],
                        "Cov. 80%": [f"{x:.3f}" for x in t.cov80]})
    U.save_table(t, "table_06_multiscale", tex, column_format="llrrrrr")


def fig_07(hd: pd.DataFrame, ws: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(U.COL2, 5.6))
    gs = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.35)
    ax = fig.add_subplot(gs[0, 0])
    for m in DAILY_SHOW:
        v = hd[hd.model == m].groupby("h").sCRPS.mean()
        ax.plot(v.index, v.values, marker=U.MODEL_MARKERS[m], ms=3.5, lw=1, color=U.MODEL_COLORS[m], label=m)
    ax.set_xlabel("Forecast horizon (days)"); ax.set_ylabel("Mean sCRPS (daily)")
    ax.legend(fontsize=5.8, ncol=2)
    U.panel_label(ax, "a", x=-0.28)
    ax = fig.add_subplot(gs[0, 1])
    for m in W_MODELS:
        v = ws[ws.model == m].groupby("h").sCRPS.mean()
        ax.plot(v.index, v.values, marker=W_MARK[m], ms=4, lw=1, color=W_COLORS[m], label=m)
    ax.set_xticks(range(1, HW + 1)); ax.set_xlabel("Forecast horizon (weeks)"); ax.set_ylabel("Mean sCRPS (weekly)")
    ax.legend(fontsize=5.8, ncol=2)
    U.panel_label(ax, "b", x=-0.28)
    # (c) habilidad vs. SN por delito: diaria vs. semanal (mejor modelo de cada escala)
    ax = fig.add_subplot(gs[1, :])
    ds = pd.read_csv(U.RESULTS / "fold_scores.csv")
    dpiv = ds.pivot_table(index="series", columns="model", values="sCRPS")
    wpiv = ws.pivot_table(index="series", columns="model", values="sCRPS")
    sk_d = 100 * (1 - dpiv["Ensemble+CQR"] / dpiv["SN-Poisson"])  # mismo método en ambas escalas
    sk_w = 100 * (1 - wpiv["Ensemble+CQR-W"] / wpiv["SN-Poisson-W"])
    order = list(sk_d.sort_values().index)
    y = np.arange(len(order))
    ax.hlines(y, sk_d[order], sk_w[order], color="0.7", lw=1.2)
    ax.scatter(sk_d[order], y, color=U.MODEL_COLORS["Ensemble+CQR"], s=20, label="Daily (h = 1-7 d)", zorder=3)
    ax.scatter(sk_w[order], y, color=U.OKABE_ITO[1], marker="s", s=20, label="Weekly (h = 1-4 wk)", zorder=3)
    ax.axvline(0, color="0.3", lw=0.6)
    ax.set_yticks(y, [U.SERIES_LABEL[s] for s in order], fontsize=7)
    ax.set_xlabel("Skill of Ensemble+CQR over seasonal-naive Poisson (% sCRPS reduction)")
    ax.legend(fontsize=6.5, loc="lower right"); ax.grid(axis="y", visible=False)
    U.panel_label(ax, "c", x=-0.17)
    U.save_figure(fig, "fig_07_horizon_and_weekly")


def main() -> None:
    hd = daily_by_horizon()
    w = weekly_panel()
    w.to_csv(U.RESULTS / "weekly_panel.csv", index=False)
    orgs = w_origins(w)
    d = run_weekly(w, orgs)
    d.to_parquet(U.RESULTS / "weekly_predictions.parquet", index=False)
    ws = weekly_scores(d, w)
    ws.to_csv(U.RESULTS / "weekly_scores.csv", index=False)
    table_06(hd, ws)
    fig_07(hd, ws)
    print(hd.groupby(["model", "h"]).sCRPS.mean().unstack().reindex(U.MODEL_ORDER).round(4))
    print(ws.groupby(["model", "h"]).sCRPS.mean().unstack().reindex(W_MODELS).round(4))
    print(ws.groupby("model")[["sCRPS", "Coverage80"]].mean().reindex(W_MODELS).round(4))


if __name__ == "__main__":
    main()
