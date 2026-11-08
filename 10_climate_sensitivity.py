from __future__ import annotations

import importlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
U = importlib.import_module("00_config_and_utils")
P = importlib.import_module("01_data_splitting_and_prep")
M = importlib.import_module("02_baseline_and_sota_models")
E = importlib.import_module("05_explainability_and_error")
D = importlib.import_module("07_dlnm_heat")

SRC_IDEAM, SRC_ALT = "IDEAM station", "NASA POWER + CHIRPS"
CLIM_EXTRA = ["rh_media", "heat_index"]
DLNM_OUT = ["Violent (assault + homicide + DV)", "Property theft (4 types)", "Assault", "Domestic violence",
            "Theft (persons)", "Burglary", "Traffic homicide", "Homicide"]


# ------------------------------------------------------------------ datos
def alt_climate() -> pd.DataFrame:
    n = pd.read_csv(U.DATA_RAW / "clima_nasa_power_monteria.csv", parse_dates=["fecha"]).set_index("fecha")
    c = pd.read_csv(U.DATA_RAW / "clima_chirps_monteria.csv", parse_dates=["fecha"]).set_index("fecha")
    a = pd.DataFrame(index=n.index)
    a["t_max"], a["t_media"], a["t_min"] = n.T2M_MAX, n.T2M, n.T2M_MIN
    a["rh_media"] = n.RH2M
    q = n.QV2M / 1000.0
    e = q * (n.PS * 10.0) / (0.622 + 0.378 * q)                        # hPa
    es = 6.112 * np.exp(17.67 * n.T2M_MAX / (n.T2M_MAX + 243.5))
    a["rh_at_tmax"] = np.clip(100 * e / es, 1, 100)
    a["heat_index"] = U.heat_index_c(a.t_max, a.rh_at_tmax)
    a["precip_mm"] = c.precip_chirps_mm.reindex(a.index)
    a["rain_day"] = (a.precip_mm >= 1.0).astype(float)
    a["precip_rm7"] = a.precip_mm.rolling(7, min_periods=4).mean()
    a["tmax_rm7"] = a.t_max.rolling(7, min_periods=4).mean()
    a.index.name = "date"
    return a


def swap_climate(p: pd.DataFrame, a: pd.DataFrame) -> pd.DataFrame:
    cols = P.CLIMATE + CLIM_EXTRA
    q = p.drop(columns=[c for c in cols if c in p]).merge(a[cols], left_on="date", right_index=True, how="left")
    return q[p.columns]


def agreement(p: pd.DataFrame, a: pd.DataFrame) -> pd.DataFrame:
    i = p.drop_duplicates("date").set_index("date")[["t_max", "t_media", "precip_mm", "rh_media", "heat_index"]]
    i = i.loc[U.TRAIN_START:U.TEST_END]
    j = i.join(a[["t_max", "t_media", "precip_mm", "rh_media", "heat_index"]], rsuffix="_alt")
    rows = []
    for v in ["t_max", "t_media", "precip_mm", "rh_media", "heat_index"]:
        x = j[[v, f"{v}_alt"]].dropna()
        rows.append({"variable": v, "n_days_ideam": len(x), "pct_days_ideam": 100 * len(x) / len(j),
                     "pearson_r": x[v].corr(x[f"{v}_alt"]), "spearman_r": x[v].corr(x[f"{v}_alt"], method="spearman"),
                     "bias_alt_minus_ideam": (x[f"{v}_alt"] - x[v]).mean(),
                     "p50_ideam": x[v].median(), "p50_alt": j[f"{v}_alt"].median(),
                     "p99_ideam": x[v].quantile(.99), "p99_alt": j[f"{v}_alt"].quantile(.99)})
    x = j[["precip_mm", "precip_mm_alt"]].dropna()
    rows.append({"variable": "rain_day_agreement", "n_days_ideam": len(x), "pct_days_ideam": 100 * len(x) / len(j),
                 "pearson_r": float(((x.precip_mm >= 1) == (x.precip_mm_alt >= 1)).mean())})
    mo = x.resample("MS").sum(min_count=25).dropna()
    rows.append({"variable": "monthly_precip", "n_days_ideam": len(mo), "pearson_r": mo.precip_mm.corr(mo.precip_mm_alt)})
    t = pd.DataFrame(rows)
    t.to_csv(U.RESULTS / "climate_source_agreement.csv", index=False)
    return t


# ------------------------------------------------------------------ (i) pronóstico
def forecast(p: pd.DataFrame, p_alt: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    scales = pd.read_csv(U.RESULTS / "train_scales.csv", index_col=0).iloc[:, 0]
    scales = {(int(k.split("|")[0]), k.split("|")[1]): v for k, v in scales.items()}
    F = P.ALL_FEATURES
    no_clim = [f for f in F if f not in P.CLIMATE]
    cfgs = {"No climate": (p, no_clim), f"Climate: {SRC_IDEAM}": None, f"Climate: {SRC_ALT}": (p_alt, F),
            f"Climate: {SRC_ALT} + humidity & heat index": (p_alt, F + CLIM_EXTRA)}
    preds = []
    full = pd.read_parquet(U.RESULTS / "predictions_LGBM-NB.parquet")
    preds.append(full[full.fold.isin(set(folds.fold))].assign(model=f"Climate: {SRC_IDEAM}"))
    for name, cfg in cfgs.items():
        if cfg is None:
            continue
        pp, feats = cfg
        for _, f in folds.iterrows():
            mu, q, _, _, te = M.fit_predict_lgbm(pp, f, feats)
            preds.append(M.package(te, name, f.fold, mu, q))
        print(f"  {name}: listo", flush=True)
    sc = M.score(pd.concat(preds, ignore_index=True), scales)
    sc.to_csv(U.RESULTS / "climate_sensitivity_scores.csv", index=False)
    base = sc[sc.model == "No climate"].set_index(["fold", "series"]).sort_index()
    rows = []
    for name in cfgs:
        g = sc[sc.model == name].set_index(["fold", "series"]).sort_index()
        r = {"Configuration": name, "sCRPS_mean": g.sCRPS.mean(), "sCRPS_sd": g.sCRPS.std(),
             "MASE_mean": g.MASE.mean(), "Coverage80_mean": g.Coverage80.mean()}
        if name != "No climate":
            d = (g.sCRPS - base.sCRPS).values
            lo, hi = U.bootstrap_ci(d)
            r.update({"delta_vs_no_climate": d.mean(), "delta_ci_low": lo, "delta_ci_high": hi,
                      "delta_pct": 100 * d.mean() / base.sCRPS.mean(),
                      "p_value": stats.wilcoxon(g.sCRPS.values, base.sCRPS.values).pvalue})
        rows.append(r)
    t = pd.DataFrame(rows)
    m = t.p_value.notna()
    t.loc[m, "p_holm"] = multipletests(t.loc[m, "p_value"], method="holm")[1]
    tex = pd.DataFrame({
        "Configuration": t.Configuration,
        "sCRPS": [f"{a:.4f} $\\pm$ {b:.3f}" for a, b in zip(t.sCRPS_mean, t.sCRPS_sd)],
        r"$\Delta$ vs. no climate [95% CI]": ["--" if pd.isna(d) else f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]"
                                              for d, lo, hi in zip(t.delta_vs_no_climate, t.delta_ci_low,
                                                                   t.delta_ci_high)],
        r"$\Delta$ %": ["--" if pd.isna(x) else f"{x:+.2f}" for x in t.delta_pct],
        "p (Holm)": ["--" if pd.isna(x) else f"{x:.2e}" for x in t.p_holm],
        "Cov. 80%": [f"{x:.3f}" for x in t.Coverage80_mean]})
    U.save_table(t, "table_12_climate_sensitivity_forecast", tex, column_format="lrrrrr")
    return t


# ------------------------------------------------------------------ (ii)+(iii) asociación
def glm_effects(p: pd.DataFrame, p_alt: pd.DataFrame) -> pd.DataFrame:
    out = []
    for src, pp in ((SRC_IDEAM, p), (SRC_ALT, p_alt)):
        g = E.glm_convergent(pp)
        g = g[g.effect.str.startswith(("Hot", "Rain"))].copy()
        g["effect"] = np.where(g.effect.str.startswith("Hot"), "Hot day (Tmax >= 35 C)", "Rain day (>= 1 mm)")
        g["p_holm_src"] = multipletests(g.glm_p, method="holm")[1]
        g["source"] = src
        out.append(g)
    g = pd.concat(out, ignore_index=True)
    g["rr"], g["rr_low"], g["rr_high"] = np.exp(g.glm_log_rr), np.exp(g.glm_ci_low), np.exp(g.glm_ci_high)
    return g


def dlnm(df_ideam: pd.DataFrame, a: pd.DataFrame) -> pd.DataFrame:
    df_alt = df_ideam.copy()
    df_alt["t_max"] = a.t_max.reindex(df_alt.index).values
    df_alt["heat_index"] = a.heat_index.reindex(df_alt.index).values
    rows = []
    for src, df in ((SRC_IDEAM, df_ideam), (SRC_ALT, df_alt)):
        for expo in ("t_max", "heat_index"):
            cb, di, C, _, _ = D.crossbasis(df[expo])
            xs = df[expo].interpolate(limit=2, limit_area="inside").dropna()
            ref, p99 = np.percentile(xs, [50, 99])
            for name in DLNM_OUT:
                y = df[D.OUTCOMES[name]].sum(axis=1)
                res, beta, V, n = D.fit_outcome(df, y, cb)
                W = float(beta @ np.linalg.solve(V, beta))
                est, se = D.cumulative_rr(di, C, beta, V, np.array([p99]), ref)
                rows.append({"source": src, "exposure": expo, "outcome": name, "n_days": n, "ref_p50": ref,
                             "p99": p99, "RR_p99": np.exp(est[0]), "RR_p99_low": np.exp(est[0] - 1.96 * se[0]),
                             "RR_p99_high": np.exp(est[0] + 1.96 * se[0]),
                             "wald_p": float(stats.chi2.sf(W, len(beta)))})
    t = pd.DataFrame(rows)
    t["wald_p_holm"] = np.nan
    for (s, e), idx in t.groupby(["source", "exposure"]).groups.items():
        t.loc[idx, "wald_p_holm"] = multipletests(t.loc[idx, "wald_p"], method="holm")[1]
    return t


def table_13(g: pd.DataFrame, dl: pd.DataFrame) -> None:
    ci = lambda r, a, b, c: f"{r[a]:.2f} [{r[b]:.2f}, {r[c]:.2f}]"  # noqa: E731
    rows = []
    for (e, o), x in dl.groupby(["exposure", "outcome"], sort=False):
        r = {"Analysis": "DLNM, cumulative RR P99 vs P50 (lags 0-7)",
             "Exposure": {"t_max": "Tmax", "heat_index": "Heat index"}[e], "Outcome": o}
        for s in (SRC_IDEAM, SRC_ALT):
            v = x[x.source == s].iloc[0]
            r[s] = ci(v, "RR_p99", "RR_p99_low", "RR_p99_high") + ("*" if v.wald_p_holm < U.ALPHA else "")
            r[f"{s} days"] = int(v.n_days)
        rows.append(r)
    for (e, s_), x in g.groupby(["effect", "series"], sort=False):
        r = {"Analysis": "GLM Poisson (HC0), rate ratio", "Exposure": e, "Outcome": U.SERIES_LABEL[s_]}
        for s in (SRC_IDEAM, SRC_ALT):
            v = x[x.source == s].iloc[0]
            r[s] = ci(v, "rr", "rr_low", "rr_high") + ("*" if v.p_holm_src < U.ALPHA else "")
            r[f"{s} days"] = np.nan
        rows.append(r)
    t = pd.DataFrame(rows)
    full = pd.concat([dl.assign(analysis="DLNM"), g.assign(analysis="GLM")], ignore_index=True)
    full.to_csv(U.RESULTS / "climate_sensitivity_association.csv", index=False)
    tex = t[["Analysis", "Exposure", "Outcome", SRC_IDEAM, SRC_ALT]].copy()
    tex["Analysis"] = tex.Analysis.where(~tex.Analysis.duplicated(), "")
    tex["Exposure"] = [e if (i == 0 or e != tex.Exposure.iloc[i - 1] or tex.Analysis.iloc[i] != "") else ""
                       for i, e in enumerate(tex.Exposure)]
    tex = tex.rename(columns={SRC_IDEAM: "IDEAM station", SRC_ALT: "NASA POWER + CHIRPS"})
    U.save_table(t, "table_13_climate_sensitivity_association", tex, column_format="lllcc")


# ------------------------------------------------------------------ figura
def fig_11(p, a, ft, g, dl):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.6), gridspec_kw={"hspace": 0.48, "wspace": 0.42})
    i = p.drop_duplicates("date").set_index("date").loc[U.TRAIN_START:U.TEST_END]
    # (a) Tmax diario IDEAM vs NASA
    ax = axes[0, 0]
    j = i[["t_max"]].join(a[["t_max"]], rsuffix="_alt").dropna()
    hb = ax.hexbin(j.t_max, j.t_max_alt, gridsize=28, cmap="viridis", mincnt=1, linewidths=0.1)
    lim = [min(j.min()) - 0.5, max(j.max()) + 0.5]
    ax.plot(lim, lim, color="k", lw=0.8, ls="--")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(r"IDEAM $T_{max}$ ($^\circ$C)")
    ax.set_ylabel(r"NASA POWER $T_{max}$ ($^\circ$C)")
    ax.text(0.03, 0.97, f"r = {j.t_max.corr(j.t_max_alt):.2f}\nn = {len(j):,} days", transform=ax.transAxes,
            va="top", fontsize=7)
    cb = fig.colorbar(hb, ax=ax, fraction=0.05, pad=0.02)
    cb.set_label("Days", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    # (b) precipitación mensual: IDEAM (con huecos) vs CHIRPS
    ax = axes[0, 1]
    mi = i.precip_mm.resample("MS").sum(min_count=25)
    mc = a.precip_mm.loc[U.TRAIN_START:U.TEST_END].resample("MS").sum()
    ax.plot(mc.index, mc.values, color=U.OKABE_ITO[5], lw=0.9, label="CHIRPS (complete)")
    ax.plot(mi.index, mi.values, color=U.OKABE_ITO[6], lw=0.9, marker="o", ms=1.6, label="IDEAM (months with data)")
    ax.axvspan(P.PRECIP_FAILURE[0], P.PRECIP_FAILURE[1], color="0.85", zorder=0, label="IDEAM gauge failure")
    ax.set_ylabel("Monthly precipitation (mm)")
    ax.set_xlabel("Date")
    ax.legend(fontsize=6, frameon=False, loc="upper left", ncol=1)
    ax.set_ylim(0, mc.max() * 1.45)
    # (c) DLNM RR P99 (Tmax e índice de calor) por fuente
    ax = axes[1, 0]
    outs = ["Violent (assault + homicide + DV)", "Property theft (4 types)", "Assault", "Domestic violence",
            "Burglary", "Traffic homicide"]
    short = {"Violent (assault + homicide + DV)": "Violent (pooled)", "Property theft (4 types)": "Property (pooled)"}
    combos = [(SRC_IDEAM, "t_max", "IDEAM Tmax", U.OKABE_ITO[6], "o"),
              (SRC_ALT, "t_max", "NASA Tmax", U.OKABE_ITO[5], "s"),
              (SRC_IDEAM, "heat_index", "IDEAM heat index (2015-19)", U.OKABE_ITO[1], "^"),
              (SRC_ALT, "heat_index", "NASA heat index (2015-25)", U.OKABE_ITO[3], "D")]
    for k, (s, e, lab, col, mk) in enumerate(combos):
        x = dl[(dl.source == s) & (dl.exposure == e)].set_index("outcome").loc[outs]
        yy = np.arange(len(outs)) + (k - 1.5) * 0.18
        ax.errorbar(x.RR_p99, yy, xerr=[x.RR_p99 - x.RR_p99_low, x.RR_p99_high - x.RR_p99], fmt=mk, color=col,
                    ms=3.2, lw=0.8, capsize=1.5, label=lab)
    ax.axvline(1, color="0.4", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xticks([0.25, 0.5, 1, 2, 4], ["0.25", "0.5", "1", "2", "4"])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_yticks(range(len(outs)), [short.get(o, o) for o in outs], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Cumulative RR, P99 vs. P50 (lags 0-7)")
    ax.legend(fontsize=5.6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    # (d) GLM día de lluvia por fuente
    ax = axes[1, 1]
    r = g[g.effect.str.startswith("Rain")]
    for k, (s, col, mk) in enumerate(((SRC_IDEAM, U.OKABE_ITO[6], "o"), (SRC_ALT, U.OKABE_ITO[5], "s"))):
        x = r[r.source == s].set_index("series").loc[U.SERIES]
        yy = np.arange(len(U.SERIES)) + (k - 0.5) * 0.3
        ax.errorbar(x.rr, yy, xerr=[x.rr - x.rr_low, x.rr_high - x.rr], fmt=mk, color=col, ms=3.2, lw=0.8,
                    capsize=1.5, label=("IDEAM rain day" if s == SRC_IDEAM else "CHIRPS rain day"))
    ax.axvline(1, color="0.4", lw=0.8, ls="--")
    ax.set_yticks(range(len(U.SERIES)), [U.SERIES_LABEL[s] for s in U.SERIES], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Rate ratio, rain day vs. dry day (GLM)")
    ax.legend(fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    for ax, lab in zip(axes.flat, "abcd"):
        U.panel_label(ax, lab, x=-0.22 if lab in "ab" else -0.5)
    U.save_figure(fig, "fig_11_climate_sensitivity")


def main():
    U.set_style()
    p, folds, _ = M.load_inputs()
    a = alt_climate()
    a.to_csv(U.RESULTS / "climate_alt_daily.csv")
    assert a.loc[U.TRAIN_START:U.TEST_END].isna().sum().sum() == 0, "NASA/CHIRPS con huecos"
    p_alt = swap_climate(p, a)
    agr = agreement(p, a)
    print(agr.round(3).to_string(index=False))
    ft = forecast(p, p_alt, M.test_folds(folds))
    print(ft[["Configuration", "sCRPS_mean", "delta_vs_no_climate", "delta_ci_low", "delta_ci_high", "p_holm"]]
          .round(5).to_string(index=False))
    g = glm_effects(p, p_alt)
    dl = dlnm(D.load(), a)
    table_13(g, dl)
    fig_11(p, a, ft, g, dl)
    print(dl[["source", "exposure", "outcome", "n_days", "RR_p99", "RR_p99_low", "RR_p99_high", "wald_p_holm"]]
          .round(3).to_string(index=False))
    print(g[["source", "effect", "series", "rr", "rr_low", "rr_high", "p_holm_src"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
