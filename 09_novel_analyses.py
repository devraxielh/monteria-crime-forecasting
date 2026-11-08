from __future__ import annotations

import importlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
U = importlib.import_module("00_config_and_utils")

QCOLS = [f"q{t:.1f}" for t in U.QUANTILES]
SKILL_MODELS = ["GLM-NB", "LGBM-NB", "Chronos-2", U.PROPOSED]
ALERT_MODELS = ["SN-Poisson", "GLM-NB", "GLM-ZINB", "LGBM-NB", "Chronos-2", "Chronos-2-FT", "TimesFM-2.5", "TiRex",
                "Stacked"]
GAMMAS = [0.001, 0.005, 0.01, 0.02]
BASE = "Stacked"  # ACI se aplica a la combinación propuesta antes de calibrar
GAMMA_MAIN = 0.005
TARGET = 0.8
ROLL = 91  # días de la ventana móvil para cobertura condicional en el tiempo


# ------------------------------------------------------------------ (1) pronosticabilidad
def series_profile(p: pd.DataFrame) -> pd.DataFrame:
    rows = []
    tr = p[(p.date >= U.TRAIN_START) & (p.date < U.TEST_START)]
    for s in U.SERIES:
        y = tr[tr.series == s].sort_values("date").y.values
        nz = y[y > 0]
        idx = np.flatnonzero(y > 0)
        adi = float(np.mean(np.diff(idx))) if len(idx) > 1 else np.nan
        cv2 = float(np.var(nz) / np.mean(nz) ** 2) if len(nz) > 1 else np.nan
        cls = ("Smooth" if adi < 1.32 and cv2 < 0.49 else "Erratic" if adi < 1.32
               else "Intermittent" if cv2 < 0.49 else "Lumpy")
        rows.append({"series": s, "mean_day": y.mean(), "zero_pct": 100 * np.mean(y == 0),
                     "ADI": adi, "CV2": cv2, "SBC_class": cls})
    return pd.DataFrame(rows)


def skill_table(scores: pd.DataFrame, prof: pd.DataFrame) -> pd.DataFrame:
    base = scores[scores.model == "SN-Poisson"].set_index(["fold", "series"]).CRPS
    out = prof.copy()
    for m in SKILL_MODELS:
        c = scores[scores.model == m].set_index(["fold", "series"]).CRPS
        sk = (1 - c / base.reindex(c.index)).groupby("series").mean() * 100
        out[f"skill_{m}"] = out.series.map(sk)
    return out


def table_09(t: pd.DataFrame) -> dict:
    corr = {}
    for m in SKILL_MODELS:
        for v in ("mean_day", "zero_pct"):
            r = stats.spearmanr(t[v], t[f"skill_{m}"])
            corr[(m, v)] = (r.statistic, r.pvalue)
    t.to_csv(U.RESULTS / "forecastability.csv", index=False)
    tex = pd.DataFrame({"Crime": t.series.map(U.SERIES_LABEL), "Mean/day": t.mean_day.map(lambda v: f"{v:.2f}"),
                        "Zero days (\\%)": t.zero_pct.map(lambda v: f"{v:.1f}"),
                        "ADI": t.ADI.map(lambda v: f"{v:.2f}"), "CV$^2$": t.CV2.map(lambda v: f"{v:.2f}"),
                        "SBC class": t.SBC_class})
    for m in SKILL_MODELS:
        tex[f"Skill {m} (\\%)"] = t[f"skill_{m}"].map(lambda v: f"{v:.1f}")
    cr = pd.DataFrame([{"model": m, "characteristic": v, "spearman_rho": r, "p": pv}
                       for (m, v), (r, pv) in corr.items()])
    cr.to_csv(U.RESULTS / "forecastability_spearman.csv", index=False)
    out = t.copy()
    U.save_table(out, "table_09_forecastability", tex_df=tex, column_format="lrrrrl" + "r" * len(SKILL_MODELS))
    f = U.TABLES / "table_09_forecastability.tex"
    f.write_text(f.read_text(encoding="utf-8").replace(r"CV\$^2\$", "CV$^2$").replace(r"(\textbackslash{}\%)", r"(\%)")
                 .replace(r"(\\\%)", r"(\%)"), encoding="utf-8")
    return corr


# ------------------------------------------------------------------ (2) conformal adaptativo
def aci_series(d: pd.DataFrame, gamma: float) -> pd.DataFrame:
    """d: predicciones de la combinación base de una serie (2022-2025) ordenadas por fecha, con columna 'origin'."""
    lo0, hi0, y = d["q0.1"].values, d["q0.9"].values, d.y.values
    score = np.maximum(lo0 - y, y - hi0)
    alpha_t = 1 - TARGET
    lo, hi, alph = np.full(len(d), np.nan), np.full(len(d), np.nan), np.full(len(d), np.nan)
    past = np.zeros(len(d), bool)
    for o in np.unique(d.origin.values):
        m = (d.origin.values == o)
        hist = score[past]
        if len(hist) >= 30:
            lvl = np.clip(1 - alpha_t, 0.0, 1.0)
            Q = np.quantile(hist, lvl, method="higher") if 0 < lvl < 1 else (hist.max() if lvl >= 1 else hist.min())
        else:
            Q = 0.0
        lo[m], hi[m], alph[m] = np.maximum(lo0[m] - Q, 0), hi0[m] + Q, alpha_t
        err = ~((y[m] >= lo[m]) & (y[m] <= hi[m]))
        for e in err:  # actualización en lote: se aplica antes del siguiente origen
            alpha_t = alpha_t + gamma * ((1 - TARGET) - float(e))
        past |= m
    return pd.DataFrame({"lo": lo, "hi": hi, "alpha_t": alph}, index=d.index)


def static_cqr_series(d: pd.DataFrame) -> pd.DataFrame:
    """CQR simétrico estático (misma función de puntaje, alpha fijo) para aislar el efecto de la adaptación."""
    return aci_series(d, 0.0)


def interval_metrics(y, lo, hi, alpha=1 - TARGET) -> dict:
    cov = (y >= lo) & (y <= hi)
    width = hi - lo
    iscore = width + (2 / alpha) * (lo - y) * (y < lo) + (2 / alpha) * (y - hi) * (y > hi)
    return {"coverage": cov.mean(), "width": width.mean(), "interval_score": iscore.mean()}


def rolling_cov(df: pd.DataFrame, lo: str, hi: str) -> pd.Series:
    c = ((df.y >= df[lo]) & (df.y <= df[hi])).astype(float)
    return c.rolling(ROLL, min_periods=ROLL).mean()


def run_aci(preds_all: pd.DataFrame, origins: pd.DataFrame, folds: pd.DataFrame):
    ens = preds_all[preds_all.model == BASE].merge(origins[["fold", "date", "origin"]], on=["fold", "date"])
    cqr = preds_all[preds_all.model == f"{BASE}+CQR"][["series", "date", "q0.1", "q0.9"]]
    cqr = cqr.rename(columns={"q0.1": "lo_cqr", "q0.9": "hi_cqr"})
    test_start = folds[folds.role == "test"].test_start.min()
    res, rows = [], []
    for s in U.SERIES:
        d = ens[ens.series == s].sort_values("date").reset_index(drop=True)
        d = d.merge(cqr[cqr.series == s].drop(columns="series"), on="date", how="left")
        d["lo_raw"], d["hi_raw"] = d["q0.1"], d["q0.9"]
        st = static_cqr_series(d)
        d["lo_scqr"], d["hi_scqr"] = st.lo, st.hi
        for g in GAMMAS:
            a = aci_series(d, g)
            d[f"lo_aci{g}"], d[f"hi_aci{g}"], d[f"alpha_aci{g}"] = a.lo, a.hi, a.alpha_t
        d = d[d.date >= test_start]
        res.append(d)
    r = pd.concat(res, ignore_index=True)
    methods = {f"{BASE} (uncalibrated)": ("lo_raw", "hi_raw"), f"{BASE}+CQR (per-level, static)": ("lo_cqr", "hi_cqr"),
               f"{BASE}+CQR (symmetric, static)": ("lo_scqr", "hi_scqr")}
    methods.update({f"{BASE}+ACI ($\\gamma$={g})": (f"lo_aci{g}", f"hi_aci{g}") for g in GAMMAS})
    for subset, ss in (("All series", U.SERIES), ("Regime-shift series", U.SHIFTED_SERIES),
                       ("Other series", [s for s in U.SERIES if s not in U.SHIFTED_SERIES])):
        sub = r[r.series.isin(ss)]
        for name, (lo, hi) in methods.items():
            m = interval_metrics(sub.y.values, sub[lo].values, sub[hi].values)
            rc = pd.concat([rolling_cov(g.sort_values("date"), lo, hi) for _, g in sub.groupby("series")]).dropna()
            m.update({"subset": subset, "method": name, "abs_cov_err": abs(m["coverage"] - TARGET),
                      "pct_windows_cov_lt_0.7": 100 * np.mean(rc < 0.7),
                      "min_rolling_cov": rc.min()})
            rows.append(m)
    t = pd.DataFrame(rows)[["subset", "method", "coverage", "abs_cov_err", "width", "interval_score",
                            "pct_windows_cov_lt_0.7", "min_rolling_cov"]]
    r.to_parquet(U.RESULTS / "aci_intervals.parquet", index=False)
    tex = t.copy()
    tex["subset"] = tex.subset.where(~tex.subset.duplicated(), "")
    for c, nd in (("coverage", 3), ("abs_cov_err", 3), ("width", 2), ("interval_score", 2),
                  ("pct_windows_cov_lt_0.7", 1), ("min_rolling_cov", 2)):
        tex[c] = t[c].map(lambda v, nd=nd: f"{v:.{nd}f}")
    tex.columns = ["Subset", "Method", "Coverage80", "$|$Cov.~error$|$", "Width80", "Interval score",
                   "Windows cov.$<$0.7 (\\%)", "Min. 91-d cov."]
    U.save_table(t, "table_10_adaptive_conformal", tex_df=tex, column_format="llrrrrrr")
    f = U.TABLES / "table_10_adaptive_conformal.tex"
    txt = f.read_text(encoding="utf-8")
    for a, b in ((r"\$\textbackslash{}gamma\$", r"$\gamma$"), (r"\$\textbackslash gamma\$", r"$\gamma$"),
                 (r"\$|\$Cov.~error\$|\$", r"$|$Cov.~error$|$"), (r"\$<\$", r"$<$")):
        txt = txt.replace(a, b)
    f.write_text(txt, encoding="utf-8")
    return r, t


# ------------------------------------------------------------------ (3) alertas
def thresholds(p: pd.DataFrame) -> dict:
    tr = p[(p.date >= U.TRAIN_START) & (p.date < U.CAL_START)]
    T = {}
    for s in U.SERIES:
        y = tr[tr.series == s].y.values
        T[s] = next(k for k in range(1, int(y.max()) + 2) if np.mean(y >= k) <= 0.10)
    return T


def run_alerts(preds: pd.DataFrame, T: dict, n_boot: int = U.N_BOOT):
    rows = []
    for m in ALERT_MODELS:
        d = preds[preds.model == m]
        for (f, s), g in d.groupby(["fold", "series"]):
            ev = (g.y.values >= T[s]).astype(int)
            k = max(1, int(round(0.10 * len(g))))
            order = np.lexsort((-g["q0.9"].values, -g["mean"].values))  # desempate por q0.9
            alert = np.zeros(len(g), bool)
            alert[order[:k]] = True
            tp = int((alert & (ev == 1)).sum())
            rows.append({"model": m, "fold": f, "series": s, "n_events": int(ev.sum()), "prevalence": ev.mean(),
                         "precision10": tp / k, "recall10": tp / ev.sum() if ev.sum() else np.nan,
                         "lift10": (tp / k) / ev.mean() if ev.mean() > 0 else np.nan,
                         "auc_pr": average_precision_score(ev, g["mean"].values) if 0 < ev.sum() < len(ev) else np.nan})
    a = pd.DataFrame(rows)
    a.to_csv(U.RESULTS / "alerts_by_fold_series.csv", index=False)
    summ = []
    for m in ALERT_MODELS:
        g = a[a.model == m]
        row = {"model": m}
        for c in ("auc_pr", "precision10", "recall10", "lift10"):
            v = g[c].dropna().values
            lo, hi = U.bootstrap_ci(v)
            row.update({f"{c}_mean": v.mean(), f"{c}_ci_low": lo, f"{c}_ci_high": hi})
        summ.append(row)
    t = pd.DataFrame(summ)
    # Wilcoxon pareado de lift: cada modelo vs SN-Poisson (Holm)
    from statsmodels.stats.multitest import multipletests
    base = a[a.model == "SN-Poisson"].set_index(["fold", "series"]).auc_pr
    pv = []
    for m in ALERT_MODELS[1:]:
        x = a[a.model == m].set_index(["fold", "series"]).auc_pr
        j = pd.concat([x, base], axis=1, keys=["m", "b"]).dropna()
        pv.append(stats.wilcoxon(j.m, j.b).pvalue)
    t["p_auc_pr_vs_SN_holm"] = [np.nan] + list(multipletests(pv, method="holm")[1])
    best = {c: t[f"{c}_mean"].max() for c in ("auc_pr", "precision10", "recall10", "lift10")}

    def cell(r, c):
        s = f"{r[f'{c}_mean']:.3f} [{r[f'{c}_ci_low']:.3f}, {r[f'{c}_ci_high']:.3f}]"
        return rf"\textbf{{{s}}}" if np.isclose(r[f"{c}_mean"], best[c]) else s

    tex = pd.DataFrame({"Model": t.model,
                        "AUC-PR": [cell(r, "auc_pr") for _, r in t.iterrows()],
                        "Precision@10\\%": [cell(r, "precision10") for _, r in t.iterrows()],
                        "Recall@10\\%": [cell(r, "recall10") for _, r in t.iterrows()],
                        "Lift@10\\%": [cell(r, "lift10") for _, r in t.iterrows()],
                        "HOLMP AUC-PR vs SN": t.p_auc_pr_vs_SN_holm.map(lambda v: "--" if np.isnan(v) else f"{v:.1e}")})
    U.save_table(t, "table_11_alerts", tex_df=tex, column_format="lccccr")
    f = U.TABLES / "table_11_alerts.tex"
    txt = f.read_text(encoding="utf-8").replace("HOLMP", r"$p_{\mathrm{Holm}}$")
    txt = txt.replace(r"\textbackslash{}textbf\{", r"\textbf{").replace(r"\textbackslash{}\%", r"\%")
    f.write_text(txt, encoding="utf-8")
    return a, t


# ------------------------------------------------------------------ figura
def fig_10(skill: pd.DataFrame, corr: dict, aci: pd.DataFrame, alerts: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.4), gridspec_kw={"hspace": 0.45, "wspace": 0.32})
    # (a) habilidad vs volumen
    ax = axes[0, 0]
    for m in SKILL_MODELS:
        ax.scatter(skill.mean_day, skill[f"skill_{m}"], s=22, marker=U.MODEL_MARKERS[m], color=U.MODEL_COLORS[m],
                   edgecolor="k", linewidth=0.3, label=f"{m} ($\\rho$={corr[(m, 'mean_day')][0]:.2f})", zorder=3)
    ax.axhline(0, color="0.4", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xticks([0.2, 0.5, 1, 2, 4], ["0.2", "0.5", "1", "2", "4"])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_ylim(-4.5, 17)
    ax.set_xlabel("Mean daily count (training, log scale)")
    ax.set_ylabel("CRPS skill vs. SN-Poisson (%)")
    ax.legend(fontsize=5.6, frameon=False, loc="lower center", ncol=2, columnspacing=0.6, handletextpad=0.2)
    # (b) Syntetos-Boylan
    ax = axes[0, 1]
    v = skill[f"skill_{U.PROPOSED}"]
    sc = ax.scatter(skill.ADI, skill.CV2, c=v, cmap="viridis", s=36, edgecolor="k", linewidth=0.3, zorder=3)
    ax.axvline(1.32, color="0.4", lw=0.8, ls="--")
    ax.axhline(0.49, color="0.4", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xticks([1, 1.32, 2, 4, 8], ["1", "1.32", "2", "4", "8"])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlim(0.95, 9)
    ax.set_ylim(0.05, 0.62)
    offs = {"hurto_personas": (-2, -9), "hurto_comercio": (-6, 5), "delitos_informaticos": (4, 2),
            "lesiones_personales": (4, 1), "violencia_intrafamiliar": (4, -7), "homicidio_transito": (-30, 5),
            "extorsion": (-14, -11), "homicidio": (4, 2)}
    for _, r in skill.iterrows():
        ax.annotate(U.SERIES_LABEL[r.series], (r.ADI, r.CV2), fontsize=5.2,
                    xytext=offs.get(r.series, (4, 2)), textcoords="offset points")
    for x, y, txt, ha in ((0.97, 0.60, "Erratic", "left"), (8.6, 0.60, "Lumpy", "right"),
                          (0.97, 0.07, "Smooth", "left"), (8.6, 0.33, "Intermittent", "right")):
        ax.text(x, y, txt, ha=ha, va="center", fontsize=6.5, color="0.35", style="italic")
    ax.set_xlabel("ADI (days between non-zero days, log scale)")
    ax.set_ylabel(r"CV$^2$ of non-zero counts")
    cb = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.02)
    cb.set_label(f"{U.PROPOSED} skill (%)", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    # (c) cobertura móvil, delitos sexuales
    ax = axes[1, 0]
    d = aci[aci.series == "delitos_sexuales"].sort_values("date")
    for (lo, hi), lab, col, ls in ((("lo_raw", "hi_raw"), BASE, U.MODEL_COLORS[BASE], ":"),
                                   (("lo_cqr", "hi_cqr"), f"{BASE}+CQR", U.MODEL_COLORS[f"{BASE}+CQR"], "--"),
                                   ((f"lo_aci{GAMMA_MAIN}", f"hi_aci{GAMMA_MAIN}"), f"{BASE}+ACI ($\\gamma$={GAMMA_MAIN})",
                                    U.OKABE_ITO[6], "-")):
        ax.plot(d.date, rolling_cov(d, lo, hi), color=col, ls=ls, lw=1.1, label=lab)
    ax.axhline(TARGET, color="k", lw=0.8)
    ax.set_ylabel(f"{ROLL}-day rolling coverage")
    ax.set_xlabel("Date (sexual offences, test period)")
    ax.set_ylim(0.4, 1.02)
    ax.legend(fontsize=6, frameon=False, loc="lower left")
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", labelsize=7)
    # (d) lift a presupuesto 10 %
    ax = axes[1, 1]
    g = alerts.groupby("model").lift10.agg(["mean", lambda x: U.bootstrap_ci(x.dropna().values)])
    g = g.reindex(ALERT_MODELS)
    means = g["mean"].values
    ci = np.array([list(c) for c in g.iloc[:, 1].values])
    y = np.arange(len(ALERT_MODELS))
    ax.barh(y, means, color=[U.MODEL_COLORS[m] for m in ALERT_MODELS], edgecolor="k", linewidth=0.4)
    ax.errorbar(means, y, xerr=[means - ci[:, 0], ci[:, 1] - means], fmt="none", ecolor="k", lw=0.8, capsize=2)
    ax.axvline(1, color="0.4", lw=0.8, ls="--")
    ax.set_yticks(y, ALERT_MODELS, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Lift of high-crime-day alerts\n(precision@10% / prevalence)")
    for ax, lab in zip(axes.flat, "abcd"):
        U.panel_label(ax, lab, x=-0.2 if lab != "d" else -0.42)
    U.save_figure(fig, "fig_10_novel_analyses")


def main():
    U.set_style()
    p = pd.read_parquet(U.RESULTS / "panel_features.parquet")
    scores = pd.read_csv(U.RESULTS / "fold_scores.csv")
    preds_all = pd.read_parquet(U.RESULTS / "oof_predictions_all_folds.parquet")
    preds = pd.read_parquet(U.RESULTS / "oof_predictions.parquet")
    folds = pd.read_csv(U.RESULTS / "folds.csv", parse_dates=["train_end", "test_start", "test_end"])
    origins = pd.read_csv(U.RESULTS / "origins.csv", parse_dates=["origin", "date"])
    prof = series_profile(p)
    skill = skill_table(scores, prof)
    corr = table_09(skill)
    aci, t10 = run_aci(preds_all, origins, folds)
    T = thresholds(p)
    pd.Series(T).to_csv(U.RESULTS / "alert_thresholds.csv")
    alerts, t11 = run_alerts(preds, T)
    fig_10(skill, corr, aci, alerts)
    pd.set_option("display.width", 200)
    print(skill.round(2).to_string(index=False))
    print({k: (round(a, 3), round(b, 4)) for k, (a, b) in corr.items()})
    print(t10.round(3).to_string(index=False))
    print("thresholds", T)
    print(t11[["model", "auc_pr_mean", "precision10_mean", "recall10_mean", "lift10_mean", "p_auc_pr_vs_SN_holm"]]
          .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
