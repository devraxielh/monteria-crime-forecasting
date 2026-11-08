from __future__ import annotations

import importlib
import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
U = importlib.import_module("00_config_and_utils")
P = importlib.import_module("01_data_splitting_and_prep")
M = importlib.import_module("02_baseline_and_sota_models")

GROUP_OF = {f: g for g, fs in P.FEATURE_GROUPS.items() for f in fs}
GROUP_OF["series_id"] = "Series ID"
GROUP_COLORS = {"Lags": U.OKABE_ITO[5], "Calendar": U.OKABE_ITO[1], "Holidays": U.OKABE_ITO[7],
                "Payday": U.OKABE_ITO[3], "Climate": U.OKABE_ITO[6], "Series ID": "0.6"}
NICE = {"t_max": r"$T_{max}$", "t_media": r"$T_{mean}$", "t_min": r"$T_{min}$", "precip_mm": "Precipitation",
        "rain_day": "Rain day", "precip_rm7": "Precip. (7-d mean)", "tmax_rm7": r"$T_{max}$ (7-d mean)",
        "rh_media": "Relative humidity", "heat_index": "Heat index",
        "festivo": "Public holiday", "puente": "Long weekend", "vispera_festivo": "Holiday eve",
        "temporada_decembrina": "December season", "semana_santa": "Holy Week", "quincena": "Payday",
        "weekend": "Weekend", "dow": "Day of week", "month": "Month", "dom": "Day of month",
        "trend": "Trend", "covid_lockdown": "COVID lockdown", "series_id": "Series ID",
        "sn4": "Same-weekday mean (4 wk)", "total_rm28": "All-crime 28-d mean", "zero28": "Zero share (28 d)",
        "rsd28": "SD (28 d)", "doy_s1": "Annual sin", "doy_c1": "Annual cos", "doy_s2": "Semiannual sin",
        "doy_c2": "Semiannual cos"}


def nice(f):
    return NICE.get(f, f.replace("lag", "Lag ").replace("rm", "Mean ") + (" d" if f.startswith("rm") else ""))


def compute_shap(p, folds) -> pd.DataFrame:
    out = []
    for _, f in folds.iterrows():
        booster = lgb.Booster(model_file=str(M.MODELS_DIR / f"lgbm_full_fold{f.fold}.txt"))
        _, te = M.split(p, f)
        X = M._prep_X(te, P.ALL_FEATURES)  # noqa: SLF001
        contrib = booster.predict(X, pred_contrib=True)
        d = pd.DataFrame(contrib[:, :-1], columns=P.ALL_FEATURES)
        d["base"] = contrib[:, -1]
        d["series"], d["date"], d["fold"] = te.series.values, te.date.values, f.fold
        vals = te[P.ALL_FEATURES].reset_index(drop=True).add_prefix("x_")
        out.append(pd.concat([d, vals], axis=1))
    return pd.concat(out, ignore_index=True)


def fig_04(sh: pd.DataFrame) -> None:
    feats = P.ALL_FEATURES
    imp = sh[feats].abs().mean().sort_values(ascending=False)
    fig = plt.figure(figsize=(U.COL2, 6.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.45, wspace=0.55)
    # (a) importancia global top-15
    ax = fig.add_subplot(gs[0, 0])
    top = imp.head(15)[::-1]
    ax.barh(range(len(top)), top.values, color=[GROUP_COLORS[GROUP_OF[f]] for f in top.index], height=0.7)
    ax.set_yticks(range(len(top)), [nice(f) for f in top.index], fontsize=7)
    ax.set_xlabel("Mean |SHAP| (log-mean scale)")
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in GROUP_COLORS.values()]
    ax.legend(handles, GROUP_COLORS.keys(), fontsize=6.5, loc="lower right")
    U.panel_label(ax, "a", x=-0.62)
    # (b) beeswarm de variables exógenas (calendario-clima)
    ax = fig.add_subplot(gs[0, 1])
    exo = ["festivo", "puente", "vispera_festivo", "temporada_decembrina", "semana_santa", "quincena",
           "weekend", "t_max", "t_media", "precip_mm", "precip_rm7", "tmax_rm7"]
    exo = list(imp[exo].sort_values().index)
    rng = np.random.default_rng(U.SEED)
    samp = sh.sample(n=min(len(sh), 8000), random_state=U.SEED)
    for i, f in enumerate(exo):
        x = samp[f].values
        v = samp[f"x_{f}"].astype(float).values
        ok = ~np.isnan(v)
        vmin, vmax = np.nanpercentile(v, 5), np.nanpercentile(v, 95)
        col = np.clip((v - vmin) / (vmax - vmin + 1e-9), 0, 1)
        jit = rng.normal(0, 0.12, len(x))
        ax.scatter(x[~ok], i + jit[~ok], s=1.5, color="0.75", alpha=0.4, lw=0, rasterized=True)
        sc = ax.scatter(x[ok], i + jit[ok], c=col[ok], cmap="viridis", s=1.5, alpha=0.6, lw=0,
                        vmin=0, vmax=1, rasterized=True)
    ax.axvline(0, color="0.3", lw=0.6)
    ax.set_yticks(range(len(exo)), [nice(f) for f in exo], fontsize=7)
    ax.set_xlabel("SHAP value (log-mean scale)")
    ax.grid(axis="y", visible=False)
    cb = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.02)
    cb.set_label("Feature value (low → high)", fontsize=7); cb.set_ticks([])
    U.panel_label(ax, "b", x=-0.55)
    # (c) dependencia de Tmax por grupos de delitos
    ax = fig.add_subplot(gs[1, 0])
    grp = {"Violent (assault, homicide, DV)": ["lesiones_personales", "homicidio", "violencia_intrafamiliar"],
           "Property theft": ["hurto_personas", "hurto_comercio", "hurto_residencias", "hurto_vehiculos"],
           "Other": ["delitos_informaticos", "delitos_sexuales", "homicidio_transito", "extorsion"]}
    bins = np.arange(29, 39.1, 1.0)
    for (lab, ss), mk, c in zip(grp.items(), ["o", "s", "^"], U.OKABE_ITO[5:8]):
        d = sh[sh.series.isin(ss)].dropna(subset=["x_t_max"])
        b = pd.cut(d["x_t_max"], bins)
        m = d.groupby(b, observed=True)["t_max"].agg(["mean", "sem", "count"])
        m = m[m["count"] >= 30]
        xc = [iv.mid for iv in m.index]
        ax.errorbar(xc, m["mean"], yerr=1.96 * m["sem"], marker=mk, ms=3.5, lw=1, capsize=2, color=c, label=lab)
    ax.axhline(0, color="0.3", lw=0.6)
    ax.set_xlabel(r"Daily $T_{max}$ ($^\circ$C)"); ax.set_ylabel(r"Mean SHAP of $T_{max}$")
    ax.legend(fontsize=6.5)
    U.panel_label(ax, "c", x=-0.3)
    # (d) cuota de atribución por grupo y delito
    ax = fig.add_subplot(gs[1, 1])
    share = group_share(sh)
    share = share.loc[U.SERIES]
    left = np.zeros(len(share))
    for g in GROUP_COLORS:
        ax.barh(range(len(share)), share[g].values, left=left, color=GROUP_COLORS[g], height=0.7, label=g)
        left += share[g].values
    ax.set_yticks(range(len(share)), [U.SERIES_LABEL[s] for s in share.index], fontsize=7)
    ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_xlabel("Share of total |SHAP|")
    ax.grid(axis="y", visible=False)
    U.panel_label(ax, "d", x=-0.55)
    U.save_figure(fig, "fig_04_explainability_shap")


def group_share(sh: pd.DataFrame) -> pd.DataFrame:
    a = sh[P.ALL_FEATURES].abs()
    g = a.T.groupby(lambda f: GROUP_OF[f]).sum().T
    g["series"] = sh.series.values
    s = g.groupby("series").sum()
    return s.div(s.sum(axis=1), axis=0)[list(GROUP_COLORS)]


def calibration(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m, g in preds.groupby("model"):
        for t, c in zip(U.QUANTILES, M.QCOLS):
            rows.append({"model": m, "tau": t, "freq_le": float(np.mean(g.y.values <= g[c].values)),
                         "freq_lt": float(np.mean(g.y.values < g[c].values))})
    return pd.DataFrame(rows)


def fig_05(preds, scores, cal, p) -> None:
    fig = plt.figure(figsize=(U.COL2, 5.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.45, wspace=0.35)
    # (a) diagrama de fiabilidad de cuantiles (conteos: banda [P(Y<q), P(Y<=q)])
    ax = fig.add_subplot(gs[0, 0])
    ax.plot([0, 1], [0, 1], color="0.4", lw=0.7, ls="--")
    for m in ["SN-Poisson", "GLM-NB", "Chronos-2", "TimesFM-2.5", "LGBM-NB", U.PROPOSED]:
        mk = U.MODEL_MARKERS[m]
        c = cal[cal.model == m]
        mid = (c.freq_le + c.freq_lt) / 2
        ax.plot(c.tau, mid, marker=mk, ms=3, lw=0.9, color=U.MODEL_COLORS[m], label=m)
        ax.fill_between(c.tau, c.freq_lt, c.freq_le, color=U.MODEL_COLORS[m], alpha=0.12, lw=0)
    ax.set_xlabel(r"Nominal quantile level $\tau$"); ax.set_ylabel("Observed frequency")
    ax.legend(fontsize=6, ncol=2, loc="upper left")
    U.panel_label(ax, "a", x=-0.25)
    # (b) mejora relativa vs. escasez de la serie
    ax = fig.add_subplot(gs[0, 1])
    zero = p[p.date < U.TEST_START].groupby("series").y.apply(lambda s: (s == 0).mean())
    piv = scores.pivot_table(index="series", columns="model", values="sCRPS")
    for m in ["GLM-NB", "Chronos-2", "LGBM-NB", U.PROPOSED]:
        mk = U.MODEL_MARKERS[m]
        rel = 100 * (1 - piv[m] / piv["SN-Poisson"])
        ax.scatter(100 * zero[rel.index], rel, marker=mk, s=18, color=U.MODEL_COLORS[m], label=m)
        z = np.polyfit(100 * zero[rel.index], rel, 1)
        xs = np.linspace(0, 90, 50)
        ax.plot(xs, np.polyval(z, xs), color=U.MODEL_COLORS[m], lw=0.7, alpha=0.7)
    ax.axhline(0, color="0.3", lw=0.6)
    ax.set_xlabel("Zero-count days in training (%)"); ax.set_ylabel("sCRPS improvement over SN-Poisson (%)")
    ax.legend(fontsize=6)
    U.panel_label(ax, "b", x=-0.25)
    # (c) caso de estudio: hurto a personas, último trimestre de prueba
    ax = fig.add_subplot(gs[1, :])
    last = preds.fold.max()
    for m, c, off in [("Chronos-2", U.MODEL_COLORS["Chronos-2"], 0), (U.PROPOSED, U.MODEL_COLORS[U.PROPOSED], 0)]:
        g = preds[(preds.model == m) & (preds.series == "hurto_personas") & (preds.fold == last)].sort_values("date")
        ax.fill_between(g.date, g["q0.1"], g["q0.9"], color=c, alpha=0.18, lw=0, step="mid")
        ax.plot(g.date, g["q0.5"], color=c, lw=1.1, label=f"{m} median / 80% PI", drawstyle="steps-mid")
    obs = g[["date", "y"]]
    ax.plot(obs.date, obs.y, "o", ms=2.5, color="black", label="Observed")
    ax.set_ylabel("Daily thefts (persons)")
    ax.legend(fontsize=6.5, ncol=3, loc="upper left")
    ax.set_ylim(0, max(obs.y.max(), g["q0.9"].max()) * 1.25)
    U.panel_label(ax, "c", x=-0.08)
    U.save_figure(fig, "fig_05_error_calibration_case_study")


# ================================================================== explicabilidad robusta
TEMP_FEATS = ["t_max", "t_media", "t_min", "tmax_rm7"]
RAIN_FEATS = ["precip_mm", "rain_day", "precip_rm7"]
# efecto -> (variables SHAP sumadas, condición "on", condición válida). log RR = E[phi|on] - E[phi|off]
EFFECTS = {
    "Public holiday": (["festivo"], lambda d: d.x_festivo == 1, None),
    "Long weekend": (["puente"], lambda d: d.x_puente == 1, None),
    "Holiday eve": (["vispera_festivo"], lambda d: d.x_vispera_festivo == 1, None),
    "Payday": (["quincena"], lambda d: d.x_quincena == 1, None),
    "December season": (["temporada_decembrina"], lambda d: d.x_temporada_decembrina == 1, None),
    "Holy Week": (["semana_santa"], lambda d: d.x_semana_santa == 1, None),
    "Weekend": (["weekend", "dow"], lambda d: d.x_weekend == 1, None),
    r"Hot day ($T_{max}\geq$35 °C)": (TEMP_FEATS, lambda d: d.x_t_max >= 35, lambda d: d.x_t_max.notna()),
    "Rain day (≥1 mm)": (RAIN_FEATS, lambda d: d.x_rain_day == 1, lambda d: d.x_rain_day.notna()),
}
HOT = 35.0
N_BOOT_RR = 1000


def fold_stability(sh: pd.DataFrame) -> pd.DataFrame:
    a = sh[P.ALL_FEATURES].abs()
    g = a.T.groupby(lambda f: GROUP_OF[f]).sum().T
    g["fold"] = sh.fold.values
    s = g.groupby("fold").sum()
    return s.div(s.sum(axis=1), axis=0)[list(GROUP_COLORS)]


def effect_rr(sh: pd.DataFrame) -> pd.DataFrame:
    """Razón de tasas implícita por SHAP, por delito, con IC 95 % por bootstrap de clústeres = folds.

    Dentro de un modelo el SHAP de una variable binaria es casi constante, por lo que un bootstrap de
    filas/semanas subestima la incertidumbre. Remuestrear los 12 folds (12 modelos reentrenados sobre
    trimestres distintos) captura la variabilidad de estimación del modelo y entre periodos.
    """
    rng = np.random.default_rng(U.SEED)
    sh = sh.assign(week=sh.fold.astype(str))
    weeks = np.array(sorted(sh.week.unique()))
    wpos = {w: i for i, w in enumerate(weeks)}
    W = rng.multinomial(len(weeks), np.full(len(weeks), 1 / len(weeks)), size=N_BOOT_RR)  # pesos (B, nFolds)
    rows = []
    for s in U.SERIES:
        d = sh[sh.series == s]
        wi = d.week.map(wpos).values
        for eff, (feats, on_fn, valid_fn) in EFFECTS.items():
            phi = d[feats].sum(axis=1).values
            valid = np.ones(len(d), bool) if valid_fn is None else valid_fn(d).values
            on = on_fn(d).fillna(False).values & valid
            off = (~on) & valid
            if on.sum() < 5 or off.sum() < 5:
                continue
            est = phi[on].mean() - phi[off].mean()
            # sumas y conteos por semana, luego remuestreo de semanas
            def agg(mask):
                sm = np.bincount(wi[mask], weights=phi[mask], minlength=len(weeks))
                ct = np.bincount(wi[mask], minlength=len(weeks)).astype(float)
                return sm, ct
            s_on, c_on = agg(on); s_off, c_off = agg(off)
            boots = (W @ s_on) / np.maximum(W @ c_on, 1e-9) - (W @ s_off) / np.maximum(W @ c_off, 1e-9)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            rows.append({"series": s, "effect": eff, "n_on": int(on.sum()), "log_rr": est, "rr": np.exp(est),
                         "rr_ci_low": np.exp(lo), "rr_ci_high": np.exp(hi),
                         "significant": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(rows)


def glm_convergent(p: pd.DataFrame) -> pd.DataFrame:
    """GLM Poisson interpretable por delito (2015-06 a 2025-12, EE robustas HC0) con las mismas definiciones."""
    import statsmodels.api as sm

    rows = []
    for s in U.SERIES:
        d = p[p.series == s].copy()
        X = pd.DataFrame(index=d.index)
        for c in ["festivo", "puente", "vispera_festivo", "quincena", "temporada_decembrina", "semana_santa",
                  "weekend", "covid_lockdown"]:
            X[c] = d[c].astype(float)
        X["hot_day"] = (d.t_max >= HOT).astype(float)
        X["tmax_missing"] = d.t_max.isna().astype(float)
        X["rain_day"] = d.rain_day.fillna(0.0)
        X["rain_missing"] = d.rain_day.isna().astype(float)
        X["log_rm28"] = np.log1p(d.rm28.fillna(d.rm28.median()))
        X["log_sn4"] = np.log1p(d.sn4.fillna(d.sn4.median()))
        X["trend"] = d.trend
        X = X.join(pd.get_dummies(d.month, prefix="m", drop_first=True).astype(float))
        X = X.drop(columns=[c for c in ("tmax_missing", "rain_missing") if X[c].nunique() < 2])  # fuente completa
        X = sm.add_constant(X)
        res = sm.GLM(d.y.values, X, family=sm.families.Poisson()).fit(cov_type="HC0")
        mapping = {"Public holiday": "festivo", "Long weekend": "puente", "Holiday eve": "vispera_festivo",
                   "Payday": "quincena", "December season": "temporada_decembrina", "Holy Week": "semana_santa",
                   "Weekend": "weekend", r"Hot day ($T_{max}\geq$35 °C)": "hot_day", "Rain day (≥1 mm)": "rain_day"}
        ci = res.conf_int()
        for eff, col in mapping.items():
            rows.append({"series": s, "effect": eff, "glm_log_rr": res.params[col],
                         "glm_ci_low": ci.loc[col, 0], "glm_ci_high": ci.loc[col, 1], "glm_p": res.pvalues[col]})
    out = pd.DataFrame(rows)
    # Holm-Bonferroni sobre la familia completa (9 efectos x 11 delitos)
    from statsmodels.stats.multitest import multipletests
    out["glm_p_holm"] = multipletests(out.glm_p.values, method="holm")[1]
    return out


def interaction_values(p: pd.DataFrame, folds: pd.DataFrame, n_per_fold: int = 220) -> pd.DataFrame:
    """Valores de interacción SHAP (TreeExplainer) en una muestra estratificada de filas de test por fold."""
    import shap

    out = []
    feats = P.ALL_FEATURES
    it, iw, idow = feats.index("t_max"), feats.index("weekend"), feats.index("dow")
    for _, f in folds.iterrows():
        booster = lgb.Booster(model_file=str(M.MODELS_DIR / f"lgbm_full_fold{f.fold}.txt"))
        _, te = M.split(p, f)
        te = te[te.t_max.notna()]
        smp = te.groupby("series").sample(n=n_per_fold // len(U.SERIES), random_state=U.SEED)
        X = M._prep_X(smp, feats)  # noqa: SLF001
        iv = shap.TreeExplainer(booster).shap_interaction_values(X)
        mabs = np.abs(iv).mean(axis=0)
        pair = 2 * (iv[:, it, iw] + iv[:, it, idow])  # t_max x (fin de semana / día de semana)
        out.append(pd.DataFrame({"fold": f.fold, "series": smp.series.values, "t_max": smp.t_max.values,
                                 "weekend": smp.weekend.values, "interaction": pair}))
        if f.fold == folds.fold.max():
            np.save(U.RESULTS / "shap_interaction_mean_abs_lastfold.npy", mabs)
    return pd.concat(out, ignore_index=True)


def fig_06(stab: pd.DataFrame, rr: pd.DataFrame, conv: pd.DataFrame, inter: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(U.COL2, 6.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.5, wspace=0.6)
    # (a) estabilidad de la atribución por grupo entre folds
    ax = fig.add_subplot(gs[0, 0])
    for i, g in enumerate(stab.columns):
        v = stab[g].values
        lo, hi = U.bootstrap_ci(v)
        ax.scatter(v, np.full(len(v), i) + np.random.default_rng(i).normal(0, 0.06, len(v)),
                   s=8, color=GROUP_COLORS[g], alpha=0.7, lw=0)
        ax.errorbar(v.mean(), i, xerr=[[v.mean() - lo], [hi - v.mean()]], fmt="D", ms=4, color="black",
                    capsize=2, lw=1)
    ax.set_yticks(range(len(stab.columns)), stab.columns, fontsize=7.5); ax.invert_yaxis()
    ax.set_xlabel("Share of total |SHAP| per fold"); ax.grid(axis="y", visible=False)
    U.panel_label(ax, "a", x=-0.4)
    # (b) heatmap de RR por delito y efecto
    ax = fig.add_subplot(gs[0, 1])
    effs = list(EFFECTS)
    M_ = rr.pivot(index="series", columns="effect", values="rr").reindex(index=U.SERIES, columns=effs)
    S_ = rr.pivot(index="series", columns="effect", values="convergent").reindex(index=U.SERIES, columns=effs)
    lg = np.log2(M_.astype(float))
    lim = np.nanmax(np.abs(lg.values))
    im = ax.imshow(lg.values, cmap="PuOr_r", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(lg.shape[0]):
        for j in range(lg.shape[1]):
            v = M_.iloc[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}{'*' if S_.iloc[i, j] else ''}", ha="center", va="center", fontsize=5,
                        color="white" if abs(lg.iloc[i, j]) > 0.6 * lim else "black")
    ax.set_xticks(range(len(effs)), [e.replace(r"($T_{max}\geq$35 °C)", "").replace(" (≥1 mm)", "") for e in effs],
                  rotation=50, ha="right", fontsize=6.5)
    ax.set_yticks(range(len(U.SERIES)), [U.SERIES_LABEL[s] for s in U.SERIES], fontsize=6.5)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(r"$\log_2$ rate ratio", fontsize=7); cb.ax.tick_params(labelsize=6)
    U.panel_label(ax, "b", x=-0.5)
    # (c) convergencia SHAP vs GLM
    ax = fig.add_subplot(gs[1, 0])
    c = conv.dropna(subset=["log_rr", "glm_log_rr"])
    mk = dict(zip(list(EFFECTS), ["o", "s", "^", "D", "v", "P", "X", "*", "h"]))
    for eff, g in c.groupby("effect"):
        ax.scatter(g.glm_log_rr, g.log_rr, marker=mk[eff], s=16, alpha=0.8, lw=0.3, edgecolor="black",
                   label=eff.replace(r"($T_{max}\geq$35 °C)", "").replace(" (≥1 mm)", ""))
    lim2 = np.nanmax(np.abs(c[["glm_log_rr", "log_rr"]].values)) * 1.08
    ax.plot([-lim2, lim2], [-lim2, lim2], color="0.4", lw=0.7, ls="--")
    ax.axhline(0, color="0.7", lw=0.5); ax.axvline(0, color="0.7", lw=0.5)
    ax.set_xlim(-lim2, lim2); ax.set_ylim(-lim2, lim2)
    ax.set_xlabel("log RR, interpretable Poisson GLM"); ax.set_ylabel("log RR, LGBM-NB SHAP (test folds)")
    ax.legend(fontsize=5.5, ncol=2, loc="upper left", handletextpad=0.2, columnspacing=0.6)
    U.panel_label(ax, "c", x=-0.3)
    # (d) interacción Tmax x fin de semana (delitos violentos y contra la propiedad)
    ax = fig.add_subplot(gs[1, 1])
    bins = np.arange(29, 39.1, 1.0)
    for (wk, lab), mk2, col in zip([(0, "Weekday"), (1, "Weekend")], ["o", "s"], [U.OKABE_ITO[5], U.OKABE_ITO[6]]):
        d = inter[inter.weekend == wk]
        b = pd.cut(d.t_max, bins)
        m = d.groupby(b, observed=True).interaction.agg(["mean", "sem", "count"])
        m = m[m["count"] >= 15]
        ax.errorbar([iv.mid for iv in m.index], m["mean"], yerr=1.96 * m["sem"], marker=mk2, ms=3.5, lw=1,
                    capsize=2, color=col, label=lab)
    ax.axhline(0, color="0.3", lw=0.6)
    ax.set_xlabel(r"Daily $T_{max}$ ($^\circ$C)")
    ax.set_ylabel(r"SHAP interaction $T_{max}\times$ weekday")
    ax.legend(fontsize=6.5)
    U.panel_label(ax, "d", x=-0.3)
    U.save_figure(fig, "fig_06_explainability_robustness")


def table_05(rr: pd.DataFrame, conv: pd.DataFrame) -> None:
    effs = list(EFFECTS)
    t = conv.copy()
    t.to_csv(U.RESULTS / "exogenous_effects_by_series.csv", index=False)
    wide = []
    for s in U.SERIES:
        row = {"Series": U.SERIES_LABEL[s]}
        for e in effs:
            r = rr[(rr.series == s) & (rr.effect == e)]
            row[e] = "--" if r.empty else (f"{r.rr.iloc[0]:.2f} [{r.rr_ci_low.iloc[0]:.2f}, {r.rr_ci_high.iloc[0]:.2f}]"
                                           + ("*" if r.convergent.iloc[0] else ""))
        wide.append(row)
    tex = pd.DataFrame(wide)
    tex.columns = ["Series"] + [e.replace("≥", r"$\geq$").replace("°C", r"$^\circ$C") for e in effs]
    U.save_table(t, "table_05_exogenous_effects", tex, column_format="l" + "c" * len(effs))


def main() -> None:
    p, folds, _ = M.load_inputs()
    folds = M.test_folds(folds)  # explicaciones solo sobre 2023-2025
    sh = compute_shap(p, folds)
    sh.to_parquet(U.RESULTS / "shap_values.parquet", index=False)
    share = group_share(sh)
    share.to_csv(U.RESULTS / "shap_group_share.csv")
    imp = sh[P.ALL_FEATURES].abs().mean().sort_values(ascending=False)
    imp.to_csv(U.RESULTS / "shap_mean_abs.csv")
    preds = pd.read_parquet(U.RESULTS / "oof_predictions.parquet")
    scores = pd.read_csv(U.RESULTS / "fold_scores.csv")
    cal = calibration(preds)
    cal.to_csv(U.RESULTS / "calibration.csv", index=False)
    fig_04(sh)
    fig_05(preds, scores, cal, p)
    # efectos exógenos medios (para el manifiesto)
    eff = {}
    for f in ["festivo", "puente", "quincena", "temporada_decembrina", "semana_santa", "weekend"]:
        on = sh[sh[f"x_{f}"] == 1][f].mean()
        off = sh[sh[f"x_{f}"] == 0][f].mean()
        eff[f] = {"mean_shap_on": on, "mean_shap_off": off, "rate_ratio": float(np.exp(on - off))}
    pd.DataFrame(eff).T.to_csv(U.RESULTS / "exogenous_effects.csv")
    print(imp.head(15).round(4)); print(share.round(3)); print(pd.DataFrame(eff).T.round(3))

    # ---- explicabilidad robusta
    stab = fold_stability(sh)
    stab.to_csv(U.RESULTS / "shap_group_share_by_fold.csv")
    rr = effect_rr(sh)
    conv = rr.merge(glm_convergent(p), on=["series", "effect"], how="left")
    # evidencia convergente: atribución SHAP estable entre folds Y efecto GLM significativo tras Holm
    # (familia de 99 pruebas) con el mismo signo
    conv["convergent"] = (conv.significant & (conv.glm_p_holm < U.ALPHA)
                          & (np.sign(conv.glm_log_rr) == np.sign(conv.log_rr)))
    rr = conv
    from scipy import stats as st
    c = conv.dropna(subset=["log_rr", "glm_log_rr"])
    rho = st.spearmanr(c.log_rr, c.glm_log_rr)
    both = c[c.significant & (c.glm_p_holm < U.ALPHA)]
    summary = {"spearman_rho": rho.statistic, "spearman_p": rho.pvalue, "n_pairs": len(c),
               "sign_agreement_all": float(np.mean(np.sign(c.log_rr) == np.sign(c.glm_log_rr))),
               "n_both_significant": len(both),
               "sign_agreement_both_sig": float(np.mean(np.sign(both.log_rr) == np.sign(both.glm_log_rr))) if len(both) else np.nan}
    pd.Series(summary).to_csv(U.RESULTS / "convergent_validity.csv")
    inter = interaction_values(p, folds)
    inter.to_parquet(U.RESULTS / "shap_interactions_tmax_weekday.parquet", index=False)
    mabs = np.load(U.RESULTS / "shap_interaction_mean_abs_lastfold.npy")
    np.fill_diagonal(mabs, 0)
    iu = np.triu_indices_from(mabs, 1)
    top = sorted(zip(mabs[iu] * 2, [P.ALL_FEATURES[i] for i in iu[0]], [P.ALL_FEATURES[j] for j in iu[1]]), reverse=True)[:15]
    pd.DataFrame(top, columns=["mean_abs_interaction", "feature_a", "feature_b"]).to_csv(
        U.RESULTS / "shap_top_interactions_lastfold.csv", index=False)
    table_05(rr, conv)
    fig_06(stab, rr, conv, inter)
    print(stab.agg(["mean", "std", "min", "max"]).round(3).T)
    print(rr[rr.convergent][["series", "effect", "rr", "rr_ci_low", "rr_ci_high", "glm_log_rr", "glm_p"]]
          .round(3).to_string())
    print(pd.Series(summary).round(3))
    print(pd.DataFrame(top[:8]))


if __name__ == "__main__":
    main()
