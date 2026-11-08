from __future__ import annotations

import importlib
import sys
import time
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

FULL = "Full model"
F = P.ALL_FEATURES


def drop(groups):
    rm = set(sum(groups, []))
    return [f for f in F if f not in rm]


ABLATIONS = {
    "− Climate": dict(features=drop([P.CLIMATE])),
    "Climate lagged 7 d (no perfect prognosis)": dict(features=F, lag_clim=True),
    "− Holidays": dict(features=drop([P.HOLIDAY])),
    "− Payday (quincena)": dict(features=drop([P.PAYDAY])),
    "− Holidays, payday & climate": dict(features=drop([P.HOLIDAY, P.PAYDAY, P.CLIMATE])),
    "− Lag features": dict(features=drop([P.LAG_FEATURES])),
    "Local per-series models": dict(features=F, local=True),
    "Poisson instead of NB": dict(poisson_from_full=True),
    "Recency-weighted training (half-life 2 y)": dict(features=F, weight_halflife=730),
}
SENS = {"num_leaves": [7, 15, 31, 63, 127], "n_estimators": [200, 400, 800, 1600]}
DEFAULTS = {"num_leaves": 31, "n_estimators": 800}


def load_scales():
    s = pd.read_csv(U.RESULTS / "train_scales.csv", index_col=0).iloc[:, 0]
    return {(int(k.split("|")[0]), k.split("|")[1]): v for k, v in s.items()}


def run_config(p, p_lag, folds, name, features=F, params=None, local=False, lag_clim=False, weight_halflife=None):
    out = []
    for _, f in folds.iterrows():
        mu, q, _, _, te = M.fit_predict_lgbm(p_lag if lag_clim else p, f, features, params=params, local=local,
                                             weight_halflife=weight_halflife)
        out.append(M.package(te, name, f.fold, mu, q))
    return pd.concat(out, ignore_index=True)


def main() -> None:
    p, folds, _ = M.load_inputs()
    folds = M.test_folds(folds)
    p_lag = M.lag_climate(p)
    scales = load_scales()
    full = pd.read_parquet(U.RESULTS / "predictions_LGBM-NB.parquet").assign(model=FULL)
    full = full[full.fold.isin(set(folds.fold))]
    preds = [full]
    t0 = time.time()
    for name, cfg in ABLATIONS.items():
        if cfg.get("poisson_from_full"):
            d = full.copy()
            d[M.QCOLS] = U.count_quantiles(d["mean"].values, 0.0)
            preds.append(d.assign(model=name))
        else:
            preds.append(run_config(p, p_lag, folds, name, **cfg))
        print(f"{name}: {time.time() - t0:.0f}s", flush=True)
    abl = M.score(pd.concat(preds, ignore_index=True), scales)
    abl.to_csv(U.RESULTS / "ablation_scores.csv", index=False)

    sens = []
    for hp, values in SENS.items():
        for v in values:
            if v == DEFAULTS[hp]:
                d = full.copy()
            else:
                d = run_config(p, p_lag, folds, "tmp", params={hp: v})
            sc = M.score(d.assign(model="tmp"), scales).assign(param=hp, value=v)
            sens.append(sc)
            print(f"{hp}={v}: {time.time() - t0:.0f}s", flush=True)
    sens = pd.concat(sens, ignore_index=True)
    sens.to_csv(U.RESULTS / "sensitivity_scores.csv", index=False)
    table_03(abl)
    fig_03(abl, sens)


def table_03(abl: pd.DataFrame) -> None:
    base = abl[abl.model == FULL].set_index(["fold", "series"]).sort_index()
    rows = []
    for name in [FULL] + list(ABLATIONS):
        g = abl[abl.model == name].set_index(["fold", "series"]).sort_index()
        r = {"Configuration": name, "sCRPS_mean": g.sCRPS.mean(), "sCRPS_sd": g.sCRPS.std(),
             "MASE_mean": g.MASE.mean(), "Coverage80_mean": g.Coverage80.mean()}
        if name != FULL:
            d = (g.sCRPS - base.sCRPS).values
            lo, hi = U.bootstrap_ci(d)
            w = stats.wilcoxon(g.sCRPS.values, base.sCRPS.values, zero_method="wilcox")
            r.update({"delta_sCRPS": d.mean(), "delta_ci_low": lo, "delta_ci_high": hi,
                      "delta_pct": 100 * d.mean() / base.sCRPS.mean(), "wilcoxon_W": w.statistic,
                      "p_value": w.pvalue, "rank_biserial": U.rank_biserial(base.sCRPS.values, g.sCRPS.values)})
        rows.append(r)
    t = pd.DataFrame(rows)
    m = t.p_value.notna()
    t.loc[m, "p_holm"] = multipletests(t.loc[m, "p_value"], method="holm")[1]
    tex = pd.DataFrame({
        "Configuration": t.Configuration,
        "sCRPS": [f"{a:.3f} $\\pm$ {b:.3f}" for a, b in zip(t.sCRPS_mean, t.sCRPS_sd)],
        r"$\Delta$ sCRPS [95% CI]": ["--" if pd.isna(d) else f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]"
                                     for d, lo, hi in zip(t.delta_sCRPS, t.delta_ci_low, t.delta_ci_high)],
        r"$\Delta$ %": ["--" if pd.isna(x) else f"{x:+.2f}" for x in t.delta_pct],
        "p (Holm)": ["--" if pd.isna(x) else f"{x:.2e}" for x in t.p_holm],
        "Cov. 80%": [f"{x:.3f}" for x in t.Coverage80_mean],
    })
    U.save_table(t, "table_03_ablation_study", tex, column_format="lrrrrr")


def fig_03(abl: pd.DataFrame, sens: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(U.COL2, 3.1), gridspec_kw={"width_ratios": [1.6, 1, 1]})
    base = abl[abl.model == FULL].set_index(["fold", "series"]).sort_index().sCRPS
    ax = axes[0]
    names = list(ABLATIONS)
    deltas, cis = [], []
    for n in names:
        d = (abl[abl.model == n].set_index(["fold", "series"]).sort_index().sCRPS - base).values
        deltas.append(100 * d.mean() / base.mean())
        lo, hi = U.bootstrap_ci(d)
        cis.append((100 * lo / base.mean(), 100 * hi / base.mean()))
    order = np.argsort(deltas)
    y = np.arange(len(names))
    for yi, i in zip(y, order):
        c = U.OKABE_ITO[6] if cis[i][0] > 0 else (U.OKABE_ITO[3] if cis[i][1] < 0 else "0.5")
        ax.errorbar(deltas[i], yi, xerr=[[deltas[i] - cis[i][0]], [cis[i][1] - deltas[i]]],
                    fmt="o", color=c, ms=4, capsize=2, lw=1)
    ax.axvline(0, color="0.3", lw=0.7)
    ax.set_yticks(y, [names[i].replace(" (no perfect prognosis)", "\n(no perfect prognosis)") for i in order], fontsize=7)
    ax.set_xlabel(r"$\Delta$ sCRPS vs. full model (%)")
    ax.grid(axis="y", visible=False)
    U.panel_label(ax, "a", x=-0.9)
    for ax, (hp, lab) in zip(axes[1:], [("num_leaves", "Number of leaves"), ("n_estimators", "Number of trees")]):
        s = sens[sens.param == hp].groupby("value").sCRPS
        mean = s.mean()
        ci = np.array([U.bootstrap_ci(v.values, n_boot=1000) for _, v in s])
        ax.fill_between(mean.index, ci[:, 0], ci[:, 1], color=U.OKABE_ITO[3], alpha=0.2, lw=0)
        ax.plot(mean.index, mean.values, "o-", color=U.OKABE_ITO[3], ms=4, lw=1)
        ax.axvline(DEFAULTS[hp], color="0.5", ls=":", lw=0.8)
        ax.set_xscale("log", base=2); ax.set_xticks(SENS[hp], [str(v) for v in SENS[hp]])
        ax.minorticks_off()
        ax.set_xlabel(lab); ax.set_ylabel("Mean sCRPS")
    U.panel_label(axes[1], "b", x=-0.35); U.panel_label(axes[2], "c", x=-0.35)
    fig.tight_layout()
    U.save_figure(fig, "fig_03_ablation_and_sensitivity")


if __name__ == "__main__":
    main()
