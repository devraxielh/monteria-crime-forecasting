from __future__ import annotations

import importlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
U = importlib.import_module("00_config_and_utils")

PARAM_MODELS = ["SN-Poisson", "GLM-Poisson", "GLM-NB", "LGBM-NB"]
K_MAX = 80  # soporte para el CRPS exacto (máximo diario observado = 33)
N_BINS = 10


def load():
    preds = pd.read_parquet(U.RESULTS / "oof_predictions.parquet")
    s = pd.read_csv(U.RESULTS / "train_scales.csv", index_col=0).iloc[:, 0]
    scales = {(int(k.split("|")[0]), k.split("|")[1]): v for k, v in s.items()}
    preds["scale"] = [scales[(f, s_)] for f, s_ in zip(preds.fold, preds.series)]
    return preds


def cdf_matrix(mu, alpha, k):
    mu = np.clip(np.asarray(mu, float), 1e-6, None)[:, None]
    alpha = np.asarray(alpha, float)[:, None]
    pois = stats.poisson.cdf(k[None, :], mu)
    n = 1.0 / np.maximum(alpha, 1e-8)
    nb = stats.nbinom.cdf(k[None, :], n, n / (n + mu))
    return np.where(alpha <= 1e-8, pois, nb)


def exact_scores(preds: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(U.SEED)
    k = np.arange(K_MAX + 1)
    rows = []
    for m in PARAM_MODELS:
        d = preds[preds.model == m].copy()
        if "nb_alpha" not in d or d.nb_alpha.isna().all():
            raise ValueError(f"{m}: falta nb_alpha en las predicciones (script 02)")
        F = cdf_matrix(d["mean"].values, d.nb_alpha.fillna(0).values, k)
        y = d.y.values.astype(int)
        ind = (k[None, :] >= y[:, None]).astype(float)
        d["crps_exact"] = ((F - ind) ** 2).sum(axis=1)
        Fy = F[np.arange(len(y)), y]
        Fy1 = np.where(y > 0, F[np.arange(len(y)), np.maximum(y - 1, 0)], 0.0)
        d["pit"] = Fy1 + rng.uniform(size=len(y)) * (Fy - Fy1)
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def coverage_variants(preds: pd.DataFrame) -> pd.DataFrame:
    lo, hi, y = preds["q0.1"].values, preds["q0.9"].values, preds.y.values
    closed = (y >= lo) & (y <= hi)
    opened = (y > lo) & (y < hi)
    on_edge = closed & ~opened
    return preds.assign(cov_closed=closed.astype(float), cov_open=opened.astype(float),
                        cov_split=opened.astype(float) + 0.5 * on_edge)


def summarise(ex: pd.DataFrame, cv: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in U.MODEL_ORDER:
        g = cv[cv.model == m]
        if g.empty:
            continue
        r = {"model": m, "coverage_closed": g.cov_closed.mean(), "coverage_open": g.cov_open.mean(),
             "coverage_split": g.cov_split.mean()}
        r["abs_err_split"] = abs(r["coverage_split"] - 0.8)
        e = ex[ex.model == m]
        if not e.empty:
            blk = e.groupby(["fold", "series"]).agg(crps_exact=("crps_exact", "mean"), scale=("scale", "first"))
            r["sCRPS_exact"] = (blk.crps_exact / blk.scale).mean()
            approx = (U.crps_from_quantiles(e.y.values, e[[f"q{t:.1f}" for t in U.QUANTILES]].values) / e.scale)
            r["sCRPS_quantile"] = approx.groupby([e.fold, e.series]).mean().mean()
            h, _ = np.histogram(e.pit, bins=N_BINS, range=(0, 1))
            r["pit_mad"] = np.mean(np.abs(h / h.sum() - 1 / N_BINS))
            r["pit_ks_stat"] = stats.kstest(e.pit, "uniform").statistic
        rows.append(r)
    return pd.DataFrame(rows)


def agreement(ex: pd.DataFrame) -> dict:
    blk = []
    for m in PARAM_MODELS:
        e = ex[ex.model == m]
        approx = U.crps_from_quantiles(e.y.values, e[[f"q{t:.1f}" for t in U.QUANTILES]].values)
        b = pd.DataFrame({"model": m, "fold": e.fold.values, "series": e.series.values,
                          "exact": e.crps_exact.values / e.scale.values, "approx": approx / e.scale.values})
        blk.append(b.groupby(["model", "fold", "series"]).mean().reset_index())
    b = pd.concat(blk)
    b.to_csv(U.RESULTS / "count_eval_block_crps.csv", index=False)
    r = stats.pearsonr(b.exact, b.approx)
    # concordancia de rangos de los modelos dentro de cada bloque
    taus = []
    for _, g in b.groupby(["fold", "series"]):
        if len(g) == len(PARAM_MODELS):
            taus.append(stats.kendalltau(g.exact, g.approx).statistic)
    return {"pearson_r_block": r.statistic, "mean_kendall_tau_within_block": float(np.nanmean(taus)),
            "n_blocks": int(b.groupby(["fold", "series"]).ngroups)}


def table_14(t: pd.DataFrame) -> None:
    tex = pd.DataFrame({
        "Model": t.model,
        "Cov. closed": t.coverage_closed.map(lambda v: f"{v:.3f}"),
        "Cov. open": t.coverage_open.map(lambda v: f"{v:.3f}"),
        "Cov. split": t.coverage_split.map(lambda v: f"{v:.3f}"),
        "sCRPS (quantiles)": t.get("sCRPS_quantile").map(lambda v: "--" if pd.isna(v) else f"{v:.4f}"),
        "sCRPS (exact)": t.get("sCRPS_exact").map(lambda v: "--" if pd.isna(v) else f"{v:.4f}"),
        "PIT MAD": t.get("pit_mad").map(lambda v: "--" if pd.isna(v) else f"{v:.4f}")})
    U.save_table(t, "table_14_count_evaluation", tex, column_format="lrrrrrr")


def fig_12(ex: pd.DataFrame, t: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), gridspec_kw={"wspace": 0.45, "width_ratios": [1.1, 1, 1.25]})
    # (a) histogramas PIT aleatorizados
    ax = axes[0]
    edges = np.linspace(0, 1, N_BINS + 1)
    for m in PARAM_MODELS:
        h, _ = np.histogram(ex[ex.model == m].pit, bins=edges)
        ax.step(edges, np.r_[h, h[-1]] / h.sum() * N_BINS, where="post", color=U.MODEL_COLORS[m], lw=1.1, label=m)
    ax.axhline(1, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Randomised PIT")
    ax.set_ylabel("Density")
    ax.set_ylim(0, 2.0)
    ax.legend(fontsize=5.8, frameon=False, loc="upper center", ncol=2)
    # (b) sCRPS exacto vs. por cuantiles (bloques)
    ax = axes[1]
    b = pd.read_csv(U.RESULTS / "count_eval_block_crps.csv")
    for m in PARAM_MODELS:
        g = b[b.model == m]
        ax.scatter(g.exact, g.approx, s=6, alpha=0.6, color=U.MODEL_COLORS[m], marker=U.MODEL_MARKERS[m], lw=0)
    lim = [min(b.exact.min(), b.approx.min()) * 0.95, max(b.exact.max(), b.approx.max()) * 1.05]
    ax.plot(lim, lim, color="k", lw=0.8, ls="--")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("sCRPS, exact (full distribution)")
    ax.set_ylabel("sCRPS, 9-quantile approximation")
    # (c) cobertura cerrada / abierta / corregida
    ax = axes[2]
    tt = t.set_index("model").reindex([m for m in U.MODEL_ORDER if m in set(t.model)])
    yy = np.arange(len(tt))
    ax.hlines(yy, tt.coverage_open, tt.coverage_closed, color="0.6", lw=2.5)
    ax.scatter(tt.coverage_split, yy, color=[U.MODEL_COLORS[m] for m in tt.index], edgecolor="k", lw=0.4, s=22,
               zorder=3)
    ax.axvline(0.8, color="k", lw=0.8, ls="--")
    ax.set_yticks(yy, tt.index, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("Coverage of 80% interval\n(bar: open to closed; dot: split)")
    for ax, lab in zip(axes, "abc"):
        U.panel_label(ax, lab, x=-0.28 if lab != "c" else -0.62)
    U.save_figure(fig, "fig_12_count_evaluation")


def main():
    U.set_style()
    preds = load()
    ex = exact_scores(preds)
    ex[["model", "fold", "series", "date", "y", "crps_exact", "pit"]].to_csv(U.RESULTS / "count_eval_exact.csv",
                                                                           index=False)
    cv = coverage_variants(preds)
    t = summarise(ex, cv)
    agr = agreement(ex)
    pd.Series(agr).to_csv(U.RESULTS / "count_eval_agreement.csv")
    table_14(t)
    fig_12(ex, t)
    pd.set_option("display.width", 200)
    print(t.round(4).to_string(index=False))
    print(agr)


if __name__ == "__main__":
    main()
