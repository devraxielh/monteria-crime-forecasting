from __future__ import annotations

import importlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
U = importlib.import_module("00_config_and_utils")

def q_alpha(k: int, alpha: float = 0.05) -> float:
    """Valor crítico de Nemenyi: cuantil del rango studentizado (gl infinitos) / sqrt(2) (Demšar, 2006)."""
    from scipy.stats import studentized_range
    return float(studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2))


EXTRA_PAIRS = [("Stacked", "LGBM-NB"), ("Stacked", "GLM-ZINB"), ("LGBM-NB+CQR", "LGBM-NB"),
               ("Chronos-2-FT", "Chronos-2"), ("TimesFM-2.5", "Chronos-2"), ("TiRex", "Chronos-2"),
               ("GLM-ZINB", "GLM-NB")]
METRICS = ["sCRPS", "MASE", "CovErr80"]  # CovErr80 = |cobertura 80 % - 0.80| por bloque


def blocks(scores: pd.DataFrame, metric: str) -> pd.DataFrame:
    return scores.pivot_table(index=["fold", "series"], columns="model", values=metric)[U.MODEL_ORDER]


def main() -> None:
    scores = pd.read_csv(U.RESULTS / "fold_scores.csv")
    scores["CovErr80"] = (scores.Coverage80 - 0.8).abs()
    rows, friedman_rows = [], []
    subsets = {"All series": scores,
               "Excl. regime-shift series": scores[~scores.series.isin(U.SHIFTED_SERIES)]}
    for subset, sc in subsets.items():
        for metric in METRICS:
            b = blocks(sc, metric)
            fr = stats.friedmanchisquare(*[b[m].values for m in U.MODEL_ORDER])
            ranks = b.rank(axis=1).mean()
            friedman_rows.append({"subset": subset, "metric": metric, "chi2": fr.statistic, "p_value": fr.pvalue,
                                  "n_blocks": len(b), **{f"rank_{m}": ranks[m] for m in U.MODEL_ORDER}})
            if metric == "sCRPS" and subset == "All series":
                nem = sp.posthoc_nemenyi_friedman(b.values)
                nem.index = nem.columns = U.MODEL_ORDER
                nem.to_csv(U.RESULTS / "nemenyi_pvalues.csv")
                fig_cd(ranks, len(b), nem)
            rows.append({"Subset": subset, "Comparison": f"Friedman (all {len(U.MODEL_ORDER)} models)",
                         "Metric": metric, "Test": "Friedman", "Statistic": fr.statistic, "p_value": fr.pvalue,
                         "Effect": np.nan, "Effect_type": "", "Mean_delta": np.nan, "CI_low": np.nan,
                         "CI_high": np.nan, "n_blocks": len(b)})
            pairs = [(U.PROPOSED, m) for m in U.MODEL_ORDER if m != U.PROPOSED] + EXTRA_PAIRS
            comps = []
            for a_, b_ in pairs:
                x, y = b[a_].values, b[b_].values
                d = x - y
                w = stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided") if np.any(d != 0) else None
                lo, hi = U.bootstrap_ci(d)
                comps.append({"Subset": subset, "Comparison": f"{a_} vs {b_}", "Metric": metric, "Test": "Wilcoxon",
                              "Statistic": np.nan if w is None else w.statistic,
                              "p_value": 1.0 if w is None else w.pvalue,
                              "Effect": U.rank_biserial(x, y), "Effect_type": "rank-biserial",
                              "Mean_delta": d.mean(), "CI_low": lo, "CI_high": hi, "n_blocks": len(b)})
            adj = multipletests([c["p_value"] for c in comps], method="holm")[1]
            for c, a in zip(comps, adj):
                c["p_holm"] = a
            rows += comps
    t = pd.DataFrame(rows)
    pd.DataFrame(friedman_rows).to_csv(U.RESULTS / "friedman_ranks.csv", index=False)

    def interp(r):
        if r.Test == "Friedman":
            return "Models differ" if r.p_value < U.ALPHA else "No difference"
        if r.p_holm >= U.ALPHA:
            return "n.s."
        mag = abs(r.Effect)
        size = "large" if mag >= 0.5 else "medium" if mag >= 0.3 else "small" if mag >= 0.1 else "negligible"
        first = r.Comparison.split(" vs ")[0]
        return f"{first} {'better' if r.Mean_delta < 0 else 'worse'} ({size})"

    t["Interpretation"] = t.apply(interp, axis=1)
    t.to_csv(U.RESULTS / "statistical_tests_full.csv", index=False)
    # la tabla del paper: subconjunto completo, métricas sCRPS y CovErr80; robustez en tabla suplementaria
    main_t = t[(t.Subset == "All series") & t.Metric.isin(["sCRPS", "CovErr80"])].reset_index(drop=True)
    tex_from(t[t.Subset == "Excl. regime-shift series"].reset_index(drop=True), "table_S1_robustness_excl_shifted")
    tex_from(main_t, "table_04_statistical_significance")
    print(t[["Subset", "Comparison", "Metric", "Statistic", "p_value", "p_holm", "Effect", "Mean_delta",
             "Interpretation"]].to_string())
    print(pd.DataFrame(friedman_rows).round(3).T)


def tex_from(t: pd.DataFrame, name: str) -> None:
    tex = pd.DataFrame({
        "Comparison": t.Comparison, "Metric": t.Metric,
        "Statistic": ["--" if pd.isna(s) else f"{s:.1f}" for s in t.Statistic],
        "p": [f"{p:.2e}" for p in t.p_value],
        "p (Holm)": ["--" if pd.isna(p) else f"{p:.2e}" for p in t.get("p_holm", pd.Series([np.nan] * len(t)))],
        r"$r_{rb}$": ["--" if pd.isna(e) else f"{e:+.3f}" for e in t.Effect],
        r"Mean $\Delta$ [95% CI]": ["--" if pd.isna(d) else f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]"
                                   for d, lo, hi in zip(t.Mean_delta, t.CI_low, t.CI_high)],
        "Interpretation": t.Interpretation,
    })
    U.save_table(t, name, tex, column_format="llrrrrrl")


def fig_cd(ranks: pd.Series, n: int, nem: pd.DataFrame) -> None:
    """Diagrama de diferencia crítica (Demšar, 2006) con cliques no significativos (Nemenyi)."""
    k = len(ranks)
    cd = q_alpha(k) * np.sqrt(k * (k + 1) / (6.0 * n))
    r = ranks.sort_values()
    names = list(r.index)
    cl = []
    for i in range(k):
        j = i
        while j + 1 < k and all(nem.loc[names[a], names[b]] >= U.ALPHA
                                for a in range(i, j + 2) for b in range(i, j + 2) if a != b):
            j += 1
        if j > i and not any(c[0] <= i and c[1] >= j for c in cl):
            cl.append((i, j))
    fig, ax = plt.subplots(figsize=(U.COL2, 3.4))
    lo, hi = 1, k
    ax.set_xlim(lo - 0.3, hi + 0.3); ax.set_ylim(-0.02, 1)
    ax.axis("off")
    ax.hlines(0.86, lo, hi, color="black", lw=0.8)
    for t in range(lo, hi + 1):
        ax.vlines(t, 0.86, 0.89, color="black", lw=0.8)
        ax.text(t, 0.91, str(t), ha="center", fontsize=7.5)
    ax.hlines(0.985, lo, lo + cd, color="black", lw=1.2)
    ax.text(lo + cd + 0.1, 0.985, f"CD = {cd:.2f}", ha="left", va="center", fontsize=7.5)
    step = 0.028
    for n_c, c in enumerate(cl):
        yb = 0.825 - n_c * step
        ax.hlines(yb, r.iloc[c[0]] - 0.04, r.iloc[c[1]] + 0.04, color="black", lw=2.0)
    top = 0.825 - len(cl) * step - 0.03
    half = (k + 1) // 2
    for i, (m, v) in enumerate(r.items()):
        left = i < half
        pos = i if left else k - 1 - i
        y = top - (top - 0.02) * pos / max(half - 1, 1)
        xt = lo - 0.15 if left else hi + 0.15
        ax.plot([v, v, xt], [0.86, y, y], color=U.MODEL_COLORS[m], lw=0.9)
        ax.text(xt + (-0.05 if left else 0.05), y, f"{m} ({v:.2f})", ha="right" if left else "left",
                va="center", fontsize=7, fontweight="bold" if m == U.PROPOSED else "normal")
    U.save_figure(fig, "fig_13_critical_difference")


if __name__ == "__main__":
    main()
