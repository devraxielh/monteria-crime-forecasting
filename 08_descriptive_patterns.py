from __future__ import annotations

import importlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib.colors import TwoSlopeNorm
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
U = importlib.import_module("00_config_and_utils")

DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
SPECIAL = {"weekend": "Weekend", "festivo": "Holiday", "puente": "Long\nweekend", "vispera_festivo": "Holiday\neve",
           "quincena": "Payday", "temporada_decembrina": "December\nseason", "semana_santa": "Holy\nWeek",
           "covid_lockdown": "COVID\nlockdown"}


def load() -> pd.DataFrame:
    p = pd.read_parquet(U.RESULTS / "panel_features.parquet")
    p = p[(p.date >= U.TRAIN_START) & (p.date <= U.TEST_END)].copy()
    p["year"] = p.date.dt.year
    p["dow"] = p.date.dt.day_of_week
    p["month"] = p.date.dt.month
    return p


FOURIER = "doy_s1 + doy_c1 + doy_s2 + doy_c2"


def ratio_matrix(p: pd.DataFrame, col: str, levels) -> pd.DataFrame:
    """Razón cruda media(categoría) / media(serie). Solo se usa para el año (panel d)."""
    g = p.groupby(["series", col]).y.mean().unstack(col).reindex(columns=levels)
    return g.div(p.groupby("series").y.mean(), axis=0).reindex(U.SERIES)


def fit_full(d: pd.DataFrame):
    return smf.glm("y ~ C(year) + C(dow) + C(month)", d, family=sm.families.Poisson()).fit()


def adjusted_matrix(p: pd.DataFrame, col: str, levels) -> pd.DataFrame:
    """Estandarización marginal: E[y | col = k, resto observado] / E[y], con el GLM ajustado por año."""
    out = {}
    for s in U.SERIES:
        d = p[p.series == s]
        res = fit_full(d)
        base = res.predict(d).mean()
        out[s] = [res.predict(d.assign(**{col: k})).mean() / base for k in levels]
    return pd.DataFrame(out, index=list(levels)).T.reindex(U.SERIES)


def special_matrix(p: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """RR ajustado de cada fecha especial y su p de Holm (familia de 8 efectos x 11 delitos)."""
    rr, pv = {}, {}
    for f in SPECIAL:
        rr[f], pv[f] = {}, {}
        dow_term = "" if f == "weekend" else "C(dow) + "
        for s in U.SERIES:
            d = p[p.series == s].copy()
            d[f] = d[f].astype(float)
            res = smf.glm(f"y ~ C(year) + {dow_term}{FOURIER} + {f}", d,
                          family=sm.families.Poisson()).fit(scale="X2")
            rr[f][s], pv[f][s] = float(np.exp(res.params[f])), float(res.pvalues[f])
    rr, pv = pd.DataFrame(rr).reindex(U.SERIES), pd.DataFrame(pv).reindex(U.SERIES)
    p_holm = multipletests(pv.values.ravel(), method="holm")[1].reshape(pv.shape)
    pv_holm = pd.DataFrame(p_holm, index=pv.index, columns=pv.columns)
    long = pd.DataFrame({"rr": rr.stack(), "p": pv.stack(), "p_holm": pv_holm.stack()})
    long.index.names = ["series", "effect"]
    long.to_csv(U.RESULTS / "descriptive_special_days_adjusted.csv")
    return rr, pv_holm


def factor_test(d: pd.DataFrame, factor: str) -> tuple[float, float]:
    full = fit_full(d)
    red_terms = [t for t in ("C(year)", "C(dow)", "C(month)") if t != f"C({factor})"]
    red = smf.glm("y ~ " + " + ".join(red_terms), d, family=sm.families.Poisson()).fit()
    phi = full.pearson_chi2 / full.df_resid
    df_num = red.df_resid - full.df_resid
    F = (red.deviance - full.deviance) / df_num / phi
    return float(F), float(stats.f.sf(F, df_num, full.df_resid))


def table_08(p, r_dow, r_mon) -> pd.DataFrame:
    rows = []
    for s in U.SERIES:
        d = p[p.series == s]
        F_d, p_d = factor_test(d, "dow")
        F_m, p_m = factor_test(d, "month")
        rd, rm = r_dow.loc[s], r_mon.loc[s]
        rows.append({"Crime": U.SERIES_LABEL[s], "Mean/day": d.y.mean(), "Total": int(d.y.sum()),
                     "Peak day": DOW[int(np.argmax(rd.values))], "RR peak day": rd.max(),
                     "Lowest day": DOW[int(np.argmin(rd.values))], "RR lowest day": rd.min(),
                     "F weekday": F_d, "p weekday": p_d,
                     "Peak month": MONTHS[int(np.argmax(rm.values))], "RR peak month": rm.max(),
                     "Lowest month": MONTHS[int(np.argmin(rm.values))], "RR lowest month": rm.min(),
                     "F month": F_m, "p month": p_m})
    t = pd.DataFrame(rows)
    adj = multipletests(np.r_[t["p weekday"].values, t["p month"].values], method="holm")[1]
    n = len(t)
    t.insert(t.columns.get_loc("p weekday") + 1, "p weekday Holm", adj[:n])
    t["p month Holm"] = adj[n:]
    tex = pd.DataFrame({
        "Crime": t.Crime, "Mean/day": t["Mean/day"].map(lambda v: f"{v:.2f}"),
        "Peak day (RR)": [f"{a} ({b:.2f})" for a, b in zip(t["Peak day"], t["RR peak day"])],
        "Lowest day (RR)": [f"{a} ({b:.2f})" for a, b in zip(t["Lowest day"], t["RR lowest day"])],
        "HOLMP weekday": t["p weekday Holm"].map(lambda v: f"{v:.1e}"),
        "Peak month (RR)": [f"{a} ({b:.2f})" for a, b in zip(t["Peak month"], t["RR peak month"])],
        "Lowest month (RR)": [f"{a} ({b:.2f})" for a, b in zip(t["Lowest month"], t["RR lowest month"])],
        "HOLMP month": t["p month Holm"].map(lambda v: f"{v:.1e}")})
    U.save_table(t, "table_08_descriptive_peaks", tex_df=tex, column_format="lrllrllr")
    # latex_escape escaparía la notación matemática: se usa un marcador y se sustituye después
    f = U.TABLES / "table_08_descriptive_peaks.tex"
    f.write_text(f.read_text(encoding="utf-8").replace("HOLMP", r"$p_{\mathrm{Holm}}$"), encoding="utf-8")
    return t


def heat(ax, m: pd.DataFrame, xlabels, vmax_log2: float, annotate: bool, fs: float = 5.5, sig=None):
    z = np.log2(m.values.astype(float))
    norm = TwoSlopeNorm(vmin=-vmax_log2, vcenter=0.0, vmax=vmax_log2)
    im = ax.imshow(np.clip(z, -vmax_log2, vmax_log2), cmap="RdBu_r", norm=norm, aspect="auto")
    ax.set_xticks(range(len(xlabels)), xlabels, fontsize=7)
    ax.set_yticks(range(len(m)), [U.SERIES_LABEL[s] for s in m.index], fontsize=7)
    ax.tick_params(length=0)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    if annotate:
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                v = m.values[i, j]
                if np.isfinite(v):
                    star = "*" if sig is not None and sig.values[i, j] < U.ALPHA else ""
                    ax.text(j, i, f"{v:.2f}{star}", ha="center", va="center", fontsize=fs,
                            color="white" if abs(np.log2(v)) > 0.6 * vmax_log2 else "black")
    return im


def fig_09(r_dow, r_mon, r_sp, r_year, sp_p):
    fig = plt.figure(figsize=(7.2, 7.0))
    gs = fig.add_gridspec(3, 2, width_ratios=[7, 12], height_ratios=[1, 1, 0.045], hspace=0.30, wspace=0.08)
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(2)])
    lim = 0.5  # log2: razones entre 0.71 y 1.41 saturan
    im = heat(axes[0, 0], r_dow, DOW, lim, True)
    heat(axes[0, 1], r_mon, MONTHS, lim, True, fs=5)
    axes[0, 1].set_yticklabels([])
    heat(axes[1, 0], r_sp, list(SPECIAL.values()), lim, True, fs=4.6, sig=sp_p)
    axes[1, 0].tick_params(axis="x", labelsize=5.5, rotation=90)
    lim_y = 1.5
    im_y = heat(axes[1, 1], r_year, [str(y) for y in r_year.columns], lim_y, True, fs=4.6)
    axes[1, 1].set_yticklabels([])
    axes[1, 1].tick_params(axis="x", labelsize=6, rotation=90)
    for ax, lab in zip(axes.flat, "abcd"):
        U.panel_label(ax, lab, x=-0.02 if lab in "bd" else -0.62, y=1.02)
    for im_, col, l_, txt in ((im, 0, lim, "Rate ratio, panels (a)–(c)"), (im_y, 1, lim_y, "Rate ratio, panel (d)")):
        cax = fig.add_subplot(gs[2, col])
        cb = fig.colorbar(im_, cax=cax, orientation="horizontal")
        ticks = np.linspace(-l_, l_, 5)
        cb.set_ticks(ticks, labels=[f"{2 ** t:.2f}" for t in ticks])
        cb.ax.tick_params(labelsize=6)
        cb.set_label(txt + " (log scale)", fontsize=7)
    U.save_figure(fig, "fig_09_temporal_patterns")


def main():
    U.set_style()
    p = load()
    r_dow = adjusted_matrix(p, "dow", range(7))
    r_mon = adjusted_matrix(p, "month", range(1, 13))
    r_sp, sp_p = special_matrix(p)
    years = sorted(p.year.unique())
    r_year = ratio_matrix(p, "year", years)
    for name, m in (("dow", r_dow), ("month", r_mon), ("special", r_sp), ("year", r_year)):
        m.to_csv(U.RESULTS / f"descriptive_ratio_{name}.csv")
        assert np.isfinite(m.values).all(), f"NaN en matriz {name}"
    t = table_08(p, r_dow, r_mon)
    fig_09(r_dow, r_mon, r_sp, r_year, sp_p)
    print(t[["Crime", "Peak day", "RR peak day", "p weekday Holm", "Peak month", "RR peak month", "p month Holm"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
