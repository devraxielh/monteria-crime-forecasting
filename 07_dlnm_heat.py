from __future__ import annotations

import importlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import build_design_matrices, dmatrix
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
U = importlib.import_module("00_config_and_utils")

MAXLAG = 7
OUTCOMES = {
    "Violent (assault + homicide + DV)": ["lesiones_personales", "homicidio", "violencia_intrafamiliar"],
    "Property theft (4 types)": ["hurto_personas", "hurto_comercio", "hurto_residencias", "hurto_vehiculos"],
    **{U.SERIES_LABEL[s]: [s] for s in U.SERIES},
}
FIG_OUTCOMES = ["Violent (assault + homicide + DV)", "Property theft (4 types)", "Domestic violence",
                "Traffic homicide"]
EXPOSURES = {"t_max": r"$T_{max}$ ($^\circ$C)", "heat_index": r"Heat index ($^\circ$C)"}


def load() -> pd.DataFrame:
    d = pd.read_csv(U.DATA_PROCESSED / "panel_diario.csv", parse_dates=["fecha"]).set_index("fecha")
    d = d.loc[U.TRAIN_START:U.TEST_END, U.SERIES]
    p = pd.read_parquet(U.RESULTS / "panel_features.parquet")
    cal = p.drop_duplicates("date").set_index("date")
    keep = ["dow", "festivo", "puente", "vispera_festivo", "quincena", "temporada_decembrina", "semana_santa",
            "covid_lockdown", "trend", "t_max", "heat_index"]
    return d.join(cal[keep])


def crossbasis(x: pd.Series):
    """Devuelve (CB matrix n x (vx*vl), design_info exposición, matriz de rezagos C, nodos)."""
    xi = x.interpolate(limit=2, limit_area="inside")
    v = xi.dropna().values
    knots = tuple(float(k) for k in np.percentile(v, [10, 50, 90]))
    lo, hi = float(v.min()), float(v.max())
    di = dmatrix(f"bs(x, knots={knots}, degree=2, lower_bound={lo}, upper_bound={hi}) - 1",
                 {"x": v}, return_type="dataframe").design_info
    lags = np.arange(MAXLAG + 1)
    C = np.column_stack([np.ones_like(lags), lags, lags ** 2]).astype(float)  # rezago: polinomio grado 2
    n = len(xi)
    vx = len(di.column_names)
    cb = np.full((n, vx * C.shape[1]), np.nan)
    Bl = []
    for L in lags:
        xl = xi.shift(L).values
        B = np.full((n, vx), np.nan)
        ok = ~np.isnan(xl)
        B[ok] = np.asarray(build_design_matrices([di], {"x": xl[ok]})[0])
        Bl.append(B)
    Bl = np.stack(Bl, axis=1)  # n x L x vx
    for j in range(C.shape[1]):
        cb[:, j * vx:(j + 1) * vx] = np.einsum("nlv,l->nv", Bl, C[:, j])
    return cb, di, C, knots, (lo, hi)


def basis_at(di, values):
    return np.asarray(build_design_matrices([di], {"x": np.atleast_1d(values).astype(float)})[0])


def fit_outcome(df: pd.DataFrame, y: pd.Series, cb: np.ndarray):
    # el modelo se construye SOLO sobre los días con ventana de exposición completa
    ok = ~np.isnan(cb).any(axis=1) & y.notna().values
    first, last = np.flatnonzero(ok)[[0, -1]]
    df, y, cb, ok = df.iloc[first:last + 1], y.iloc[first:last + 1], cb[first:last + 1], ok[first:last + 1]
    years = (df.index.max() - df.index.min()).days / 365.25
    tdf = max(int(round(4 * years)), 4)
    T = dmatrix(f"bs(trend, df={tdf}, degree=3) - 1", {"trend": df.trend.values}, return_type="dataframe")
    X = pd.DataFrame(cb, index=df.index, columns=[f"cb{i}" for i in range(cb.shape[1])])
    X = X.join(pd.get_dummies(df.dow, prefix="dow", drop_first=True).astype(float))
    for c in ["festivo", "puente", "vispera_festivo", "quincena", "temporada_decembrina", "semana_santa",
              "covid_lockdown"]:
        if df.loc[ok, c].nunique() > 1:  # p. ej. COVID no existe en 2015-2019
            X[c] = df[c].astype(float)
    X = X.join(T.set_index(df.index).add_prefix("time_"))
    X = sm.add_constant(X)
    res = sm.GLM(y.values[ok], X.values[ok], family=sm.families.Poisson()).fit(scale="X2")
    k = cb.shape[1]
    beta = res.params[1:1 + k]
    V = res.cov_params()[1:1 + k, 1:1 + k]
    return res, beta, V, int(ok.sum())


def cumulative_rr(di, C, beta, V, x, ref):
    """log RR acumulado 0..MAXLAG para cada x frente a ref, con EE por método delta."""
    vx = len(di.column_names)
    Bx, Br = basis_at(di, x), basis_at(di, [ref])
    D = Bx - Br  # m x vx
    csum = C.sum(axis=0)  # suma de la base de rezagos
    G = np.einsum("mv,j->mjv", D, csum).reshape(len(x), -1)  # orden j-mayor como en cb
    est = G @ beta
    se = np.sqrt(np.einsum("mi,ij,mj->m", G, V, G))
    return est, se


def lag_rr(di, C, beta, V, x, ref):
    D = (basis_at(di, [x]) - basis_at(di, [ref]))[0]
    G = np.stack([np.concatenate([D * C[l, j] for j in range(C.shape[1])]) for l in range(C.shape[0])])
    est = G @ beta
    se = np.sqrt(np.einsum("li,ij,lj->l", G, V, G))
    return est, se


def main() -> None:
    df = load()
    curves, lagr, rows = [], [], []
    for expo, lab in EXPOSURES.items():
        cb, di, C, knots, (lo, hi) = crossbasis(df[expo])
        xs = df[expo].interpolate(limit=2, limit_area="inside").dropna()
        ref, p90, p99 = np.percentile(xs, [50, 90, 99])
        grid = np.linspace(np.percentile(xs, 1), np.percentile(xs, 99.5), 60)
        for name, cols in OUTCOMES.items():
            y = df[cols].sum(axis=1)
            res, beta, V, n = fit_outcome(df, y, cb)
            # prueba de Wald global de la asociación (todos los coeficientes de la cross-basis = 0)
            W = float(beta @ np.linalg.solve(V, beta))
            pw = float(stats.chi2.sf(W, len(beta)))
            est, se = cumulative_rr(di, C, beta, V, grid, ref)
            curves.append(pd.DataFrame({"exposure": expo, "outcome": name, "x": grid, "log_rr": est, "se": se}))
            e2, s2 = cumulative_rr(di, C, beta, V, np.array([p90, p99]), ref)
            le, ls = lag_rr(di, C, beta, V, p99, ref)
            lagr.append(pd.DataFrame({"exposure": expo, "outcome": name, "lag": np.arange(MAXLAG + 1),
                                      "log_rr": le, "se": ls}))
            rows.append({"Exposure": expo, "Outcome": name, "n_days": n, "dispersion": res.scale,
                         "ref": ref, "p90": p90, "p99": p99,
                         "RR_p90": np.exp(e2[0]), "RR_p90_low": np.exp(e2[0] - 1.96 * s2[0]),
                         "RR_p90_high": np.exp(e2[0] + 1.96 * s2[0]),
                         "RR_p99": np.exp(e2[1]), "RR_p99_low": np.exp(e2[1] - 1.96 * s2[1]),
                         "RR_p99_high": np.exp(e2[1] + 1.96 * s2[1]), "wald_chi2": W, "wald_p": pw})
            print(f"{expo:10s} {name:38s} n={n} RR99={np.exp(e2[1]):.3f} "
                  f"[{np.exp(e2[1] - 1.96 * s2[1]):.3f}, {np.exp(e2[1] + 1.96 * s2[1]):.3f}] p={pw:.2e}", flush=True)
    curves, lagr, t = pd.concat(curves), pd.concat(lagr), pd.DataFrame(rows)
    from statsmodels.stats.multitest import multipletests
    t["wald_p_holm"] = np.nan
    for e in EXPOSURES:
        m = t.Exposure == e
        t.loc[m, "wald_p_holm"] = multipletests(t.loc[m, "wald_p"], method="holm")[1]
    curves.to_csv(U.RESULTS / "dlnm_curves.csv", index=False)
    lagr.to_csv(U.RESULTS / "dlnm_lag_response.csv", index=False)
    ci = lambda r, k: f"{r[k]:.3f} [{r[k + '_low']:.3f}, {r[k + '_high']:.3f}]"  # noqa: E731
    tex = pd.DataFrame({
        "Exposure": t.Exposure.map({"t_max": "Tmax", "heat_index": "Heat index"}), "Outcome": t.Outcome,
        "Days": t.n_days, "Ref. (P50)": t.ref.round(1), "RR P90 vs P50 [95% CI]": [ci(r, "RR_p90") for _, r in t.iterrows()],
        "RR P99 vs P50 [95% CI]": [ci(r, "RR_p99") for _, r in t.iterrows()],
        "Wald p (Holm)": [f"{p:.2e}" for p in t.wald_p_holm]})
    U.save_table(t, "table_07_dlnm_heat", tex, column_format="llrrccr")
    fig_08(curves, lagr, t)


def fig_08(curves, lagr, t) -> None:
    fig, axes = plt.subplots(2, len(FIG_OUTCOMES), figsize=(U.COL2, 4.4), sharey="row")
    letters = iter("abcdefgh")
    for j, name in enumerate(FIG_OUTCOMES):
        ax = axes[0, j]
        for expo, col, ls in [("t_max", U.OKABE_ITO[6], "-"), ("heat_index", U.OKABE_ITO[5], "--")]:
            c = curves[(curves.outcome == name) & (curves.exposure == expo)]
            ax.fill_between(c.x, np.exp(c.log_rr - 1.96 * c.se), np.exp(c.log_rr + 1.96 * c.se), color=col,
                            alpha=0.15, lw=0)
            ax.plot(c.x, np.exp(c.log_rr), ls, color=col, lw=1.1,
                    label=r"$T_{max}$" if expo == "t_max" else "Heat index")
        ax.axhline(1, color="0.3", lw=0.6)
        ax.set_xlabel(r"Exposure ($^\circ$C)", fontsize=8)
        if j == 0:
            ax.set_ylabel("Cumulative RR (lag 0-7)")
            ax.legend(fontsize=6, loc="upper left")
        ax.text(0.03, 0.03, name.replace(" (assault + homicide + DV)", "\n(assault+hom.+DV)")
                .replace(" (4 types)", "\n(4 types)"), transform=ax.transAxes, fontsize=6.5, va="bottom")
        U.panel_label(ax, next(letters), x=-0.2)
        ax = axes[1, j]
        lr = lagr[(lagr.outcome == name) & (lagr.exposure == "t_max")]
        ax.fill_between(lr.lag, np.exp(lr.log_rr - 1.96 * lr.se), np.exp(lr.log_rr + 1.96 * lr.se),
                        color=U.OKABE_ITO[6], alpha=0.15, lw=0)
        ax.plot(lr.lag, np.exp(lr.log_rr), "o-", color=U.OKABE_ITO[6], ms=3, lw=1)
        ax.axhline(1, color="0.3", lw=0.6)
        ax.set_xlabel("Lag (days)", fontsize=8); ax.set_xticks(range(0, MAXLAG + 1, 2))
        if j == 0:
            ax.set_ylabel(r"RR at $T_{max}$ P99 vs. P50")
        U.panel_label(ax, next(letters), x=-0.2)
    fig.tight_layout()
    U.save_figure(fig, "fig_08_dlnm_heat")


if __name__ == "__main__":
    main()
