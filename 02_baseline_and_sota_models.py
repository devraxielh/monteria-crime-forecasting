from __future__ import annotations

import importlib
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
U = importlib.import_module("00_config_and_utils")
P = importlib.import_module("01_data_splitting_and_prep")

warnings.filterwarnings("ignore")
QCOLS = [f"q{t:.1f}" for t in U.QUANTILES]
INNER_VAL_DAYS = 182
MODELS_DIR = U.RESULTS / "models"
MODELS_DIR.mkdir(exist_ok=True)

LGBM_PARAMS = dict(objective="poisson", learning_rate=0.03, n_estimators=800, num_leaves=31,
                   min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                   reg_lambda=1.0, random_state=U.SEED, n_jobs=4, deterministic=True,
                   force_row_wise=True, verbose=-1)
FT_CONFIG = dict(finetune_mode="lora", learning_rate=1e-5, num_steps=200, batch_size=32, context_length=512)


# ------------------------------------------------------------------ datos
def load_inputs():
    p = pd.read_parquet(U.RESULTS / "panel_features.parquet")
    folds = pd.read_csv(U.RESULTS / "folds.csv", parse_dates=["train_end", "test_start", "test_end"])
    origins = pd.read_csv(U.RESULTS / "origins.csv", parse_dates=["origin", "date"])
    return p, folds, origins


def test_folds(folds):
    return folds[folds.role == "test"]


def lag_climate(p: pd.DataFrame, k: int = U.MIN_LAG) -> pd.DataFrame:
    """Variante sin 'pronóstico perfecto': el clima entra rezagado k días."""
    q = p.sort_values(["series", "date"]).copy()
    q[P.CLIMATE] = q.groupby("series")[P.CLIMATE].shift(k)
    return q.sort_index()


def split(p, fold_row):
    tr = p[p.date <= fold_row.train_end]
    te = p[(p.date >= fold_row.test_start) & (p.date <= fold_row.test_end)]
    return tr, te


def package(te: pd.DataFrame, model: str, fold: int, mean, q) -> pd.DataFrame:
    out = te[["series", "date", "y"]].copy()
    out["model"], out["fold"], out["mean"] = model, fold, np.asarray(mean, float)
    q = np.sort(np.clip(np.asarray(q, float), 0, None), axis=1)
    out[QCOLS] = q
    return out


def nb_quantiles_by_series(series: pd.Series, mu, alphas: dict) -> np.ndarray:
    q = np.zeros((len(mu), len(U.QUANTILES)))
    mu = np.asarray(mu, float)
    for s, a in alphas.items():
        m = (series.values == s)
        if m.any():
            q[m] = U.count_quantiles(mu[m], a)
    return q


# ------------------------------------------------------------------ LightGBM global
def _lgbm_fit(X, y, params, w=None):
    m = lgb.LGBMRegressor(**params)
    m.fit(X, y, sample_weight=w, categorical_feature=["series_id"] if "series_id" in X.columns else "auto")
    return m


def _prep_X(df, features):
    X = df[features].copy()
    if "series_id" in X:
        X["series_id"] = X["series_id"].astype("category")
    return X


def _recency_w(df, end, halflife):
    if halflife is None:
        return None
    age = (end - df.date).dt.days.values
    return 0.5 ** (age / halflife)


def fit_predict_lgbm(p, fold_row, features, params=None, dist="nb", local=False, weight_halflife=None):
    """Devuelve (mean, quantiles, alphas, model, test_df). Dispersión NB en validación interna."""
    params = {**LGBM_PARAMS, **(params or {})}
    tr, te = split(p, fold_row)
    inner_end = fold_row.train_end - pd.Timedelta(days=INNER_VAL_DAYS)
    tr_in, va = tr[tr.date <= inner_end], tr[tr.date > inner_end]
    feats = [f for f in features if not (local and f == "series_id")]
    groups = [(s, tr_in.series == s, va.series == s, tr.series == s, te.series == s) for s in U.SERIES] \
        if local else [(None, slice(None), slice(None), slice(None), slice(None))]
    mu_va, mu_te, model = np.zeros(len(va)), np.zeros(len(te)), None
    for s, m_in, m_va, m_tr, m_te in groups:
        a, b = tr_in[m_in], va[m_va]
        mi = _lgbm_fit(_prep_X(a, feats), a.y, params, _recency_w(a, inner_end, weight_halflife))
        mu_va[np.asarray(m_va) if local else slice(None)] = mi.predict(_prep_X(b, feats))
        c, d = tr[m_tr], te[m_te]
        model = _lgbm_fit(_prep_X(c, feats), c.y, params, _recency_w(c, fold_row.train_end, weight_halflife))
        mu_te[np.asarray(m_te) if local else slice(None)] = model.predict(_prep_X(d, feats))
    alphas = {s: (U.nb_alpha_moments(va.y[va.series == s], mu_va[(va.series == s).values]) if dist == "nb" else 0.0)
              for s in U.SERIES}
    q = nb_quantiles_by_series(te.series, mu_te, alphas)
    return mu_te, q, alphas, (None if local else model), te


# ------------------------------------------------------------------ GLM por serie
def glm_pipeline(num_feats):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import PoissonRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

    lags = list(P.LAG_FEATURES)
    other = [f for f in num_feats if f not in lags + ["dow", "month"]]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False), ["dow", "month"]),
        ("lag", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                          ("sc", StandardScaler())]), lags),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                          ("sc", StandardScaler())]), other),
    ])
    return Pipeline([("pre", pre), ("glm", PoissonRegressor(alpha=1e-3, max_iter=3000))])


def fit_predict_glm(p, fold_row):
    feats = [f for f in P.ALL_FEATURES if f != "series_id"]
    tr, te = split(p, fold_row)
    inner_end = fold_row.train_end - pd.Timedelta(days=INNER_VAL_DAYS)
    mu_te, alphas = np.zeros(len(te)), {}
    for s in U.SERIES:
        a, b = tr[(tr.series == s) & (tr.date <= inner_end)], tr[(tr.series == s) & (tr.date > inner_end)]
        pipe_in = glm_pipeline(feats).fit(a[feats], a.y)
        alphas[s] = U.nb_alpha_moments(b.y, pipe_in.predict(b[feats]))
        c, mt = tr[tr.series == s], (te.series == s).values
        pipe = glm_pipeline(feats).fit(c[feats], c.y)
        mu_te[mt] = pipe.predict(te.loc[mt, feats])
    return mu_te, alphas, te


# ------------------------------------------------------------------ ZINB por serie
ZINB_BIN = ["doy_s1", "doy_c1", "doy_s2", "doy_c2", "festivo", "puente", "quincena", "temporada_decembrina",
            "semana_santa", "covid_lockdown"]


def _zinb_design(df, st=None):
    X = pd.DataFrame(index=df.index)
    for d in range(1, 7):
        X[f"dow{d}"] = (df.dow == d).astype(float)
    for c in ZINB_BIN:
        X[c] = df[c].astype(float)
    cont = pd.DataFrame({"log_sn4": np.log1p(df.sn4), "log_rm28": np.log1p(df.rm28),
                         "log_rm365": np.log1p(df.rm365), "tmax": df.t_max}, index=df.index)
    X["tmax_na"] = cont.tmax.isna().astype(float)
    if st is None:  # estadísticos SOLO del entrenamiento del fold
        st = {"med": cont.median(), "mu": cont.fillna(cont.median()).mean(), "sd": cont.fillna(cont.median()).std() + 1e-9}
    cont = (cont.fillna(st["med"]) - st["mu"]) / st["sd"]
    X = pd.concat([X, cont], axis=1)
    X.insert(0, "const", 1.0)
    Xi = X[["const", "log_rm28"]]
    return X, Xi, st


def fit_predict_zinb(p, fold_row):
    """ZINB (MLE) arrancando desde el NB-MLE con las mismas variables. Si la inflación colapsa (no converge o
    log-verosimilitud no finita: no hay exceso de ceros respecto a la NB), se usa su límite NB-MLE."""
    from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
    from statsmodels.discrete.discrete_model import NegativeBinomialP

    tr, te = split(p, fold_row)
    q = np.zeros((len(te), len(U.QUANTILES)))
    mean = np.zeros(len(te))
    n_reduced = 0
    for s in U.SERIES:
        a, mt = tr[tr.series == s], (te.series == s).values
        b = te[mt]
        X, Xi, st = _zinb_design(a)
        Xt, Xit, _ = _zinb_design(b, st)
        nb = NegativeBinomialP(a.y.values, X, p=2).fit(method="newton", maxiter=200, disp=0)
        ok = False
        try:
            start = np.concatenate([[-2.0, 0.0], np.asarray(nb.params)])
            res = ZeroInflatedNegativeBinomialP(a.y.values, X, exog_infl=Xi, p=2).fit(
                start_params=start, method="bfgs", maxiter=500, disp=0)
            K = int(max(a.y.max(), 5) * 3 + 10)
            pr = np.asarray(res.predict(Xt, exog_infl=Xit, which="prob", y_values=np.arange(K)))
            mu = np.asarray(res.predict(Xt, exog_infl=Xit, which="mean"))
            ok = (res.mle_retvals.get("converged", False) and np.isfinite(res.llf) and res.llf >= nb.llf - 1e-6
                  and np.isfinite(pr).all() and np.isfinite(mu).all())
        except Exception:  # noqa: BLE001
            ok = False
        if ok:
            cdf = np.cumsum(pr / pr.sum(axis=1, keepdims=True), axis=1)
            q[mt] = np.column_stack([(cdf < t).sum(axis=1) for t in U.QUANTILES])
            mean[mt] = mu
        else:
            n_reduced += 1
            mu = np.asarray(nb.predict(Xt))
            q[mt] = U.count_quantiles(mu, float(np.asarray(nb.params)[-1]))
            mean[mt] = mu
    return mean, q, te, n_reduced


# ------------------------------------------------------------------ Chronos (zero-shot y fine-tuned)
CHRONOS_CTX = 1024
COVARIATES_C2 = ["dow", "month", "doy_s1", "doy_c1", "weekend", "covid_lockdown"] + P.HOLIDAY + P.PAYDAY + P.CLIMATE


def _c2_frame(p, cov, start, end):
    wide = p.pivot(index="date", columns="series", values="y")
    rows = []
    for s in U.SERIES:
        c = cov.loc[start:end].copy()
        c["target"] = wide.loc[start:end, s].reindex(c.index)
        c["id"], c["timestamp"] = s, c.index
        rows.append(c.dropna(subset=["target"]))
    return pd.concat(rows, ignore_index=True)


def finetune_c2(base_pipe, p, cov, year: int):
    import torch
    from chronos.chronos2 import preprocess as pp

    torch.manual_seed(U.SEED)
    df = _c2_frame(p, cov, U.TRAIN_START, pd.Timestamp(f"{year - 1}-12-31"))
    inp = pp.from_data_frame(df, target_columns=["target"], prediction_length=U.HORIZON,
                             known_covariates_names=COVARIATES_C2, id_column="id", timestamp_column="timestamp")
    return base_pipe.fit(inp, prediction_length=U.HORIZON, output_dir=str(MODELS_DIR / f"chronos2_ft_{year}"),
                         remove_printer_callback=True, seed=U.SEED, **FT_CONFIG)


def run_chronos(p, origins, folds, variant: str) -> pd.DataFrame:
    import torch
    from chronos import BaseChronosPipeline

    name = {"bolt": "amazon/chronos-bolt-small", "c2": "amazon/chronos-2", "c2ft": "amazon/chronos-2"}[variant]
    pipe = BaseChronosPipeline.from_pretrained(name, device_map=U.torch_device(), dtype=torch.float32)
    wide = p.pivot(index="date", columns="series", values="y")
    hist = pd.read_csv(U.DATA_PROCESSED / "panel_diario.csv", parse_dates=["fecha"]).set_index("fecha")[U.SERIES]
    hist = hist.loc[:U.TEST_END]
    cov = p.drop_duplicates("date").set_index("date")[COVARIATES_C2].astype(float)
    fold_year = dict(zip(folds.fold, folds.test_start.dt.year))
    pipes = {}
    if variant == "c2ft":
        for y in sorted(set(fold_year.values())):
            t0 = time.time()
            pipes[y] = finetune_c2(pipe, p, cov, y)
            print(f"  fine-tuning {y}: {time.time() - t0:.0f}s", flush=True)
    rows = []
    for (fold, o), grp in origins.groupby(["fold", "origin"]):
        dates = pd.DatetimeIndex(sorted(grp.date.unique()))
        H = len(dates)
        if variant == "bolt":
            ctx = [torch.tensor(hist.loc[:o, s].dropna().values[-CHRONOS_CTX:], dtype=torch.float32) for s in U.SERIES]
            q, _ = pipe.predict_quantiles(ctx, prediction_length=H, quantile_levels=list(U.QUANTILES))
            q = q.numpy()
            for i, s in enumerate(U.SERIES):
                for j, t in enumerate(dates):
                    rows.append((s, t, fold, *q[i, j]))
        else:
            pp_ = pipes[fold_year[fold]] if variant == "c2ft" else pipe
            start = o - pd.Timedelta(days=CHRONOS_CTX - 1)
            ctx = _c2_frame(p, cov, start, o)
            fut = pd.concat([cov.loc[dates].assign(id=s, timestamp=dates) for s in U.SERIES], ignore_index=True)
            out = pp_.predict_df(ctx, future_df=fut, id_column="id", timestamp_column="timestamp", target="target",
                                 prediction_length=H, quantile_levels=list(U.QUANTILES))
            for _, r in out.iterrows():
                rows.append((r["id"], r["timestamp"], fold, *[r[str(t)] for t in U.QUANTILES]))
    df = pd.DataFrame(rows, columns=["series", "date", "fold"] + QCOLS)
    df = df.merge(p[["series", "date", "y"]], on=["series", "date"], how="inner")
    q = np.sort(np.clip(df[QCOLS].values, 0, None), axis=1)
    df[QCOLS] = q
    df["mean"] = q.mean(axis=1)  # media aproximada por el promedio de cuantiles (Chronos no da la media)
    df["model"] = {"bolt": "Chronos-Bolt", "c2": "Chronos-2", "c2ft": "Chronos-2-FT"}[variant]
    return df[["series", "date", "y", "model", "fold", "mean"] + QCOLS]


# ------------------------------------------------------------------ otros modelos fundacionales (zero-shot)
TSFM_CTX = 1024


def run_tsfm(p, origins, folds, variant: str) -> pd.DataFrame:
    """TimesFM 2.5 (google/timesfm-2.5-200m-pytorch) y TiRex (NX-AI/TiRex), zero-shot y univariados: contexto de
    hasta 1024 días de la propia serie hasta el origen; deciles 0.1..0.9 del modelo."""
    import torch

    dev = U.torch_device()
    hist = pd.read_csv(U.DATA_PROCESSED / "panel_diario.csv", parse_dates=["fecha"]).set_index("fecha")[U.SERIES]
    hist = hist.loc[:U.TEST_END]
    if variant == "timesfm":
        import timesfm

        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
        model.compile(timesfm.ForecastConfig(max_context=TSFM_CTX, max_horizon=128, normalize_inputs=True,
                                             use_continuous_quantile_head=True, force_flip_invariance=True,
                                             infer_is_positive=True, fix_quantile_crossing=True))
    else:
        from tirex import load_model

        model = load_model("NX-AI/TiRex", device=dev)
    rows = []
    for (fold, o), grp in origins.groupby(["fold", "origin"]):
        dates = pd.DatetimeIndex(sorted(grp.date.unique()))
        H = len(dates)
        ctx = [hist.loc[:o, s].dropna().values[-TSFM_CTX:].astype(float) for s in U.SERIES]
        if variant == "timesfm":
            _, qq = model.forecast(horizon=H, inputs=[c.copy() for c in ctx])
            q = np.asarray(qq)[:len(U.SERIES), :H, 1:10]  # columna 0 = media; 1..9 = deciles
        else:
            L = min(len(c) for c in ctx)
            x = torch.tensor(np.stack([c[-L:] for c in ctx]), dtype=torch.float32)
            qq, _ = model.forecast(context=x, prediction_length=H)
            q = qq.detach().cpu().numpy()[:, :H, :]
        for i, s in enumerate(U.SERIES):
            for j, t in enumerate(dates):
                rows.append((s, t, fold, *q[i, j]))
    df = pd.DataFrame(rows, columns=["series", "date", "fold"] + QCOLS)
    df = df.merge(p[["series", "date", "y"]], on=["series", "date"], how="inner")
    q = np.sort(np.clip(df[QCOLS].values, 0, None), axis=1)
    df[QCOLS] = q
    df["mean"] = q.mean(axis=1)
    df["model"] = {"timesfm": "TimesFM-2.5", "tirex": "TiRex"}[variant]
    return df[["series", "date", "y", "model", "fold", "mean"] + QCOLS]


# ------------------------------------------------------------------ combinación ponderada (stacking de cuantiles)
def _stack_weights(Q: np.ndarray, y: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Pesos en el símplex que minimizan la pérdida pinball escalada (equivale al sCRPS) de la mezcla lineal de
    cuantiles, resuelto EXACTAMENTE como programa lineal (constrained quantile regression averaging, Wang et al.
    2018): min sum_i,tau [tau u+ + (1-tau) u-] / escala_i  s.a.  sum_m w_m q_imtau + u+ - u- = y_i,  w >= 0,
    sum w = 1. La solución exacta da ceros exactos (un optimizador continuo deja pesos ~1e-5 que vuelven
    no enteros cuantiles enteros y distorsionan la cobertura). Q: (n, M, K)."""
    from scipy import sparse
    from scipy.optimize import linprog

    n, M, K = Q.shape
    N = n * K
    taus = U.QUANTILES
    A_q = sparse.csr_matrix(Q.transpose(0, 2, 1).reshape(N, M))
    I = sparse.identity(N, format="csr")
    A = sparse.vstack([sparse.hstack([A_q, I, -I]),
                       sparse.hstack([sparse.csr_matrix(np.ones((1, M))), sparse.csr_matrix((1, 2 * N))])]).tocsr()
    b = np.r_[np.repeat(y, K), 1.0]
    wgt = np.repeat(1.0 / scale, K) / n
    c = np.r_[np.zeros(M), np.tile(taus, n) * wgt, np.tile(1 - taus, n) * wgt]
    res = linprog(c, A_eq=A, b_eq=b, bounds=[(0, None)] * (M + 2 * N), method="highs")
    if not res.success:
        raise RuntimeError(f"stacking LP failed: {res.message}")
    w = np.clip(res.x[:M], 0, None)
    w[w < 1e-6] = 0.0
    return w / w.sum()


def stack(preds, folds, scales, members=tuple(U.STACK_MEMBERS), name="Stacked"):
    """Para el fold k, los pesos se estiman SOLO con las predicciones de los folds < k (todas las series a la vez);
    el fold 0 usa pesos iguales. Al promediar cuantiles ordenados con pesos >= 0 el resultado sigue ordenado."""
    keys = ["series", "date", "fold"]
    tabs = {m: preds[preds.model == m].set_index(keys).sort_index() for m in members}
    idx = tabs[members[0]].index
    for m in members[1:]:
        idx = idx.intersection(tabs[m].index)
    Q = np.stack([tabs[m].loc[idx, QCOLS].values for m in members], axis=1)
    MU = np.stack([tabs[m].loc[idx, "mean"].values for m in members], axis=1)
    y = tabs[members[0]].loc[idx, "y"].values
    fold = idx.get_level_values("fold").values
    ser = idx.get_level_values("series").values
    sc = np.array([scales[(f, s)] for f, s in zip(fold, ser)])
    out_q, out_mu, wrows = np.zeros((len(idx), len(QCOLS))), np.zeros(len(idx)), []
    for k in sorted(folds.fold):
        cur, past = fold == k, fold < k
        w = np.full(len(members), 1 / len(members)) if not past.any() else _stack_weights(Q[past], y[past], sc[past])
        out_q[cur] = np.einsum("nmk,m->nk", Q[cur], w)
        out_mu[cur] = MU[cur] @ w
        wrows.append({"fold": k, **dict(zip(members, w))})
    pd.DataFrame(wrows).to_csv(U.RESULTS / "stack_weights.csv", index=False)
    out = pd.DataFrame(index=idx).reset_index()
    out["y"], out["mean"], out["model"] = y, out_mu, name
    out[QCOLS] = np.sort(np.clip(out_q, 0, None), axis=1)
    return out[["series", "date", "y", "model", "fold", "mean"] + QCOLS]


# ------------------------------------------------------------------ post-procesamiento
def ensemble(preds, members=("LGBM-NB", "Chronos-2"), name="Ensemble"):
    keys = ["series", "date", "fold"]
    a = preds[preds.model == members[0]].set_index(keys).sort_index()
    b = preds[preds.model == members[1]].set_index(keys).reindex(a.index)
    out = a[["y"]].copy()
    out["mean"] = (a["mean"] + b["mean"]) / 2
    out[QCOLS] = (a[QCOLS].values + b[QCOLS].values) / 2
    out["model"] = name
    return out.reset_index()[["series", "date", "y", "model", "fold", "mean"] + QCOLS]


def cqr(preds, model, folds, name=None):
    """CQR secuencial: para cada fold usa como calibración todos los folds anteriores de la misma serie."""
    d = preds[preds.model == model]
    out = []
    for _, f in folds.iterrows():
        cur = d[d.fold == f.fold].copy()
        for s in U.SERIES:
            m = (cur.series == s).values
            cal = d[(d.fold < f.fold) & (d.series == s)]
            cur.loc[m, QCOLS] = U.conformalize(cur.loc[m, QCOLS].values, cal[QCOLS].values, cal.y.values)
        out.append(cur)
    r = pd.concat(out, ignore_index=True)
    r["model"] = name or f"{model}+CQR"
    return r.drop(columns="nb_alpha", errors="ignore")


# ------------------------------------------------------------------ evaluación
def train_scales(p, folds) -> dict:
    sc = {}
    for _, f in folds.iterrows():
        tr = p[p.date <= f.train_end]
        for s in U.SERIES:
            sc[(f.fold, s)] = U.snaive_scale(tr[tr.series == s].sort_values("date").y.values, 7)
    return sc


def score(preds: pd.DataFrame, scales: dict) -> pd.DataFrame:
    rows = []
    for (m, f, s), g in preds.groupby(["model", "fold", "series"]):
        r = U.evaluate_block(g.y.values, g[QCOLS].values, g["mean"].values, scales[(f, s)])
        rows.append({"model": m, "fold": f, "series": s, "n": len(g), **r})
    return pd.DataFrame(rows)


def run_models(p, folds, origins) -> pd.DataFrame:
    out, fallbacks = [], []
    t0 = time.time()
    for _, f in folds.iterrows():
        tr, te = split(p, f)
        mu = te["sn4"].fillna(te["rm28"]).clip(lower=1e-3).values
        out.append(package(te, "SN-Poisson", f.fold, mu, U.count_quantiles(mu, 0.0)).assign(nb_alpha=0.0))
        mu, alphas, te_g = fit_predict_glm(p, f)
        out.append(package(te_g, "GLM-Poisson", f.fold, mu, U.count_quantiles(mu, 0.0)).assign(nb_alpha=0.0))
        out.append(package(te_g, "GLM-NB", f.fold, mu, nb_quantiles_by_series(te_g.series, mu, alphas))
                   .assign(nb_alpha=te_g.series.map(alphas).values))
        zmean, zq, te_z, nfb = fit_predict_zinb(p, f)
        out.append(package(te_z, "GLM-ZINB", f.fold, zmean, zq))
        fallbacks.append({"fold": f.fold, "zinb_reduced_to_nb_series": nfb})
        mu, q, alphas, model, te_l = fit_predict_lgbm(p, f, P.ALL_FEATURES)
        model.booster_.save_model(str(MODELS_DIR / f"lgbm_full_fold{f.fold}.txt"))
        pd.Series(alphas).to_csv(MODELS_DIR / f"lgbm_full_fold{f.fold}_alpha.csv")
        out.append(package(te_l, "LGBM-NB", f.fold, mu, q).assign(nb_alpha=te_l.series.map(alphas).values))
        print(f"fold {f.fold} listo ({time.time() - t0:.0f}s; ZINB reducido a NB en {nfb} series)", flush=True)
    pd.DataFrame(fallbacks).to_csv(U.RESULTS / "zinb_fallbacks.csv", index=False)
    preds = pd.concat(out, ignore_index=True)
    for v in ("bolt", "c2", "c2ft"):
        t1 = time.time()
        preds = pd.concat([preds, run_chronos(p, origins, folds, v)], ignore_index=True)
        print(f"Chronos {v} listo ({time.time() - t1:.0f}s)", flush=True)
    for v in ("timesfm", "tirex"):
        t1 = time.time()
        preds = pd.concat([preds, run_tsfm(p, origins, folds, v)], ignore_index=True)
        print(f"{v} listo ({time.time() - t1:.0f}s)", flush=True)
    preds = pd.concat([preds, ensemble(preds)], ignore_index=True)
    preds = pd.concat([preds, stack(preds, folds, train_scales(p, folds))], ignore_index=True)
    print(pd.read_csv(U.RESULTS / "stack_weights.csv").round(3).to_string(index=False), flush=True)
    preds = pd.concat([preds, cqr(preds, "LGBM-NB", folds), cqr(preds, "Ensemble", folds),
                       cqr(preds, "Stacked", folds)], ignore_index=True)
    return preds


# ------------------------------------------------------------------ tabla y figura principales
def table_02(scores: pd.DataFrame) -> None:
    """Tabla principal agrupada por familia de modelos (un encabezado en cursiva por familia)."""
    metrics = ["sCRPS", "CRPS", "MASE", "RMSE", "Coverage80", "Width80"]
    rows, tex_rows = [], []
    best = {m: (scores.groupby("model")[m].mean().min() if m != "Coverage80"
                else scores.groupby("model")[m].mean().sub(0.8).abs().min()) for m in metrics}
    for model in U.MODEL_ORDER:
        g = scores[scores.model == model]
        r = {"Family": U.MODEL_FAMILY[model], "Model": model}
        tr = {"Model": model + (" (proposed)" if model == U.PROPOSED else "")}
        for m in metrics:
            mean, sd = g[m].mean(), g[m].std()
            lo, hi = U.bootstrap_ci(g[m].values)
            r.update({f"{m}_mean": mean, f"{m}_sd": sd, f"{m}_ci_low": lo, f"{m}_ci_high": hi})
            crit = abs(mean - 0.8) if m == "Coverage80" else mean
            cell = f"{mean:.3f} $\\pm$ {sd:.3f}"
            tr[m] = f"\\textbf{{{cell}}}" if np.isclose(crit, best[m]) else cell
        rows.append(r); tex_rows.append(tr)
    tex = pd.DataFrame(tex_rows)
    tex.columns = ["Model", "sCRPS", "CRPS", "MASE", "RMSE", "Cov. 80%", "Width 80%"]
    U.save_table(pd.DataFrame(rows), "table_02_main_results_cv", tex)
    # separadores por familia: una fila de encabezado en cursiva antes de cada grupo
    f = U.TABLES / "table_02_main_results_cv.tex"
    lines = f.read_text(encoding="utf-8").splitlines()
    out, prev, ncol = [], None, tex.shape[1]
    body = False
    for ln in lines:
        if ln.strip() == r"\midrule":
            body = True
            continue
        if body and not ln.startswith(r"\bottomrule") and "&" in ln:
            name = ln.split("&")[0].strip().replace(" (proposed)", "").replace(r"\_", "_")
            fam = U.MODEL_FAMILY.get(name)
            if fam != prev:
                out.append(r"\midrule")
                out.append(rf"\multicolumn{{{ncol}}}{{@{{}}l}}{{\textit{{{fam}}}}} \\")
                prev = fam
        out.append(ln)
    f.write_text("\n".join(out) + "\n", encoding="utf-8")


def fig_02(scores: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig = plt.figure(figsize=(U.COL2, 7.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], width_ratios=[1, 1.1], hspace=0.42, wspace=0.42)
    order = U.MODEL_ORDER
    # (a) sCRPS por bloque
    ax = fig.add_subplot(gs[0, 0])
    sns.boxplot(data=scores, x="sCRPS", y="model", order=order, hue="model", palette=U.MODEL_COLORS,
                showfliers=False, width=0.6, linewidth=0.7, ax=ax, legend=False)
    means = scores.groupby("model").sCRPS.mean().reindex(order)
    ax.scatter(means.values, range(len(order)), marker="D", s=12, color="white", edgecolor="black", zorder=5)
    ax.set_xlabel("Scaled CRPS (lower is better)"); ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=6.5)
    U.panel_label(ax, "a", x=-0.46)
    # (b) exactitud frente a calibración: sCRPS medio (IC 95 %) vs cobertura del 80 % (IC 95 %)
    ax = fig.add_subplot(gs[0, 1])
    ax.axhspan(0.78, 0.82, color="0.92", zorder=0)
    ax.axhline(0.8, color="0.3", ls="--", lw=0.8, zorder=1)
    for m in order:
        g = scores[scores.model == m]
        x, yv = g.sCRPS.mean(), g.Coverage80.mean()
        xl, xh = U.bootstrap_ci(g.sCRPS.values)
        yl, yh = U.bootstrap_ci(g.Coverage80.values)
        ax.errorbar(x, yv, xerr=[[x - xl], [xh - x]], yerr=[[yv - yl], [yh - yv]], fmt="none",
                    ecolor=U.MODEL_COLORS[m], elinewidth=0.6, alpha=0.7, zorder=2)
        ax.scatter(x, yv, marker=U.MODEL_MARKERS[m], s=34 if m == U.PROPOSED else 22, color=U.MODEL_COLORS[m],
                   edgecolor="black", linewidth=0.4, zorder=3, label=m)
    ax.set_xlabel("Scaled CRPS (lower is better)")
    ax.set_ylabel("Coverage of 80% interval")
    ax.text(0.99, 0.02, "under-coverage (too narrow)", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6, color="0.35")
    ax.text(0.99, 0.98, "over-coverage (too wide)", transform=ax.transAxes, ha="right", va="top",
            fontsize=6, color="0.35")
    ax.legend(fontsize=5.2, ncol=1, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5),
              handletextpad=0.3, borderaxespad=0.2)
    U.panel_label(ax, "b", x=-0.24)
    # (c) sCRPS relativo al baseline por delito
    ax = fig.add_subplot(gs[1, :])
    piv = scores.pivot_table(index="series", columns="model", values="sCRPS", aggfunc="mean")
    rel = piv.div(piv["SN-Poisson"], axis=0)[order[1:]]
    rel = rel.loc[U.SERIES]
    rel.index = [U.SERIES_LABEL[s] for s in rel.index]
    sns.heatmap(rel, cmap="cividis_r", annot=True, fmt=".2f", annot_kws={"fontsize": 5.2},
                cbar_kws={"label": "sCRPS relative to SN-Poisson", "shrink": 0.85}, ax=ax, linewidths=0.3)
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=35, labelsize=6.5); ax.tick_params(axis="y", labelsize=7)
    for lbl in ax.get_xticklabels():
        lbl.set_horizontalalignment("right")
    U.panel_label(ax, "c", x=-0.17)
    U.save_figure(fig, "fig_02_main_performance_comparison")


def main() -> None:
    p, folds, origins = load_inputs()
    preds = run_models(p, folds, origins)
    preds.to_parquet(U.RESULTS / "oof_predictions_all_folds.parquet", index=False)
    tf = set(test_folds(folds).fold)
    preds_test = preds[preds.fold.isin(tf)]
    preds_test.to_parquet(U.RESULTS / "oof_predictions.parquet", index=False)
    for m, g in preds.groupby("model"):
        g.to_parquet(U.RESULTS / f"predictions_{m}.parquet", index=False)
    scales = train_scales(p, folds)
    pd.Series({f"{k[0]}|{k[1]}": v for k, v in scales.items()}).to_csv(U.RESULTS / "train_scales.csv")
    scores = score(preds_test, scales)
    scores.to_csv(U.RESULTS / "fold_scores.csv", index=False)
    table_02(scores)
    fig_02(scores)
    print(scores.groupby("model")[["sCRPS", "CRPS", "MASE", "RMSE", "Coverage80", "Width80"]].mean()
          .reindex(U.MODEL_ORDER).round(4))


if __name__ == "__main__":
    main()
