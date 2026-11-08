from __future__ import annotations

import os
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy import stats  # noqa: E402

import sys  # noqa: E402

for _s in (sys.stdout, sys.stderr):  # consola de Windows (cp1252) no imprime ≥, °, −
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

# ---------------------------------------------------------------- semillas
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
try:
    import torch

    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
except ImportError:  # pragma: no cover
    pass

# ---------------------------------------------------------------- rutas
SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_RAW = ROOT / "data" / "raw"
RESULTS = SCRIPTS / "results"
TABLES = SCRIPTS / "tables"
FIGURES = SCRIPTS / "figures"
for _d in (RESULTS, TABLES, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- diseño experimental
SERIES = [  # 11 delitos con >= 0.15 casos/día en promedio (2013-2025)
    "hurto_personas", "lesiones_personales", "violencia_intrafamiliar", "delitos_informaticos",
    "hurto_comercio", "hurto_residencias", "hurto_vehiculos", "delitos_sexuales",
    "homicidio", "homicidio_transito", "extorsion",
]
SERIES_LABEL = {
    "hurto_personas": "Theft (persons)", "lesiones_personales": "Assault",
    "violencia_intrafamiliar": "Domestic violence", "delitos_informaticos": "Cybercrime",
    "hurto_comercio": "Theft (commerce)", "hurto_residencias": "Burglary",
    "hurto_vehiculos": "Vehicle theft", "delitos_sexuales": "Sexual offences",
    "homicidio": "Homicide", "homicidio_transito": "Traffic homicide", "extorsion": "Extortion",
}
TRAIN_START = pd.Timestamp("2015-06-01")   # primer mes completo con clima IDEAM
CAL_START = pd.Timestamp("2022-01-01")     # folds 2022: solo calibración conformal (no se evalúan)
TEST_START = pd.Timestamp("2023-01-01")
TEST_END = pd.Timestamp("2025-12-31")      # 2026 excluido: cifras provisionales
HORIZON = 7                                # días; orígenes semanales no solapados
MIN_LAG = HORIZON                          # modelo directo: rezagos >= horizonte
QUANTILES = np.round(np.arange(0.1, 0.91, 0.1), 2)  # rango soportado por Chronos-Bolt
PI_LOW, PI_HIGH = 0.1, 0.9                 # intervalo de predicción del 80 %
N_BOOT = 2000
ALPHA = 0.05

BASE_MODELS = ["SN-Poisson", "GLM-Poisson", "GLM-NB", "GLM-ZINB", "LGBM-NB",
               "Chronos-Bolt", "Chronos-2", "Chronos-2-FT", "TimesFM-2.5", "TiRex"]
# Ensemble = LGBM-NB + Chronos-2 (promedio simple de cuantiles); Stacked = combinación lineal de cuantiles con pesos
# que minimizan el sCRPS en los folds anteriores (ver 02); +CQR = calibración conformal por nivel de cuantil.
POST_MODELS = ["LGBM-NB+CQR", "Ensemble", "Ensemble+CQR", "Stacked", "Stacked+CQR"]
MODEL_ORDER = BASE_MODELS + POST_MODELS
PROPOSED = "Stacked+CQR"
STACK_MEMBERS = ["GLM-NB", "GLM-ZINB", "LGBM-NB", "Chronos-2", "Chronos-2-FT", "TimesFM-2.5", "TiRex"]
MODEL_FAMILY = {"SN-Poisson": "Naive benchmark", "GLM-Poisson": "Count regression", "GLM-NB": "Count regression",
                "GLM-ZINB": "Count regression", "LGBM-NB": "Gradient boosting",
                "Chronos-Bolt": "Foundation models", "Chronos-2": "Foundation models",
                "Chronos-2-FT": "Foundation models", "TimesFM-2.5": "Foundation models", "TiRex": "Foundation models",
                "LGBM-NB+CQR": "Combinations and calibration", "Ensemble": "Combinations and calibration",
                "Ensemble+CQR": "Combinations and calibration", "Stacked": "Combinations and calibration",
                "Stacked+CQR": "Combinations and calibration"}
SHIFTED_SERIES = ["delitos_sexuales", "delitos_informaticos"]  # cambios de régimen fuertes train -> test

# ---------------------------------------------------------------- estilo Q1
OKABE_ITO = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
MODEL_COLORS = {"SN-Poisson": "#999999", "GLM-Poisson": "#9ecae1", "GLM-NB": "#56B4E9", "GLM-ZINB": "#0072B2",
                "LGBM-NB": "#66C2A4", "Chronos-Bolt": "#F0C05A", "Chronos-2": "#E69F00", "Chronos-2-FT": "#D55E00",
                "TimesFM-2.5": "#8C6D31", "TiRex": "#B07AA1",
                "LGBM-NB+CQR": "#009E73", "Ensemble": "#CC79A7", "Ensemble+CQR": "#7B2D8E",
                "Stacked": "#F4A582", "Stacked+CQR": "#B2182B"}
MODEL_MARKERS = dict(zip(MODEL_ORDER, ["o", "s", "D", "d", "P", "^", "v", "<", ">", "p", "X", "h", "H", "8", "*"]))
COL1, COL2 = 3.5, 7.0


def set_style() -> None:
    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 8, "legend.frameon": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600, "figure.dpi": 150,
        "axes.spines.top": False, "axes.spines.right": False,
    })


set_style()


def panel_label(ax, label: str, x: float = -0.12, y: float = 1.05) -> None:
    ax.text(x, y, f"({label})", transform=ax.transAxes, fontweight="bold", fontsize=11, va="bottom")


def save_figure(fig, name: str) -> None:
    """Guarda .pdf y .png (600 DPI) borrando cualquier título residual (regla NO TITLES)."""
    for ax in fig.get_axes():
        ax.set_title("")
    if fig._suptitle is not None:  # noqa: SLF001
        fig._suptitle.set_text("")  # noqa: SLF001
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)


def latex_escape(s) -> str:
    r"""Escapa caracteres especiales de LaTeX fuera de los segmentos matemáticos $...$; respeta los
    caracteres ya escapados (p. ej. \%)."""
    import re
    parts = str(s).split("$")
    for i in range(0, len(parts), 2):  # solo segmentos de texto (índices pares)
        p = re.sub(r"\\(?![%&#_{}$])", lambda m: r"\textbackslash{}", parts[i])
        parts[i] = re.sub(r"(?<!\\)([_%&#])", lambda m: "\\" + m.group(1), p)
    return "$".join(parts)


def save_table(df: pd.DataFrame, name: str, tex_df: pd.DataFrame | None = None,
               column_format: str | None = None) -> None:
    """CSV con valores completos + .tex booktabs (sin entorno table) listo para \\input{}."""
    df.to_csv(TABLES / f"{name}.csv", index=False)
    t = (tex_df if tex_df is not None else df).copy()
    t.columns = [latex_escape(c) for c in t.columns]
    cols = column_format or ("l" + "r" * (t.shape[1] - 1))
    lines = [rf"\begin{{tabular}}{{{cols}}}", r"\toprule", " & ".join(t.columns) + r" \\", r"\midrule"]
    for _, row in t.iterrows():
        lines.append(" & ".join(str(v) for v in row.values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"


# ---------------------------------------------------------------- métricas
def pinball(y: np.ndarray, q: np.ndarray, taus: np.ndarray = QUANTILES) -> np.ndarray:
    """Pérdida cuantílica por observación. y: (n,), q: (n, k)."""
    d = y[:, None] - q
    return np.mean(np.maximum(taus * d, (taus - 1) * d), axis=1)


def crps_from_quantiles(y: np.ndarray, q: np.ndarray, taus: np.ndarray = QUANTILES) -> np.ndarray:
    """Aproximación del CRPS = 2 * pérdida cuantílica media sobre los niveles tau."""
    return 2.0 * pinball(y, q, taus)


def snaive_scale(y_train: np.ndarray, m: int = 7) -> float:
    """MAE in-sample del pronóstico estacional ingenuo (lag m); escala de MASE y sCRPS."""
    y_train = np.asarray(y_train, dtype=float)
    s = np.mean(np.abs(y_train[m:] - y_train[:-m]))
    return float(s) if s > 0 else 1.0


def evaluate_block(y, q, mean, scale) -> dict:
    """Métricas de un bloque (serie x fold). q: cuantiles (n, k) en el orden de QUANTILES."""
    y = np.asarray(y, float)
    med = q[:, list(QUANTILES).index(0.5)]
    lo, hi = q[:, list(QUANTILES).index(PI_LOW)], q[:, list(QUANTILES).index(PI_HIGH)]
    crps = crps_from_quantiles(y, q).mean()
    return {
        "CRPS": crps,
        "sCRPS": crps / scale,
        "MASE": np.mean(np.abs(y - med)) / scale,
        "RMSE": float(np.sqrt(np.mean((y - mean) ** 2))),
        "Coverage80": float(np.mean((y >= lo) & (y <= hi))),
        "Width80": float(np.mean(hi - lo)),
    }


def heat_index_c(t_c, rh):
    """Índice de calor NOAA (Rothfusz + ajustes) en °C a partir de temperatura (°C) y HR (%)."""
    t = np.asarray(t_c, float) * 9 / 5 + 32
    rh = np.clip(np.asarray(rh, float), 0, 100)
    simple = 0.5 * (t + 61.0 + (t - 68.0) * 1.2 + rh * 0.094)
    hi = (-42.379 + 2.04901523 * t + 10.14333127 * rh - 0.22475541 * t * rh - 6.83783e-3 * t ** 2
          - 5.481717e-2 * rh ** 2 + 1.22874e-3 * t ** 2 * rh + 8.5282e-4 * t * rh ** 2 - 1.99e-6 * t ** 2 * rh ** 2)
    adj1 = ((13 - rh) / 4) * np.sqrt(np.clip((17 - np.abs(t - 95.0)) / 17, 0, None))
    hi = np.where((rh < 13) & (t >= 80) & (t <= 112), hi - adj1, hi)
    adj2 = ((rh - 85) / 10) * ((87 - t) / 5)
    hi = np.where((rh > 85) & (t >= 80) & (t <= 87), hi + adj2, hi)
    hi = np.where((simple + t) / 2 < 80, simple, hi)
    return (hi - 32) * 5 / 9


def conformalize(q_target: np.ndarray, q_cal: np.ndarray, y_cal: np.ndarray,
                 taus: np.ndarray = QUANTILES) -> np.ndarray:
    """CQR por nivel (Romano et al., 2019, versión asimétrica): q'_tau = q_tau + c_tau, con c_tau el
    cuantil empírico (corrección de muestra finita) de los residuos y - q_tau en calibración."""
    n = len(y_cal)
    out = q_target.copy().astype(float)
    if n < 30:
        return out
    for j, t in enumerate(taus):
        s = y_cal - q_cal[:, j]
        lvl = min(1.0, t * (n + 1) / n) if t >= 0.5 else max(0.0, (t * (n + 1) - 1) / n)
        out[:, j] = q_target[:, j] + np.quantile(s, lvl, method="higher" if t >= 0.5 else "lower")
    return np.sort(np.clip(out, 0, None), axis=1)


# ---------------------------------------------------------------- distribuciones de conteo
def nb_alpha_moments(y, mu) -> float:
    """Dispersión NB2 por momentos: Var = mu + alpha mu^2."""
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    a = (np.sum((y - mu) ** 2 - y) ) / max(np.sum(mu ** 2), 1e-9)
    return float(max(a, 0.0))


def count_quantiles(mu, alpha: float = 0.0, taus: np.ndarray = QUANTILES) -> np.ndarray:
    """Cuantiles de Poisson (alpha=0) o NB2(mu, alpha)."""
    mu = np.clip(np.asarray(mu, float), 1e-6, None)
    if alpha <= 1e-8:
        return np.column_stack([stats.poisson.ppf(t, mu) for t in taus])
    n = 1.0 / alpha
    p = n / (n + mu)
    return np.column_stack([stats.nbinom.ppf(t, n, p) for t in taus])


# ---------------------------------------------------------------- estadística
def bootstrap_ci(x, n_boot: int = N_BOOT, seed: int = SEED, func=np.mean):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    boots = np.array([func(rng.choice(x, size=len(x), replace=True)) for _ in range(n_boot)])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def rank_biserial(x, y) -> float:
    """Correlación rango-biserial pareada (Kerby): >0 indica x < y (x mejor si es pérdida)."""
    d = np.asarray(y, float) - np.asarray(x, float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def load_module(name: str):
    import importlib
    import sys

    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    return importlib.import_module(name)


# ---------------------------------------------------------------- GPU (Chronos)
def torch_device() -> str:
    """'cuda' si hay GPU NVIDIA disponible (RTX 4090 en la máquina del estudio), si no 'cpu'.
    Activa opciones deterministas de PyTorch/cuDNN para que dos corridas en la misma GPU coincidan."""
    import os

    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        return "cuda"
    return "cpu"
