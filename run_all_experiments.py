from __future__ import annotations

import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

S = Path(__file__).resolve().parent
TABLES = ["table_01_cohort_summary", "table_02_main_results_cv", "table_03_ablation_study",
          "table_04_statistical_significance", "table_05_exogenous_effects", "table_06_multiscale",
          "table_07_dlnm_heat", "table_08_descriptive_peaks", "table_09_forecastability", "table_10_adaptive_conformal", "table_11_alerts", "table_12_climate_sensitivity_forecast", "table_13_climate_sensitivity_association", "table_14_count_evaluation", "table_S1_robustness_excl_shifted"]
FIGS = ["fig_01_data_distribution", "fig_02_main_performance_comparison", "fig_03_ablation_and_sensitivity",
        "fig_04_explainability_shap", "fig_05_error_calibration_case_study", "fig_06_explainability_robustness",
        "fig_07_horizon_and_weekly", "fig_08_dlnm_heat", "fig_09_temporal_patterns", "fig_10_novel_analyses", "fig_11_climate_sensitivity", "fig_12_count_evaluation",
        "fig_13_critical_difference", "fig_14_study_area"]
STEPS = ["01_data_splitting_and_prep.py", "02_baseline_and_sota_models.py", "03_ablation_and_sensitivity.py",
         "04_statistical_tests.py", "05_explainability_and_error.py", "06_multiscale_horizon.py",
         "07_dlnm_heat.py", "08_descriptive_patterns.py", "09_novel_analyses.py", "10_climate_sensitivity.py", "11_count_evaluation.py", "12_study_area_map.py"]


def snapshot() -> Path | None:
    src = S / "tables"
    if not all((src / f"{t}.csv").exists() for t in TABLES):
        return None
    dst = S / "_previous_tables"
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst)
    return dst


def compare(prev: Path) -> list[str]:
    diffs = []
    for t in TABLES:
        a, b = pd.read_csv(prev / f"{t}.csv"), pd.read_csv(S / "tables" / f"{t}.csv")
        num = a.select_dtypes("number").columns
        if a.shape != b.shape or not np.allclose(a[num].values, b[num].values, rtol=1e-9, atol=1e-12, equal_nan=True):
            diffs.append(t)
    return diffs


def check_no_titles() -> list[str]:
    bad = []
    pat = re.compile(r"(set_title|plt\.title|suptitle)\(\s*([^)\s][^)]*)\)")
    for f in S.glob("*.py"):
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            for m in pat.finditer(line):
                arg = m.group(2).strip()
                if arg not in ('""', "''") and f.name != "run_all_experiments.py":
                    bad.append(f"{f.name}:{i}: {line.strip()}")
    return bad


def main() -> int:
    t0 = time.time()
    prev = snapshot()
    for d in ("results", "tables", "figures"):
        shutil.rmtree(S / d, ignore_errors=True)
    for step in STEPS:
        t = time.time()
        print(f"\n=== {step}", flush=True)
        r = subprocess.run([sys.executable, "-P", str(S / step)], cwd=S.parent)
        if r.returncode != 0:
            print(f"FALLÓ {step}"); return 1
        print(f"--- {step}: {time.time() - t:.0f}s", flush=True)

    ok = True
    missing = [f"{t}.{e}" for t in TABLES for e in ("csv", "tex") if not (S / "tables" / f"{t}.{e}").exists()]
    missing += [f"{f}.{e}" for f in FIGS for e in ("pdf", "png") if not (S / "figures" / f"{f}.{e}").exists()]
    print("\n[check] archivos:", "OK" if not missing else f"FALTAN {missing}"); ok &= not missing
    low = [f for f in FIGS if (S / "figures" / f"{f}.png").exists()
           and min(Image.open(S / "figures" / f"{f}.png").info.get("dpi", (0, 0))) < 599]
    print("[check] DPI >= 600:", "OK" if not low else f"BAJO {low}"); ok &= not low
    bad = check_no_titles()
    print("[check] NO TITLES:", "OK" if not bad else bad); ok &= not bad
    if prev is not None:
        diffs = compare(prev)
        print("[check] reproducibilidad vs. ejecución previa:", "OK (cifras idénticas)" if not diffs else f"DIFIEREN {diffs}")
        ok &= not diffs
    else:
        print("[check] reproducibilidad: sin ejecución previa para comparar")
    print(f"\nTiempo total: {(time.time() - t0) / 60:.1f} min. Resultado: {'TODO OK' if ok else 'HAY FALLAS'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
