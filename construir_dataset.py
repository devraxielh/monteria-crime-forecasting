from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

INICIO_REGIMEN = pd.Timestamp("2013-01-01")  # cambio de registro: los conteos se multiplican x3-x5
ANIO_PROVISIONAL = 2026                       # Policía: datos provisionales hasta marzo del año siguiente
DROP = ["cod_depto", "departamento", "cod_muni", "municipio"]

# --- MinDefensa / SIEDCO -------------------------------------------------------
frames = []
for f in sorted(RAW.glob("*.csv")):
    if f.stem.startswith("medlegal"):
        continue
    d = pd.read_csv(f).drop(columns=DROP, errors="ignore").dropna(axis=1, how="all")
    d["fecha_hecho"] = pd.to_datetime(d["fecha_hecho"])
    d["delito"] = f.stem
    frames.append(d)
eventos = pd.concat(frames, ignore_index=True).sort_values(["fecha_hecho", "delito"])
eventos.to_csv(OUT / "eventos_mindefensa.csv", index=False)

fin = eventos["fecha_hecho"].max()
idx = pd.date_range(eventos["fecha_hecho"].min(), fin, freq="D", name="fecha")
diario = (eventos.pivot_table(index="fecha_hecho", columns="delito", values="cantidad", aggfunc="sum")
          .reindex(idx).fillna(0).astype(int))
# delitos informáticos no existen como serie antes de jul-2013
diario.loc[diario.index < "2013-07-01", "delitos_informaticos"] = pd.NA
# hurto a personas no tiene ningún registro en 2008 -> faltante, no cero
diario.loc[diario.index.year == 2008, "hurto_personas"] = pd.NA

delitos = list(diario.columns)
diario["total"] = diario[delitos].sum(axis=1, min_count=1)
cal = pd.DataFrame(index=idx)
cal["anio"], cal["mes"], cal["dia"] = idx.year, idx.month, idx.day
cal["dia_semana"] = idx.day_of_week          # 0 = lunes
cal["semana_iso"] = idx.isocalendar().week.astype(int).values
cal["fin_de_semana"] = (idx.day_of_week >= 5).astype(int)
cal["quincena"] = idx.day.isin([1, 2, 15, 16, 30, 31]).astype(int)  # días de pago típicos
cal["regimen_post2013"] = (idx >= INICIO_REGIMEN).astype(int)
cal["provisional"] = (idx.year >= ANIO_PROVISIONAL).astype(int)
diario = cal.join(diario).reset_index()
diario.to_csv(OUT / "panel_diario.csv", index=False)

mensual = (diario.set_index("fecha")[delitos + ["total"]]
           .resample("MS").sum(min_count=1))
mensual = mensual[mensual.index <= fin.replace(day=1)]
mensual.insert(0, "anio", mensual.index.year)
mensual.insert(1, "mes", mensual.index.month)
mensual["regimen_post2013"] = (mensual.index >= INICIO_REGIMEN).astype(int)
mensual["provisional"] = (mensual.index.year >= ANIO_PROVISIONAL).astype(int)
mensual.reset_index().to_csv(OUT / "panel_mensual.csv", index=False)

# --- Medicina Legal --------------------------------------------------------------
ml = pd.read_csv(RAW / "medlegal_homicidios_victimas.csv")
ml = ml.replace(r"(?i)^\s*sin informaci[oó]n\s*$", pd.NA, regex=True)
ml = ml.dropna(axis=1, how="all").rename(columns={"a_o_del_hecho": "anio_del_hecho"})
ml.to_csv(OUT / "medlegal_homicidios.csv", index=False)

print(f"eventos: {len(eventos)} | diario: {len(diario)} días x {len(delitos)} delitos "
      f"| mensual: {len(mensual)} meses | medlegal: {ml.shape}")
