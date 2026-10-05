"""``act9-predecir``: usa el modelo guardado con partidos nuevos (Deployment).

Recibe un CSV con las estadísticas de los partidos (con o sin la columna
``result``) y escribe otro con la predicción y, si el modelo las da, las
probabilidades de cada clase. Es el comando que correría un proceso por lotes
en producción, por ejemplo cada jornada.

    act9-predecir --entrada jornada.csv --salida predicciones.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..artefactos import cargar_modelo
from ..data import CATEGORICAL_FEATURES, FEATURES, TARGET
from ._comun import DIR_DATOS, DIR_MODELOS, DIR_REPORTES, etapa


def main(argv: list[str] | None = None, prog: str = "act9-predecir") -> int:
    parser = argparse.ArgumentParser(
        prog=prog, description="Predice el resultado de partidos con el modelo guardado."
    )
    parser.add_argument("--modelo", metavar="JOBLIB",
                        default=str(DIR_MODELOS / "modelo.joblib"))
    parser.add_argument("--entrada", metavar="CSV", default=str(DIR_DATOS / "test.csv"),
                        help="Partidos a predecir (por defecto, el conjunto de prueba).")
    parser.add_argument("--salida", metavar="CSV",
                        default=str(DIR_REPORTES / "predicciones.csv"))
    args = parser.parse_args(argv)

    for ruta in (args.modelo, args.entrada):
        if not Path(ruta).exists():
            etapa(prog, f"ERROR: no existe {ruta}")
            return 1

    modelo, meta = cargar_modelo(args.modelo)
    df = pd.read_csv(args.entrada).dropna(how="all")
    faltan = [c for c in FEATURES if c not in df.columns]
    if faltan:
        etapa(prog, f"ERROR: al CSV le faltan las columnas {faltan}")
        return 1
    etapa(prog, f"{len(df)} partidos leidos de {args.entrada}; "
                f"modelo {meta.get('modelo', '?')}")

    salida = df[CATEGORICAL_FEATURES].copy()
    salida["prediccion"] = modelo.predict(df[FEATURES])
    # Pipeline.predict_proba solo existe si el último paso lo tiene (un SVC sin
    # probability=True no lo tiene).
    if hasattr(modelo, "predict_proba"):
        proba = modelo.predict_proba(df[FEATURES])
        for i, clase in enumerate(modelo.classes_):
            salida[f"prob_{clase.replace(' ', '_').lower()}"] = proba[:, i].round(3)
    if TARGET in df.columns:
        salida["real"] = df[TARGET]
        aciertos = int((salida["prediccion"] == salida["real"]).sum())
        etapa(prog, f"aciertos: {aciertos}/{len(salida)}")

    ruta = Path(args.salida)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    salida.to_csv(ruta, index=False)
    conteo = "  ".join(f"{k}={v}" for k, v in salida["prediccion"].value_counts().items())
    etapa(prog, f"predicciones: {conteo}")
    etapa(prog, f"escrito {ruta}")
    print()
    print(salida.head(5).to_string(index=False))
    return 0
