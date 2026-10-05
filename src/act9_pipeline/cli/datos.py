"""``act9-datos``: primera etapa del pipeline (Data Preparation).

Lee el CSV crudo, verifica que traiga las columnas que el modelo necesita,
lo limpia y lo separa en ``train.csv`` y ``test.csv``. Las etapas siguientes
leen esos archivos, no el CSV crudo.

    act9-datos                                  # CSV empaquetado -> datos/procesados/
    act9-datos --entrada otro.csv --salida tmp/ # cualquier CSV con las mismas columnas
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..data import FEATURES, TARGET, extraer_datos, filtrar_datos, separar_datos
from ._comun import DIR_DATOS, etapa


def main(argv: list[str] | None = None, prog: str = "act9-datos") -> int:
    parser = argparse.ArgumentParser(
        prog=prog, description="Limpia el CSV de partidos y lo separa en train/test."
    )
    parser.add_argument(
        "--entrada", metavar="CSV", default=None,
        help="CSV crudo de partidos (por defecto, el que viene dentro del paquete).",
    )
    parser.add_argument(
        "--salida", metavar="DIR", default=str(DIR_DATOS),
        help=f"Carpeta donde escribir train.csv y test.csv (por defecto {DIR_DATOS}).",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    origen = args.entrada or "CSV empaquetado en act9_pipeline/datasets"
    etapa(prog, f"leyendo {origen}")
    crudo = extraer_datos(args.entrada)

    faltan = [c for c in FEATURES + [TARGET] if c not in crudo.columns]
    if faltan:
        etapa(prog, f"ERROR: al CSV le faltan las columnas {faltan}")
        return 1

    limpio = filtrar_datos(crudo)
    X_train, X_test, y_train, y_test = separar_datos(
        limpio, test_size=args.test_size, seed=args.seed
    )

    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)
    X_train.assign(**{TARGET: y_train}).to_csv(salida / "train.csv", index=False)
    X_test.assign(**{TARGET: y_test}).to_csv(salida / "test.csv", index=False)

    clases = "  ".join(f"{k}={v}" for k, v in limpio[TARGET].value_counts().items())
    etapa(prog, f"crudo: {crudo.shape[0]} filas -> limpio: {len(limpio)} partidos")
    etapa(prog, f"clases: {clases}")
    etapa(prog, f"train={len(X_train)}  test={len(X_test)}  (estratificado, seed={args.seed})")
    etapa(prog, f"escrito {salida / 'train.csv'} y {salida / 'test.csv'}")
    return 0
