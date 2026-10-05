"""``act9-evaluar``: mide el modelo guardado en el conjunto de prueba (Evaluation).

Además de imprimir las métricas, funciona como **compuerta de calidad**: si el
``f1_macro`` queda por debajo de ``--umbral``, el comando termina con código 1.
La función ``main`` devuelve ese número y el ejecutable que generó el
instalador lo usa como código de salida, así que ``make`` o GitHub Actions
detienen el pipeline sin tener que leer la salida.

    act9-evaluar
    act9-evaluar --umbral 0.55
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ..artefactos import cargar_modelo
from ..data import FEATURES, TARGET
from ..evaluation import baseline, evaluar_en_prueba, matriz_confusion
from ._comun import DIR_DATOS, DIR_MODELOS, DIR_REPORTES, banner, etapa


def main(argv: list[str] | None = None, prog: str = "act9-evaluar") -> int:
    parser = argparse.ArgumentParser(
        prog=prog, description="Evalua el modelo guardado en el conjunto de prueba."
    )
    parser.add_argument("--modelo", metavar="JOBLIB",
                        default=str(DIR_MODELOS / "modelo.joblib"))
    parser.add_argument("--entrada", metavar="CSV", default=str(DIR_DATOS / "test.csv"),
                        help="Conjunto de prueba que escribio act9-datos.")
    parser.add_argument("--salida", metavar="DIR", default=str(DIR_REPORTES),
                        help="Carpeta donde escribir metricas.json.")
    parser.add_argument(
        "--umbral", type=float, default=0.0,
        help="f1_macro minimo aceptable; si no se alcanza, el codigo de salida es 1.",
    )
    args = parser.parse_args(argv)

    for ruta in (args.modelo, args.entrada):
        if not Path(ruta).exists():
            etapa(prog, f"ERROR: no existe {ruta}. Corre antes las etapas anteriores.")
            return 1

    modelo, meta = cargar_modelo(args.modelo)
    df = pd.read_csv(args.entrada)
    X, y = df[FEATURES], df[TARGET]
    etapa(prog, f"modelo {meta.get('modelo', '?')} ({meta.get('familia_ganadora', '?')}), "
                f"entrenado con {meta.get('comando', '?')} el {meta.get('entrenado_en', '?')}")
    etapa(prog, f"{len(df)} partidos de prueba leidos de {args.entrada}")

    metricas = evaluar_en_prueba(modelo, X, y)
    dummy = baseline(meta.get("clase_mayoritaria", y.value_counts().idxmax()), y)
    aprobado = metricas["f1_macro"] >= args.umbral

    banner(f"RESULTADO - accuracy={metricas['accuracy']:.3f}  "
           f"f1_macro={metricas['f1_macro']:.3f}")
    print(f"Baseline (clase mayoritaria): accuracy={dummy['accuracy']:.3f}  "
          f"f1_macro={dummy['f1_macro']:.3f}\n")
    print(metricas["reporte"])
    print("Matriz de confusion:")
    print(matriz_confusion(modelo, X, y).to_string())
    print()

    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)
    resumen = {
        "modelo": meta.get("modelo"),
        "familia": meta.get("familia_ganadora"),
        "accuracy": round(metricas["accuracy"], 4),
        "f1_macro": round(metricas["f1_macro"], 4),
        "baseline_accuracy": round(dummy["accuracy"], 4),
        "baseline_f1_macro": round(dummy["f1_macro"], 4),
        "umbral_f1_macro": args.umbral,
        "aprobado": aprobado,
    }
    (salida / "metricas.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    etapa(prog, f"metricas guardadas en {salida / 'metricas.json'}")

    if not aprobado:
        etapa(prog, f"FALLA: f1_macro={metricas['f1_macro']:.3f} < umbral {args.umbral:.3f} "
                    "-> codigo de salida 1")
        return 1
    etapa(prog, f"OK: f1_macro={metricas['f1_macro']:.3f} >= umbral {args.umbral:.3f}")
    return 0
