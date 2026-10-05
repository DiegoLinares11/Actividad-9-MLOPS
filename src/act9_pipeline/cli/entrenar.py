"""``act9-entrenar``: calibra el pipeline y guarda el mejor modelo (Modeling).

Las familias de modelos no están fijas en el código: se descubren en el grupo
de entry points ``act9_pipeline.modelos``. Con ``--familias`` se elige cuáles
entran a la búsqueda y en qué orden.

    act9-entrenar                                   # todas las familias instaladas
    act9-entrenar --familias svc,random_forest      # solo esas dos, en ese orden
    act9-entrenar --n-iter 20 --n-jobs 1            # búsqueda más corta
"""

from __future__ import annotations

import argparse
import datetime as dt
import platform
import socket
import time
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.utils import estimator_html_repr

from .. import __version__
from ..artefactos import guardar_modelo
from ..data import FEATURES, TARGET
from ..plugins import GRUPO_MODELOS, descubrir_modelos
from ..tuning import SCORING, busqueda, resumen_mejor
from ._comun import DIR_DATOS, DIR_MODELOS, etapa


def main(argv: list[str] | None = None, prog: str = "act9-entrenar") -> int:
    parser = argparse.ArgumentParser(
        prog=prog, description="Calibra el pipeline con RandomizedSearchCV y guarda el modelo."
    )
    parser.add_argument(
        "--entrada", metavar="CSV", default=str(DIR_DATOS / "train.csv"),
        help="Conjunto de entrenamiento que escribio act9-datos.",
    )
    parser.add_argument(
        "--salida", metavar="DIR", default=str(DIR_MODELOS),
        help="Carpeta donde guardar modelo.joblib y metadatos.json.",
    )
    parser.add_argument(
        "--familias", default="todas",
        help="Familias separadas por comas, en el orden deseado, o 'todas' "
        "(ordenadas por nombre). Ver: act9 modelos",
    )
    parser.add_argument("--n-iter", type=int, default=60,
                        help="Combinaciones a muestrear (por defecto 60).")
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="Procesos en paralelo (-1 = todos los nucleos).")
    parser.add_argument(
        "--diagrama", metavar="HTML", default="pipeline_diagram.html",
        help="Donde guardar el diagrama del pipeline ganador ('' para no guardarlo).",
    )
    args = parser.parse_args(argv)

    disponibles = descubrir_modelos()
    if args.familias == "todas":
        nombres = list(disponibles)
    else:
        nombres = [n.strip() for n in args.familias.split(",") if n.strip()]
        desconocidas = [n for n in nombres if n not in disponibles]
        if desconocidas:
            etapa(prog, f"ERROR: no hay plugins llamados {desconocidas} en {GRUPO_MODELOS}")
            etapa(prog, f"disponibles: {', '.join(disponibles)}")
            return 2
    if not nombres:
        etapa(prog, f"ERROR: no hay ninguna familia registrada en {GRUPO_MODELOS}")
        return 2
    familias = [disponibles[n] for n in nombres]
    etapa(prog, f"familias descubiertas en '{GRUPO_MODELOS}': {', '.join(disponibles)}")
    etapa(prog, f"familias en la busqueda: {', '.join(nombres)}")

    entrada = Path(args.entrada)
    if not entrada.exists():
        etapa(prog, f"ERROR: no existe {entrada}. Corre primero act9-datos (o make datos).")
        return 1
    df = pd.read_csv(entrada)
    X, y = df[FEATURES], df[TARGET]
    etapa(prog, f"{len(df)} partidos de entrenamiento leidos de {entrada}")

    etapa(prog, f"RandomizedSearchCV: {args.n_iter} combinaciones x 5 folds, metrica {SCORING}")
    search = busqueda(familias, n_iter=args.n_iter, n_jobs=args.n_jobs)
    inicio = time.perf_counter()
    search.fit(X, y)
    segundos = time.perf_counter() - inicio

    resumen = resumen_mejor(search)
    mejor = search.best_estimator_
    clase = type(mejor.named_steps["modelo"]).__name__
    # best_params_["modelo"] es el mismo objeto que entregó el plugin, así que
    # se identifica la familia aunque dos plugins usen la misma clase.
    familia = next(f.nombre for f in familias if f.estimador is search.best_params_["modelo"])
    origen = disponibles[familia].distribucion

    metadatos = {
        "paquete": f"act9-pipeline {__version__}",
        "comando": prog,
        "entrenado_en": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "equipo": socket.gethostname(),
        "familias_evaluadas": nombres,
        "familia_ganadora": familia,
        "plugin_de": origen,
        "modelo": clase,
        "metrica_optimizada": SCORING,
        "cv_mejor_score": round(resumen["mejor_score_cv"], 4),
        "cv_desviacion": round(resumen["desviacion_cv"], 4),
        "combinaciones_evaluadas": resumen["combinaciones_evaluadas"],
        "segundos_entrenamiento": round(segundos, 1),
        "mejores_hiperparametros": {
            k: str(v) for k, v in search.best_params_.items() if k != "modelo"
        },
        "clase_mayoritaria": str(y.value_counts().idxmax()),
        "clases": sorted(map(str, mejor.classes_)),
        "versiones": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
        },
    }
    ruta = guardar_modelo(mejor, metadatos, args.salida)

    etapa(prog, f"listo en {segundos:.1f} s | mejor {SCORING} en CV = "
                f"{resumen['mejor_score_cv']:.3f} (+/- {resumen['desviacion_cv']:.3f})")
    etapa(prog, f"ganador: {familia} -> {clase} (plugin de {origen})")
    for k, v in sorted(metadatos["mejores_hiperparametros"].items()):
        print(f"           {k} = {v}")
    etapa(prog, f"modelo guardado en {ruta} ({ruta.stat().st_size / 1024:.0f} KB)")

    if args.diagrama:
        Path(args.diagrama).write_text(estimator_html_repr(mejor), encoding="utf-8")
        etapa(prog, f"diagrama del pipeline en {args.diagrama}")
    return 0
