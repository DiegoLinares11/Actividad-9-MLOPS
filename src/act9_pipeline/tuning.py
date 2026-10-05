"""Calibración de hiperparámetros sobre las familias descubiertas como plugins.

Es la búsqueda de la Actividad 3 (``RandomizedSearchCV`` con una lista de
espacios, uno por familia de modelos), con una diferencia: la lista ya no está
escrita a mano. Se arma con lo que ``plugins.descubrir_modelos()`` encuentre
instalado, así que instalar un plugin agrega su familia a la búsqueda sin
modificar este archivo.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from .pipeline import SEED, construir_pipeline
from .plugins import Familia

#: Métrica que se optimiza. Con clases desbalanceadas (71/48/25), ``f1_macro``
#: pesa igual a las tres clases y no premia ignorar los empates.
SCORING = "f1_macro"

#: Validación cruzada estratificada con semilla fija: mismo resultado en
#: cualquier computadora.
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

#: Hiperparámetros del preprocesamiento, comunes a todas las familias.
ESPACIO_COMUN: dict = {
    "preprocesamiento__numericas__imputar__strategy": ["median", "mean"],
    "seleccion__k": [10, 20, 40, "all"],
}


def espacio_de_busqueda(familias: list[Familia]) -> list[dict]:
    """Un diccionario por familia, con el prefijo ``modelo__`` en sus llaves.

    ``RandomizedSearchCV`` elige primero uno de los diccionarios al azar y
    luego muestrea dentro de él, así que el **orden** de la lista importa para
    la reproducibilidad.
    """
    return [
        {
            **ESPACIO_COMUN,
            "modelo": [f.estimador],
            **{f"modelo__{k}": v for k, v in f.espacio.items()},
        }
        for f in familias
    ]


def busqueda(familias: list[Familia], n_iter: int = 60, n_jobs: int = -1) -> RandomizedSearchCV:
    """``RandomizedSearchCV`` sobre el pipeline completo y las familias dadas."""
    return RandomizedSearchCV(
        estimator=construir_pipeline(),
        param_distributions=espacio_de_busqueda(familias),
        n_iter=n_iter,
        scoring=SCORING,
        cv=CV,
        n_jobs=n_jobs,
        random_state=SEED,
        refit=True,
        return_train_score=True,
    )


def resumen_mejor(search) -> dict:
    """Diccionario compacto con el mejor resultado de la búsqueda."""
    idx = int(np.argmin(search.cv_results_["rank_test_score"]))
    return {
        "mejor_score_cv": float(search.best_score_),
        "desviacion_cv": float(search.cv_results_["std_test_score"][idx]),
        "combinaciones_evaluadas": int(len(search.cv_results_["params"])),
        "mejores_parametros": search.best_params_,
    }
