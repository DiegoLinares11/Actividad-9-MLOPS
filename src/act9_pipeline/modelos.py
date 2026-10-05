"""Familias de modelos que trae el paquete, registradas como *plugins*.

Cada función de este módulo está declarada en ``pyproject.toml`` como un entry
point del grupo ``act9_pipeline.modelos``::

    [project.entry-points."act9_pipeline.modelos"]
    random_forest = "act9_pipeline.modelos:random_forest"
    ...

El paquete **no importa este módulo directamente** para saber qué modelos
existen: los descubre leyendo la metadata de todo lo que está instalado
(ver ``plugins.py``). Por eso un paquete externo puede agregar una familia
nueva sin tocar este código, como hace ``plugins/act9-modelo-knn``.

Contrato de un plugin
---------------------
Una función sin argumentos que devuelve un diccionario con:

``estimador``
    Un clasificador de scikit-learn sin entrenar.
``espacio``
    Hiperparámetros a calibrar con ``RandomizedSearchCV``, con los nombres
    propios del estimador (``C``, ``max_depth``, ...). El paquete les agrega
    el prefijo ``modelo__`` para ubicarlos dentro del pipeline.
``descripcion``
    Una línea que se muestra en ``act9 modelos``.

Los espacios son los mismos de la Actividad 3 (``tuning.espacio_aleatorio``);
la justificación de cada hiperparámetro está documentada allá.
"""

from __future__ import annotations

from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from .pipeline import SEED


def random_forest() -> dict:
    """Bosque aleatorio: ``max_depth`` y ``min_samples_leaf`` frenan el sobreajuste."""
    return {
        "estimador": RandomForestClassifier(random_state=SEED),
        "espacio": {
            "n_estimators": randint(100, 600),
            "max_depth": [None, 4, 6, 10, 16],
            "min_samples_leaf": randint(1, 6),
            "max_features": ["sqrt", "log2", None],
            "class_weight": [None, "balanced"],
        },
        "descripcion": "RandomForestClassifier (bosque aleatorio)",
    }


def regresion_logistica() -> dict:
    """Regresión logística: ``C`` se busca en escala logarítmica."""
    return {
        "estimador": LogisticRegression(max_iter=5000, random_state=SEED),
        "espacio": {
            "C": loguniform(1e-3, 1e2),
            "class_weight": [None, "balanced"],
        },
        "descripcion": "LogisticRegression (lineal, regularizacion L2)",
    }


def svc() -> dict:
    """Máquina de vectores de soporte: ``C`` y ``gamma`` se compensan entre sí."""
    return {
        "estimador": SVC(random_state=SEED),
        "espacio": {
            "C": loguniform(1e-2, 1e2),
            "gamma": loguniform(1e-4, 1e0),
            "kernel": ["rbf", "linear"],
            "class_weight": [None, "balanced"],
        },
        "descripcion": "SVC (maquina de vectores de soporte)",
    }


def hist_gradient_boosting() -> dict:
    """Gradient boosting por histogramas: ``learning_rate`` y ``max_iter`` van juntos."""
    return {
        "estimador": HistGradientBoostingClassifier(random_state=SEED),
        "espacio": {
            "learning_rate": uniform(0.02, 0.28),
            "max_iter": randint(80, 400),
            "max_leaf_nodes": [7, 15, 31],
            "min_samples_leaf": randint(5, 25),
        },
        "descripcion": "HistGradientBoostingClassifier (boosting por histogramas)",
    }
