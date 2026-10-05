"""act9_pipeline

Paquete de la Actividad 9: el pipeline de scikit-learn de la Actividad 3
(dataset de la UEFA Champions League del Ejercicio 1), reorganizado alrededor
de **entry points** de Python:

- ``console_scripts``: un comando por etapa del pipeline (``act9-datos``,
  ``act9-entrenar``, ``act9-evaluar``, ``act9-predecir``) más un comando
  general con subcomandos (``act9``) y el de evidencia (``act9-demo``).
- Grupo propio ``act9_pipeline.modelos``: las familias de modelos se
  registran como plugins y se descubren al ejecutar, así que otro paquete
  puede agregar una familia nueva sin modificar este.
"""

from .artefactos import cargar_modelo, guardar_modelo
from .data import (
    CATEGORICAL_FEATURES,
    FEATURES,
    LEAKAGE_COLUMNS,
    NUMERIC_FEATURES,
    PERCENT_FEATURES,
    RATIO_FEATURES,
    TARGET,
    cargar_datos,
    extraer_datos,
    filtrar_datos,
    separar_datos,
)
from .evaluation import baseline, evaluar_en_prueba, matriz_confusion
from .pipeline import SEED, construir_pipeline, construir_preprocesamiento
from .plugins import GRUPO_MODELOS, Familia, comandos_del_paquete, descubrir_modelos
from .transformers import PorcentajeATexto, RatioATexto
from .tuning import CV, SCORING, busqueda, espacio_de_busqueda, resumen_mejor

__all__ = [
    # datos
    "NUMERIC_FEATURES",
    "PERCENT_FEATURES",
    "RATIO_FEATURES",
    "CATEGORICAL_FEATURES",
    "LEAKAGE_COLUMNS",
    "FEATURES",
    "TARGET",
    "extraer_datos",
    "filtrar_datos",
    "separar_datos",
    "cargar_datos",
    # pipeline
    "SEED",
    "construir_pipeline",
    "construir_preprocesamiento",
    "PorcentajeATexto",
    "RatioATexto",
    # entry points
    "GRUPO_MODELOS",
    "Familia",
    "descubrir_modelos",
    "comandos_del_paquete",
    # calibración y evaluación
    "SCORING",
    "CV",
    "espacio_de_busqueda",
    "busqueda",
    "resumen_mejor",
    "evaluar_en_prueba",
    "baseline",
    "matriz_confusion",
    # artefactos
    "guardar_modelo",
    "cargar_modelo",
]

__version__ = "0.1.0"
