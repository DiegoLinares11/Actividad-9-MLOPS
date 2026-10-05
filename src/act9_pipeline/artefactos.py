"""Guardar y cargar el modelo entrenado entre una etapa y otra.

Cada etapa del pipeline es un comando distinto (un entry point), así que no
comparten memoria: ``act9-entrenar`` deja el modelo en disco y
``act9-evaluar`` / ``act9-predecir`` lo leen. Junto al modelo se guarda un JSON
con sus metadatos (cómo y con qué versiones se entrenó), igual que en el
servicio entrenador de la Actividad 4.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib

NOMBRE_MODELO = "modelo.joblib"
NOMBRE_METADATOS = "metadatos.json"


def guardar_modelo(modelo, metadatos: dict, directorio: str | Path) -> Path:
    """Escribe ``modelo.joblib`` y ``metadatos.json`` en ``directorio``."""
    directorio = Path(directorio)
    directorio.mkdir(parents=True, exist_ok=True)
    ruta = directorio / NOMBRE_MODELO
    joblib.dump(modelo, ruta)
    (directorio / NOMBRE_METADATOS).write_text(
        json.dumps(metadatos, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return ruta


def cargar_modelo(ruta: str | Path):
    """Devuelve ``(modelo, metadatos)``; los metadatos se buscan junto al modelo."""
    ruta = Path(ruta)
    modelo = joblib.load(ruta)
    ruta_meta = ruta.with_name(NOMBRE_METADATOS)
    metadatos = (
        json.loads(ruta_meta.read_text(encoding="utf-8")) if ruta_meta.exists() else {}
    )
    return modelo, metadatos
