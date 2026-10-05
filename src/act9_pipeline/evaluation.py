"""Evaluación del modelo entrenado sobre el conjunto de prueba reservado."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def evaluar_en_prueba(pipe, X_test, y_test) -> dict:
    """Métricas del pipeline ya entrenado sobre el conjunto reservado."""
    y_pred = pipe.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "reporte": classification_report(y_test, y_pred, digits=3, zero_division=0),
    }


def baseline(clase_mayoritaria: str, y_test) -> dict:
    """Un modelo sin mérito que siempre predice la clase mayoritaria.

    Equivale a ``DummyClassifier(strategy="most_frequent")``. La clase se
    guarda en los metadatos al entrenar, así que para evaluar no hace falta
    volver a leer el conjunto de entrenamiento.
    """
    y_pred = [clase_mayoritaria] * len(y_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
    }


def matriz_confusion(pipe, X_test, y_test) -> pd.DataFrame:
    """Matriz de confusión con etiquetas, como DataFrame."""
    etiquetas = sorted(pd.unique(y_test))
    m = confusion_matrix(y_test, pipe.predict(X_test), labels=etiquetas)
    return pd.DataFrame(
        m,
        index=[f"real: {e}" for e in etiquetas],
        columns=[f"pred: {e}" for e in etiquetas],
    )
