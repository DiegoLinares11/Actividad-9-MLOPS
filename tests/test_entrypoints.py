"""Pruebas de los entry points del paquete.

Ejecutar con::

    pip install -e .[dev]
    pytest -q          # o: make test
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from importlib.metadata import EntryPoint, entry_points

import pytest
from sklearn.tree import DecisionTreeClassifier

import act9_pipeline.plugins as plugins
from act9_pipeline import GRUPO_MODELOS, descubrir_modelos, espacio_de_busqueda
from act9_pipeline.cli import datos, entrenar, evaluar, main, predecir

COMANDOS = {
    "act9": "act9_pipeline.cli:main",
    "act9-demo": "act9_pipeline.cli.demo:main",
    "act9-datos": "act9_pipeline.cli.datos:main",
    "act9-entrenar": "act9_pipeline.cli.entrenar:main",
    "act9-evaluar": "act9_pipeline.cli.evaluar:main",
    "act9-predecir": "act9_pipeline.cli.predecir:main",
}
FAMILIAS_PROPIAS = {"random_forest", "regresion_logistica", "svc", "hist_gradient_boosting"}


def plugin_de_prueba() -> dict:
    """Un plugin "de terceros" definido fuera del paquete, para las pruebas."""
    return {
        "estimador": DecisionTreeClassifier(random_state=0),
        "espacio": {"max_depth": [2, 3]},
        "descripcion": "árbol de decisión de prueba",
    }


def _con_entry_points_extra(monkeypatch, *extra: EntryPoint) -> None:
    """Simula que hay paquetes instalados que registran ``extra`` en el grupo."""
    reales = list(entry_points(group=GRUPO_MODELOS))
    monkeypatch.setattr(plugins, "entry_points", lambda group: [*reales, *extra])


# --------------------------------------------------------------------------
#  La metadata instalada
# --------------------------------------------------------------------------

def test_console_scripts_registrados():
    """pyproject.toml -> entry_points.txt: cada comando apunta a su función."""
    registrados = {
        ep.name: ep.value
        for ep in entry_points(group="console_scripts")
        if ep.value.startswith("act9_pipeline")
    }
    assert registrados == COMANDOS


def test_las_referencias_se_pueden_cargar():
    """``modulo:objeto`` de cada comando se importa y es una función."""
    for ep in entry_points(group="console_scripts"):
        if ep.value.startswith("act9_pipeline"):
            assert callable(ep.load())


def test_grupo_propio_trae_las_cuatro_familias():
    familias = descubrir_modelos()
    assert FAMILIAS_PROPIAS <= set(familias)
    for nombre in FAMILIAS_PROPIAS:
        assert familias[nombre].distribucion.startswith("act9-pipeline")
    # Ordenadas por nombre: el espacio de búsqueda no depende del sistema.
    assert list(familias) == sorted(familias)


def test_espacio_de_busqueda_usa_el_prefijo_del_pipeline():
    familias = list(descubrir_modelos().values())
    for espacio, familia in zip(espacio_de_busqueda(familias), familias):
        assert espacio["modelo"] == [familia.estimador]
        assert all(k.startswith(("modelo", "preprocesamiento__", "seleccion__")) for k in espacio)


# --------------------------------------------------------------------------
#  Plugins de otros paquetes
# --------------------------------------------------------------------------

def test_un_plugin_externo_se_descubre_y_entrena(monkeypatch, tmp_path):
    extra = EntryPoint("arbol", "test_entrypoints:plugin_de_prueba", GRUPO_MODELOS)
    _con_entry_points_extra(monkeypatch, extra)
    assert "arbol" in descubrir_modelos()

    assert datos.main(["--salida", str(tmp_path)]) == 0
    assert entrenar.main([
        "--entrada", str(tmp_path / "train.csv"), "--salida", str(tmp_path),
        "--familias", "arbol", "--n-iter", "2", "--n-jobs", "1", "--diagrama", "",
    ]) == 0
    meta = json.loads((tmp_path / "metadatos.json").read_text(encoding="utf-8"))
    assert meta["familia_ganadora"] == "arbol"
    assert meta["modelo"] == "DecisionTreeClassifier"


def test_un_plugin_roto_se_omite_sin_tumbar_el_pipeline(monkeypatch):
    roto = EntryPoint("roto", "modulo_que_no_existe:fabrica", GRUPO_MODELOS)
    _con_entry_points_extra(monkeypatch, roto)
    with pytest.warns(UserWarning, match="Se omite el plugin 'roto'"):
        familias = descubrir_modelos()
    assert "roto" not in familias
    assert FAMILIAS_PROPIAS <= set(familias)


def test_familia_desconocida_es_error_de_uso(tmp_path):
    assert entrenar.main(["--familias", "no_existe", "--salida", str(tmp_path)]) == 2


# --------------------------------------------------------------------------
#  Las funciones de los comandos
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def carpeta_entrenada(tmp_path_factory):
    """datos -> entrenar una sola vez (búsqueda corta) para varias pruebas."""
    tmp = tmp_path_factory.mktemp("pipeline")
    assert datos.main(["--salida", str(tmp)]) == 0
    assert entrenar.main([
        "--entrada", str(tmp / "train.csv"), "--salida", str(tmp),
        "--familias", "regresion_logistica", "--n-iter", "5", "--n-jobs", "1",
        "--diagrama", "",
    ]) == 0
    return tmp


def test_etapas_encadenadas(carpeta_entrenada):
    tmp = carpeta_entrenada
    args = ["--modelo", str(tmp / "modelo.joblib"), "--entrada", str(tmp / "test.csv")]
    assert evaluar.main([*args, "--salida", str(tmp)]) == 0
    assert predecir.main([*args, "--salida", str(tmp / "pred.csv")]) == 0
    metricas = json.loads((tmp / "metricas.json").read_text(encoding="utf-8"))
    assert metricas["f1_macro"] > metricas["baseline_f1_macro"]
    assert (tmp / "pred.csv").read_text(encoding="utf-8").startswith("home_team,away_team,prediccion")


def test_compuerta_de_calidad_devuelve_1(carpeta_entrenada):
    tmp = carpeta_entrenada
    codigo = evaluar.main([
        "--modelo", str(tmp / "modelo.joblib"), "--entrada", str(tmp / "test.csv"),
        "--salida", str(tmp / "gate"), "--umbral", "0.99",
    ])
    assert codigo == 1


def test_comando_general_despacha():
    assert main(["--help"]) == 0
    assert main([]) == 2
    assert main(["etapa_que_no_existe"]) == 2
    assert main(["modelos"]) == 0


def test_misma_calibracion_que_la_actividad_3(tmp_path):
    """Con las familias en el orden de la Actividad 3, el resultado es idéntico."""
    assert datos.main(["--salida", str(tmp_path)]) == 0
    assert entrenar.main([
        "--entrada", str(tmp_path / "train.csv"), "--salida", str(tmp_path),
        "--familias", "random_forest,regresion_logistica,svc,hist_gradient_boosting",
        "--diagrama", "",
    ]) == 0
    meta = json.loads((tmp_path / "metadatos.json").read_text(encoding="utf-8"))
    assert meta["modelo"] == "LogisticRegression"
    assert meta["cv_mejor_score"] == pytest.approx(0.619, abs=5e-4)


# --------------------------------------------------------------------------
#  Los ejecutables que creó el instalador
# --------------------------------------------------------------------------

requiere_instalacion = pytest.mark.skipif(
    shutil.which("act9") is None, reason="los comandos act9-* no están en el PATH"
)


@requiere_instalacion
def test_el_ejecutable_corre(tmp_path):
    r = subprocess.run(["act9", "info"], capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode == 0
    assert "act9-entrenar" in r.stdout


@requiere_instalacion
def test_el_codigo_de_salida_llega_al_sistema(carpeta_entrenada):
    """Lo que devuelve main() es el código de salida del proceso."""
    tmp = carpeta_entrenada
    r = subprocess.run(
        ["act9-evaluar", "--modelo", str(tmp / "modelo.joblib"),
         "--entrada", str(tmp / "test.csv"), "--salida", str(tmp / "sub"),
         "--umbral", "0.99"],
        capture_output=True, text=True,
    )
    assert r.returncode == 1


def test_python_m_funciona_sin_los_ejecutables():
    r = subprocess.run([sys.executable, "-m", "act9_pipeline", "--help"],
                       capture_output=True, text=True)
    assert r.returncode == 0
    assert "act9 <etapa>" in r.stdout
