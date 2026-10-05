"""Comando general ``act9`` con subcomandos.

Hay dos formas de exponer un pipeline como comandos, y el paquete trae las dos
apuntando a las **mismas funciones**:

- un comando por etapa (``act9-datos``, ``act9-entrenar``, ...), cómodo para un
  Makefile, un Dockerfile o un orquestador, que llaman un ejecutable por paso;
- un solo comando con subcomandos (``act9 datos``, ``act9 entrenar``, ...),
  al estilo de ``git`` o ``docker``, más fácil de descubrir con ``act9 --help``.

Ambas se declaran en ``[project.scripts]`` de ``pyproject.toml``. Al instalar,
pip/uv/Poetry/conda crean un ejecutable por cada una (en Windows un ``.exe``
dentro de ``Scripts/``) que importa la función y la llama sin argumentos. Lo
que la función devuelve es el código de salida del proceso.
"""

from __future__ import annotations

import sys

from . import datos, demo, entrenar, evaluar, info, predecir

#: subcomando -> (función, descripción)
ETAPAS = {
    "info": (info.main, "Equipo, ambiente y entry points registrados"),
    "modelos": (info.main_modelos, "Familias de modelos descubiertas como plugins"),
    "datos": (datos.main, "Limpia el CSV y lo separa en train/test"),
    "entrenar": (entrenar.main, "Calibra el pipeline y guarda el modelo"),
    "evaluar": (evaluar.main, "Mide el modelo en el conjunto de prueba"),
    "predecir": (predecir.main, "Predice el resultado de partidos nuevos"),
    "demo": (demo.main, "Todo lo anterior de una vez (evidencia de ejecucion)"),
}


def _ayuda() -> str:
    lineas = ["uso: act9 <etapa> [opciones]", "", "etapas:"]
    lineas += [f"  {nombre:<10} {desc}" for nombre, (_, desc) in ETAPAS.items()]
    lineas += [
        "",
        "Cada etapa tambien existe como comando propio: act9-datos, act9-entrenar, ...",
        "Ayuda de una etapa: act9 <etapa> --help",
    ]
    return "\n".join(lineas)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(_ayuda())
        return 0 if argv else 2
    etapa, resto = argv[0], argv[1:]
    if etapa not in ETAPAS:
        print(f"act9: etapa desconocida '{etapa}'\n\n{_ayuda()}", file=sys.stderr)
        return 2
    funcion = ETAPAS[etapa][0]
    return funcion(resto, prog=f"act9 {etapa}")
