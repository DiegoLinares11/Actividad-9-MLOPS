"""Información sobre la computadora y el ambiente de Python en el que se corre.

Sirve de evidencia: cada comando imprime en qué equipo, con qué Python y con
qué gestor de paquetes se instaló el paquete (pip, uv, Poetry o conda).
"""

from __future__ import annotations

import datetime as _dt
import os
import platform
import socket
import sys
from pathlib import Path


def _creado_por(prefijo: Path) -> str:
    """Quién creó el venv, según su ``pyvenv.cfg`` y su ubicación."""
    cfg = prefijo / "pyvenv.cfg"
    texto = cfg.read_text(encoding="utf-8", errors="ignore") if cfg.exists() else ""
    if any(linea.startswith("uv =") for linea in texto.splitlines()):
        return "uv"
    if "pypoetry" in str(prefijo).lower():
        return "Poetry"
    return "venv"


def describir_ambiente() -> dict[str, str]:
    """Tipo, nombre y ruta del ambiente de Python activo.

    - Un ambiente de conda tiene la carpeta ``conda-meta``.
    - En un venv, ``sys.prefix`` apunta al ambiente y ``sys.base_prefix`` al
      Python con el que se creó; si son iguales no hay ambiente virtual.
    """
    prefijo = Path(sys.prefix)
    if (prefijo / "conda-meta").is_dir():
        tipo, gestor = "conda", "conda"
        nombre = os.environ.get("CONDA_DEFAULT_ENV") or prefijo.name
    elif sys.prefix != sys.base_prefix:
        tipo, gestor = "venv", _creado_por(prefijo)
        nombre = prefijo.name
    else:
        tipo, gestor = "global (sin ambiente virtual)", "-"
        nombre = "-"
    return {
        "tipo": tipo,
        "nombre": nombre,
        "gestor": gestor,
        "ejecutable": sys.executable,
        "python": platform.python_version(),
    }


def ruta_corta(ruta: str | Path) -> str:
    """Ruta relativa a la carpeta que contiene el ambiente, si está dentro.

    ``.../Actividad9/.venv/Scripts/act9.EXE`` -> ``.venv/Scripts/act9.EXE``
    """
    base = Path(sys.prefix).parent
    try:
        return str(Path(ruta).resolve().relative_to(base.resolve()))
    except ValueError:
        return str(ruta)


def describir_equipo() -> dict[str, str]:
    return {
        "hostname": socket.gethostname(),
        "sistema": f"{platform.system()} {platform.release()}",
        "fecha": f"{_dt.datetime.now():%Y-%m-%d %H:%M:%S}",
    }
