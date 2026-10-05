"""Utilidades compartidas por los comandos."""

from __future__ import annotations

from pathlib import Path

#: Rutas por defecto, relativas a la carpeta desde donde se corre el comando.
#: Son las mismas que usa el Makefile.
DIR_DATOS = Path("datos/procesados")
DIR_MODELOS = Path("modelos")
DIR_REPORTES = Path("reportes")


def banner(texto: str, ancho: int = 68) -> None:
    print("\n" + "=" * ancho)
    print(texto)
    print("=" * ancho)


def etapa(prog: str, texto: str) -> None:
    """Primera línea de cada comando: deja claro qué entry point se ejecutó."""
    print(f"[{prog}] {texto}")
