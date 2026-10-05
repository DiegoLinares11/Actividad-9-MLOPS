"""Shim de compatibilidad.

Toda la configuración del paquete vive en ``pyproject.toml``. Este archivo se
mantiene para poder instalar con herramientas o comandos que aún esperan un
``setup.py`` (por ejemplo ``python setup.py ...``).
"""

from setuptools import setup

if __name__ == "__main__":
    setup()
