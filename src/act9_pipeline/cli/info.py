"""``act9 info`` y ``act9 modelos``: qué entry points dejó instalados el paquete.

Todo se lee de la metadata instalada (``entry_points.txt`` de cada
``.dist-info``) con ``importlib.metadata``, no del código fuente. Si un
comando no aparece aquí, el instalador no lo registró.
"""

from __future__ import annotations

import argparse
import shutil
import sys

from .. import __version__
from ..ambiente import describir_ambiente, describir_equipo, ruta_corta
from ..plugins import GRUPO_MODELOS, comandos_del_paquete, descubrir_modelos
from ._comun import banner


def imprimir_equipo_y_ambiente() -> None:
    eq, amb = describir_equipo(), describir_ambiente()
    print(f"Paquete act9_pipeline v{__version__}")
    print(f"Equipo (hostname): {eq['hostname']}")
    print(f"Sistema operativo: {eq['sistema']}")
    print(f"Fecha/hora:        {eq['fecha']}")
    print(f"Ambiente:          {amb['tipo']} '{amb['nombre']}', creado con {amb['gestor']}")
    print(f"Python:            {amb['python']}")
    print(f"Interprete:        {ruta_corta(amb['ejecutable'])}")
    print(f"Comando invocado:  {ruta_corta(sys.argv[0])}")


def imprimir_comandos() -> None:
    print("\nconsole_scripts del paquete (nombre = referencia -> ejecutable):")
    comandos = comandos_del_paquete()
    if not comandos:
        print("  (ninguno: el paquete no esta instalado, solo se importa desde el codigo)")
    for ep in comandos:
        ruta = shutil.which(ep.name)
        ruta = ruta_corta(ruta) if ruta else "NO ESTA EN EL PATH"
        print(f"  {ep.name:<14} = {ep.value:<33} -> {ruta}")


def imprimir_modelos() -> None:
    print(f"\nFamilias de modelos en el grupo '{GRUPO_MODELOS}':")
    familias = descubrir_modelos()
    if not familias:
        print("  (ninguna)")
    for f in familias.values():
        print(f"  {f.nombre:<23} = {f.referencia:<45} [{f.distribucion}]")


def main(argv: list[str] | None = None, prog: str = "act9 info") -> int:
    argparse.ArgumentParser(
        prog=prog, description="Muestra el equipo, el ambiente y los entry points."
    ).parse_args(argv)
    banner("ACTIVIDAD 9 - Entry points del pipeline (UEFA Champions League)")
    imprimir_equipo_y_ambiente()
    imprimir_comandos()
    imprimir_modelos()
    return 0


def main_modelos(argv: list[str] | None = None, prog: str = "act9 modelos") -> int:
    argparse.ArgumentParser(
        prog=prog, description="Lista las familias de modelos registradas como plugins."
    ).parse_args(argv)
    familias = descubrir_modelos()
    print(f"Grupo de entry points: {GRUPO_MODELOS}  ({len(familias)} familias)\n")
    for f in familias.values():
        print(f"{f.nombre}")
        print(f"    {f.descripcion}")
        print(f"    referencia: {f.referencia}")
        print(f"    paquete:    {f.distribucion}")
        print(f"    calibra:    {', '.join(f.espacio)}")
    return 0
