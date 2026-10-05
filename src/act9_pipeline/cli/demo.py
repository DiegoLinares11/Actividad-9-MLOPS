"""``act9-demo``: evidencia de que el paquete se instaló y corre en una computadora.

Imprime el equipo, el ambiente y los entry points registrados, y después
encadena las cuatro etapas (datos, entrenar, evaluar, predecir) llamando a las
mismas funciones que usan los comandos ``act9-*``. Los archivos intermedios
van a una carpeta temporal, así que no toca ``datos/``, ``modelos/`` ni
``reportes/``. Basta con tomarle una captura de pantalla.

Como todas las semillas están fijas, el resultado debe salir idéntico en
cualquier máquina con las mismas familias instaladas.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from . import datos, entrenar, evaluar, info, predecir
from ._comun import banner


def main(argv: list[str] | None = None, prog: str = "act9-demo") -> int:
    parser = argparse.ArgumentParser(
        prog=prog, description="Corre el pipeline completo e imprime la evidencia."
    )
    parser.add_argument("--familias", default="todas")
    parser.add_argument("--n-iter", type=int, default=60)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args(argv)

    banner("ACTIVIDAD 9 - Entry points del pipeline (UEFA Champions League)")
    info.imprimir_equipo_y_ambiente()
    info.imprimir_comandos()
    info.imprimir_modelos()

    with tempfile.TemporaryDirectory(prefix="act9-demo-") as tmp:
        tmp = Path(tmp)
        pasos = [
            ("act9-datos", datos.main, ["--salida", str(tmp)]),
            ("act9-entrenar", entrenar.main, [
                "--entrada", str(tmp / "train.csv"), "--salida", str(tmp),
                "--familias", args.familias, "--n-iter", str(args.n_iter),
                "--n-jobs", str(args.n_jobs), "--diagrama", "",
            ]),
            ("act9-evaluar", evaluar.main, [
                "--modelo", str(tmp / "modelo.joblib"),
                "--entrada", str(tmp / "test.csv"), "--salida", str(tmp),
            ]),
            ("act9-predecir", predecir.main, [
                "--modelo", str(tmp / "modelo.joblib"),
                "--entrada", str(tmp / "test.csv"),
                "--salida", str(tmp / "predicciones.csv"),
            ]),
        ]
        for nombre, funcion, argumentos in pasos:
            print(f"\n--- {nombre} " + "-" * (63 - len(nombre)))
            codigo = funcion(argumentos, prog=nombre)
            if codigo != 0:
                print(f"\n[ERROR] {nombre} termino con codigo {codigo}")
                return codigo

    print("\n[OK] Las cuatro etapas corrieron en este equipo.")
    return 0
