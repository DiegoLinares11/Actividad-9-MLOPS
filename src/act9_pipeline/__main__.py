"""Permite ``python -m act9_pipeline <etapa>``.

Hace lo mismo que el comando ``act9``. Es el plan B cuando la carpeta de
scripts del ambiente (donde el instalador dejó los ejecutables de los entry
points) no está en el ``PATH``: ``-m`` solo necesita que el paquete se pueda
importar.
"""

from .cli import main

raise SystemExit(main())
