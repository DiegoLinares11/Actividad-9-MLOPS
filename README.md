# Actividad 9 · Entry points de Python en el pipeline de scikit-learn

Curso **Machine Learning Engineering (MLE/MLOps)**, Universidad del Valle de Guatemala.
Diego Linares · Andy Fuentes · Christian Echeverria · Diederich Solis

Investigamos los **entry points** de Python (especificación de PyPA) y adaptamos el pipeline de
scikit-learn de la Actividad 3 (dataset de la UEFA Champions League del Ejercicio 1) para usarlos de
dos formas:

1. **`console_scripts`**: cada etapa del pipeline es un comando que crea el instalador
   (`act9-datos`, `act9-entrenar`, `act9-evaluar`, `act9-predecir`), más un comando general con
   subcomandos (`act9`) y uno de evidencia (`act9-demo`).
2. **Un grupo propio, `act9_pipeline.modelos`**: las familias de modelos se registran como
   plugins y `act9-entrenar` las descubre al arrancar. Otro paquete puede agregar una familia sin
   tocar este (ver [`plugins/act9-modelo-knn`](plugins/act9-modelo-knn)).

Encima de los comandos hay un **Makefile** que los encadena y un workflow de **GitHub Actions**
que llama al Makefile.

| Entregable | Dónde está |
|---|---|
| Link al repositorio | https://github.com/DiegoLinares11/Actividad-9-MLOPS |
| Investigación sobre entry points | [Reporte](reporte/Actividad9_MLOPS.pdf), sección 1 |
| Código del pipeline adaptado | [`src/act9_pipeline/`](src/act9_pipeline), [`pyproject.toml`](pyproject.toml) |
| Pregunta 1: usos en producción | [Respuestas](#respuestas-a-las-preguntas) |
| Pregunta 2: gestores de paquetes | [Respuestas](#respuestas-a-las-preguntas) |
| Pregunta 3: relación con un Makefile | [Respuestas](#respuestas-a-las-preguntas), [`Makefile`](Makefile) |
| Reporte completo (PDF y LaTeX) | [reporte/Actividad9_MLOPS.pdf](reporte/Actividad9_MLOPS.pdf) |

---

## Inicio rápido

Con **venv + pip** (Windows, PowerShell):

```powershell
git clone https://github.com/DiegoLinares11/Actividad-9-MLOPS.git
cd Actividad-9-MLOPS
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
act9-demo
```

En Linux o macOS solo cambia la activación: `source .venv/bin/activate`.

Con **conda** (trae GNU make, así que en Windows también funciona el Makefile):

```bash
conda env create -f environment.yml
conda activate act9-mlops
make pipeline
```

Con **uv** o con **Poetry** no hace falta crear ni activar el ambiente:

```bash
uv run act9-demo
poetry install && poetry run act9-demo
```

> Si la carpeta de scripts del ambiente no está en el `PATH`, los comandos no se encuentran.
> `python -m act9_pipeline <etapa>` hace lo mismo que `act9 <etapa>`.

---

## Los entry points del paquete

Se declaran en [`pyproject.toml`](pyproject.toml):

```toml
[project.scripts]                       # grupo especial console_scripts
act9          = "act9_pipeline.cli:main"
act9-demo     = "act9_pipeline.cli.demo:main"
act9-datos    = "act9_pipeline.cli.datos:main"
act9-entrenar = "act9_pipeline.cli.entrenar:main"
act9-evaluar  = "act9_pipeline.cli.evaluar:main"
act9-predecir = "act9_pipeline.cli.predecir:main"

[project.entry-points."act9_pipeline.modelos"]   # grupo propio (plugins)
random_forest          = "act9_pipeline.modelos:random_forest"
regresion_logistica    = "act9_pipeline.modelos:regresion_logistica"
svc                    = "act9_pipeline.modelos:svc"
hist_gradient_boosting = "act9_pipeline.modelos:hist_gradient_boosting"
```

| Comando | Etapa de CRISP-DM | Qué hace | Lee | Escribe |
|---|---|---|---|---|
| `act9-datos` | Data Preparation | Valida columnas, limpia (151 → 144 partidos) y separa train/test estratificado | CSV crudo (empaquetado o `--entrada`) | `datos/procesados/train.csv`, `test.csv` |
| `act9-entrenar` | Modeling | `RandomizedSearchCV` sobre las familias descubiertas como plugins | `train.csv` | `modelos/modelo.joblib`, `metadatos.json`, `pipeline_diagram.html` |
| `act9-evaluar` | Evaluation | Métricas, baseline y matriz de confusión. **Código de salida 1** si `f1_macro < --umbral` | modelo + `test.csv` | `reportes/metricas.json` |
| `act9-predecir` | Deployment | Predicciones (y probabilidades, si el modelo las da) para partidos nuevos | modelo + cualquier CSV | `reportes/predicciones.csv` |
| `act9 <etapa>` | — | Las mismas funciones como subcomandos, más `act9 info` y `act9 modelos` | | |
| `act9-demo` | — | Equipo, ambiente, entry points registrados y las cuatro etapas en una carpeta temporal | | |

Al instalar, el instalador escribe esas líneas en
`act9_pipeline-0.1.0.dist-info/entry_points.txt` y crea un ejecutable por comando. En Windows es
un lanzador `.exe` que lleva dentro este `__main__.py` (lo sacamos del `.exe` que generó pip):

```python
import sys
from act9_pipeline.cli.entrenar import main
if __name__ == '__main__':
    sys.argv[0] = sys.argv[0].removesuffix('.exe')
    sys.exit(main())
```

Es decir: el comando importa la función, la llama sin argumentos y usa lo que devuelve como código
de salida. Por eso cada `main()` de este paquete devuelve `0`, `1` o `2`.

### Plugins: agregar una familia de modelos sin tocar el paquete

[`plugins/act9-modelo-knn`](plugins/act9-modelo-knn) es otro paquete, con su propio
`pyproject.toml`, que no tiene comandos y no importa nada de `act9_pipeline`. Solo se registra en
el mismo grupo:

```toml
[project.entry-points."act9_pipeline.modelos"]
knn = "act9_modelo_knn:knn"
```

```bash
make plugin-knn            # pip install -e plugins/act9-modelo-knn
act9 modelos               # ahora son 5 familias; knn dice "paquete: act9-modelo-knn 0.1.0"
act9-entrenar --familias knn
```

`plugins.descubrir_modelos()` lee el grupo con `importlib.metadata.entry_points()`, ordena por
nombre (para que el resultado no dependa del orden del sistema de archivos), valida que cada plugin
cumpla el contrato y omite con una advertencia el que falle, sin detener el entrenamiento.

---

## El Makefile

Cada receta llama a un entry point, y cada objetivo es el archivo que produce esa etapa, así que
`make` sabe qué volver a correr comparando fechas:

```
CSV crudo ──act9-datos──> train.csv, test.csv ──act9-entrenar──> modelo.joblib
                                                                    │
                     metricas.json <──act9-evaluar──────────────────┤
                           │                                        │
                           └──────────> predicciones.csv <──act9-predecir
```

| Objetivo | Receta |
|---|---|
| `make install` / `install-dev` | `pip install -e .`: aquí se crean los ejecutables |
| `make datos` | `act9-datos --salida datos/procesados` |
| `make entrenar` | `act9-entrenar ... --familias $(FAMILIAS) --n-iter $(N_ITER)` |
| `make evaluar` | `act9-evaluar ... --umbral $(UMBRAL)` |
| `make predecir` | `act9-predecir ...` (depende de `metricas.json`: no se predice con un modelo que no pasó la compuerta) |
| `make pipeline` | las cuatro, solo las que hagan falta |
| `make test`, `build`, `clean`, `info`, `modelos`, `plugin-knn` | utilidades |

Lo que comprobamos en Windows con GNU Make 4.4.1 (instalado con conda):

| Situación | Qué hizo make |
|---|---|
| `make pipeline` desde cero | Corrió las cuatro etapas |
| `make pipeline` otra vez | `Nothing to be done for 'pipeline'` |
| El modelo es más nuevo que las métricas | Solo repitió `act9-evaluar` y `act9-predecir` |
| `make pipeline UMBRAL=0.90` | `act9-evaluar` devolvió 1, make se detuvo, borró `metricas.json` (`.DELETE_ON_ERROR`) y no corrió la predicción |

La variable `RUN` antepone el gestor de paquetes a cada comando, así que el mismo Makefile sirve
para todos:

```bash
make pipeline                              # ambiente activado (venv o conda)
make pipeline RUN="uv run"
make pipeline RUN="poetry run"
make pipeline RUN="conda run -n act9-mlops"
```

El workflow [`.github/workflows/pipeline.yml`](.github/workflows/pipeline.yml) solo llama a
objetivos del Makefile (`make install-dev`, `make test`, `make pipeline`, ...), con un job para
pip y otro para uv. Los dos jobs pasaron en Ubuntu, incluida la prueba que exige el 0.619 de la Actividad 3.

---

## Resultados

Todas las semillas están fijas, así que estos números salen iguales en cualquier computadora con
las mismas familias instaladas.

| Configuración | Mejor `f1_macro` en CV | Modelo ganador | Test: accuracy / `f1_macro` |
|---|---|---|---|
| Familias en el orden de la Actividad 3 | **0.619** | LogisticRegression | **0.690 / 0.631** (idéntico a la Actividad 3) |
| `--familias todas` (orden alfabético, por defecto) | 0.652 | SVC | 0.655 / 0.604 |
| Con el plugin `knn` instalado (5 familias) | 0.622 | SVC | 0.655 / 0.604 |
| Solo `--familias knn` | 0.559 | KNeighborsClassifier | 0.621 / 0.534 |
| Baseline (clase mayoritaria) | — | — | 0.483 / 0.217 |

El pipeline dio exactamente lo mismo con los cuatro gestores de paquetes:

| Gestor | Ambiente | Python | Ejecutables creados | `make pipeline` |
|---|---|---|---|---|
| venv + pip | `.venv` | 3.14.2 | `act9-*.exe` (lanzador de pip, 108 KB) | 0.652 → 0.655 / 0.604 |
| conda (+ pip dentro) | `act9-mlops` | 3.12.14 | `act9-*.exe` | igual |
| uv (`uv run`, `uvx`) | `.venv-uv` / temporal | 3.14.2 | `act9-*.exe` (lanzador de uv, 47 KB) | igual |
| Poetry 2.5 (`poetry run`) | caché de Poetry | 3.14.2 | `act9-*` + `act9-*.cmd` | igual |

Dos lecciones salieron de estas tablas:

- **Instalar un plugin cambia el resultado.** `RandomizedSearchCV` reparte las 60 combinaciones
  entre las familias de la lista, así que agregar una (o cambiar el orden) cambia qué combinaciones
  se prueban. En producción conviene fijar `--familias` explícitamente.
- **Un mejor puntaje en CV no garantiza un mejor test.** El orden por defecto ganó en CV (0.652 vs
  0.619) pero perdió en test (0.604 vs 0.631). Con 29 partidos de prueba la diferencia es de un
  solo acierto (19 contra 20).

---

## Respuestas a las preguntas

### 1. ¿Qué usos tendría un entry point de Python en ambientes de producción?

- **Una interfaz estable para lanzar el código.** El Dockerfile, el cron, Airflow o el job de
  Kubernetes llaman `act9-entrenar`, no `python src/act9_pipeline/cli/entrenar.py`. Si mañana se
  mueve el código, solo cambia la línea del `pyproject.toml`. En la Actividad 4 el contenedor
  hacía `CMD ["python", "entrenar.py"]` con el script copiado a mano; con entry points sería
  `CMD ["act9-entrenar"]` después de instalar el wheel.
- **Un comando por etapa.** Cada paso del pipeline se programa, reintenta y escala por separado
  (un DAG de Airflow con un `BashOperator` por comando, un `CronJob` que solo corre
  `act9-predecir` cada jornada).
- **Códigos de salida como contrato.** Lo que devuelve la función es el código de salida del
  proceso. `act9-evaluar --umbral 0.5` devuelve 1 si el modelo no alcanza la métrica, y eso
  detiene make, GitHub Actions o el orquestador sin leer la salida: una compuerta de calidad.
- **Plugins.** Agregar modelos, conectores de datos o métricas sin tocar ni volver a desplegar el
  paquete principal: basta instalar otro paquete. Así funcionan los plugins de pytest (grupo
  `pytest11`), los comandos de Flask (`flask.commands`) y los backends de MLflow
  (`mlflow.tracking_store`).
- **Reproducibilidad.** El comando queda amarrado a la versión instalada del paquete (y del
  ambiente), no a un archivo suelto que alguien pudo modificar en el servidor.

### 2. ¿Se podría utilizar un entry point con un manejador de paquetes?

Sí, con todos. Los entry points son parte del estándar de empaquetado de Python
(`[project.scripts]` y `[project.entry-points]` en `pyproject.toml`, `entry_points.txt` en la
metadata), así que cualquier instalador que cumpla el estándar los lee. Lo probamos con los cuatro:

| Gestor | Cómo se declaran | Cómo se ejecutan | Lo que vimos |
|---|---|---|---|
| pip | `[project.scripts]` | Activar el ambiente y `act9-entrenar` | Crea `Scripts\act9-*.exe` |
| uv | `[project.scripts]` (necesita un `build-system`) | `uv run act9-entrenar`, `uvx --from . act9 info` | Crea el ambiente solo; con `uvx` en uno temporal |
| Poetry 2 | `[project.scripts]` (antes `[tool.poetry.scripts]`) | `poetry run act9-entrenar` | Usa el `.venv` del proyecto si existe; crea `act9-*` + `.cmd` |
| conda | Con pip dentro de `environment.yml`, o `build: entry_points:` en un `meta.yaml` de conda-build | `conda activate` o `conda run -n act9-mlops act9-entrenar` | Los ejecutables quedan en `envs\act9-mlops\Scripts` |
| pipx | `[project.scripts]` | `pipx install .` | Un ambiente aislado por herramienta (no lo probamos; `uvx` hace lo mismo) |

El mismo `pyproject.toml` sirvió para los cuatro sin cambiar una línea. Lo que cambia es dónde
queda el ambiente y qué tipo de ejecutable se crea.

### 3. ¿Qué relación podría tener un entry point con un Makefile?

Son capas que se complementan:

- **El entry point define *qué* se ejecuta** (la interfaz de cada etapa, dentro del paquete).
- **El Makefile define *cuándo y en qué orden*** (las dependencias entre etapas, fuera del
  paquete). Sus recetas se vuelven cortas y legibles (`act9-entrenar --n-iter $(N_ITER)`) y no
  dependen de rutas internas del código.
- **GitHub Actions solo llama al Makefile**, como en el tutorial de DataCamp (Awan, 2024): el
  mismo `make pipeline` corre igual en la laptop y en CI.

Además, el Makefile aporta lo que un entry point no tiene: **ejecución incremental** (solo repite
las etapas cuyas entradas cambiaron) y **variables** (`UMBRAL`, `N_ITER`, `RUN`). Y el entry point
aporta lo que el Makefile necesita: **códigos de salida** confiables para saber si debe detenerse.
El objetivo `install` es el punto de unión: es el que crea los comandos que usan los demás.

---

## Estructura del repositorio

```
Actividad-9-MLOPS/
├── pyproject.toml               # entry points: [project.scripts] y grupo act9_pipeline.modelos
├── setup.py                     # shim de compatibilidad
├── requirements.txt
├── environment.yml              # ambiente de conda (incluye GNU make)
├── Makefile                     # encadena los entry points
├── .github/workflows/pipeline.yml
├── src/act9_pipeline/
│   ├── cli/                     # una función main() por comando
│   │   ├── __init__.py          #   act9 (subcomandos)
│   │   ├── datos.py  entrenar.py  evaluar.py  predecir.py
│   │   ├── info.py              #   act9 info / act9 modelos
│   │   └── demo.py              #   act9-demo
│   ├── plugins.py               # descubrimiento con importlib.metadata
│   ├── modelos.py               # las 4 familias propias (plugins)
│   ├── tuning.py                # RandomizedSearchCV con las familias descubiertas
│   ├── data.py  transformers.py  pipeline.py   # de la Actividad 3
│   ├── evaluation.py  artefactos.py  ambiente.py
│   ├── __main__.py              # python -m act9_pipeline
│   └── datasets/champions_league_matches.csv
├── plugins/act9-modelo-knn/     # paquete externo que agrega la familia knn
├── tests/test_entrypoints.py    # 14 pruebas
├── notebooks/entrypoints_demo.ipynb
├── pipeline_diagram.html        # diagrama del pipeline ganador
├── dist/                        # wheel y sdist (el wheel trae entry_points.txt)
└── reporte/                     # reporte en LaTeX y PDF
```

## Limitaciones

- **make no se entera de los cambios en las variables.** `make entrenar N_ITER=20` no reentrena si
  el modelo ya existe, porque solo compara fechas de archivos. Hay que correr `make clean` o borrar
  el modelo.
- **En Windows make no viene instalado.** Lo resolvimos con el paquete `make` de conda-forge; el
  tutorial de DataCamp sugiere GnuWin32, pero esa versión (3.81) no soporta el objetivo agrupado
  `&:` que usa el Makefile (requiere make 4.3+).
- **Si la carpeta de scripts no está en el `PATH`, los comandos no existen**, aunque el paquete sí
  esté instalado. El plan B es `python -m act9_pipeline`.
- **Poetry reutiliza el `.venv` del proyecto si ya existe** y reemplaza los ejecutables que había
  creado pip. Conviene un ambiente por herramienta.
- El conjunto de prueba sigue siendo de 29 partidos: las diferencias de test entre modelos son de
  uno o dos aciertos.

## Referencias

- Awan, A. A. (2024). *GitHub Actions and MakeFile: A hands-on introduction*. DataCamp.
  https://www.datacamp.com/tutorial/makefile-github-actions-tutorial
- Python Packaging Authority. (s.f.). *Entry points specification*.
  https://packaging.python.org/en/latest/specifications/entry-points/
- Python Packaging Authority. (s.f.). *Creating and discovering plugins*.
  https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/
- Astral. (s.f.). *Configuring projects: Entry points*. Documentación de uv.
  https://docs.astral.sh/uv/concepts/projects/config/
- Poetry. (s.f.). *The pyproject.toml file*. https://python-poetry.org/docs/pyproject/
- conda-build. (s.f.). *Defining metadata (meta.yaml): Python entry points*.
  https://docs.conda.io/projects/conda-build/en/stable/resources/define-metadata.html
