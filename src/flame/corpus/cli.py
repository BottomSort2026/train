"""`python main.py corpus <command>`: one command per stage, plus `build` (all stages) and `search`.

Each stage prints when it starts and ends. If it fails, the message names the stage, so you can fix the
problem and re-run from that stage by hand.
"""
import argparse
import time
import traceback
from importlib import import_module

# Ordered pipeline: (command, module, help). Modules are imported lazily so `--help` stays fast.
STAGES = [
    ("inventory", "flame.corpus.inventory", "Descarga metadatos y listas de archivos de PSI (con caché)"),
    ("select", "flame.corpus.select", "Selecciona automáticamente los documentos con las reglas de corpus.yml"),
    ("fetch", "flame.corpus.fetch", "Descarga los documentos seleccionados (con caché)"),
    ("extract", "flame.corpus.extract", "Extrae el texto de cada documento, página por página"),
    ("chunk", "flame.corpus.chunk", "Divide el texto en fragmentos"),
    ("embed", "flame.corpus.embed", "Calcula los embeddings de los fragmentos"),
    ("load", "flame.corpus.load", "Construye corpus.db y manifest.json"),
]


def run_stage(index: int, name: str, module: str, verbose: bool) -> bool:
    print(f"▶ [{index}/{len(STAGES)}] {name}")
    start = time.time()
    try:
        import_module(module).run()
    except Exception as e:
        print(f"✘ La etapa '{name}' falló: {e}")
        if verbose:
            traceback.print_exc()
        print(f"  Corrige el problema y vuelve a correr: python main.py corpus {name}")
        return False
    print(f"✔ {name} ({time.time() - start:.1f} s)")
    return True


def _stage_command(index: int, name: str, module: str):
    def handler(args: argparse.Namespace) -> int:
        return 0 if run_stage(index, name, module, args.verbose) else 1
    return handler


def _build(args: argparse.Namespace) -> int:
    for index, (name, module, _) in enumerate(STAGES, 1):
        if not run_stage(index, name, module, args.verbose):
            remaining = [n for n, _, _ in STAGES[index:]]
            if remaining:
                print(f"  Etapas pendientes después de '{name}': {', '.join(remaining)}")
            return 1
    return 0


def _search(args: argparse.Namespace) -> int:
    from flame.corpus import search

    try:
        search.run(args.query, args.k)
    except Exception as e:
        print(f"✘ search falló: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1
    return 0


def register(commands: argparse._SubParsersAction) -> None:
    for index, (name, module, help_text) in enumerate(STAGES, 1):
        p = commands.add_parser(name, help=help_text)
        p.add_argument("-v", "--verbose", action="store_true", help="muestra el traceback completo si falla")
        p.set_defaults(func=_stage_command(index, name, module))

    p = commands.add_parser("build", help="Corre todas las etapas en orden y se detiene en la primera que falle")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=_build)

    p = commands.add_parser("search", help="Búsqueda de prueba sobre corpus.db (consulta en inglés)")
    p.add_argument("query", help='por ejemplo: "how does CO2 dilution suppress droplet flames"')
    p.add_argument("-k", type=int, default=None, help="cantidad de resultados (por defecto: search.k)")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=_search)
