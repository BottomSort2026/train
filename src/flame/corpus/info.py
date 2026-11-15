"""`corpus info`: what corpus.db contains and which versions to pin (in corpus.yml and in the engine service)."""
import json
from importlib import metadata

from configs.corpus.config_corpus import corpus_settings
from flame.common.io import rel

_PACKAGES = ("torch", "sentence-transformers", "transformers", "sqlite-vec", "pymupdf", "numpy")


def run() -> None:
    s = corpus_settings
    manifest = s.paths.manifest
    if manifest.exists():
        info = json.loads(manifest.read_text(encoding="utf-8"))
        em = info["embedding"]
        print(f"  corpus.db ({rel(s.paths.output_db)}), construido {info['built_at']}, etapas {info['build_stages']}")
        print(f"  {info['counts']['documents']} documentos, {info['counts']['chunks']} fragmentos, "
              f"sqlite-vec {info['sqlite_vec_version']}")
        print(f"  Modelo: {em['model_id']}  revision en corpus.yml = {em['revision']}")
        print(f"          revision descargada  = {em.get('resolved_revision')}")
        if em["revision"] is None and em.get("resolved_revision"):
            print(f"  -> Para fijarla: embedding.revision: {em['resolved_revision']}  (en corpus.yml y en engine)")
    else:
        print(f"  Todavía no existe {rel(manifest)}. Corre 'build'.")

    print("  Versiones instaladas (engine debe usar las mismas para torch y sentence-transformers):")
    for pkg in _PACKAGES:
        print(f"    {pkg:22} {_version(pkg)}")


def _version(pkg: str) -> str:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        pass
    # some conda-forge builds register a different distribution name (e.g. PyMuPDF): ask the module itself
    module = {"pymupdf": "pymupdf", "sentence-transformers": "sentence_transformers"}.get(pkg, pkg.replace("-", "_"))
    try:
        return str(getattr(__import__(module), "__version__", "instalado (versión desconocida)"))
    except ImportError:
        return "(no instalado)"
