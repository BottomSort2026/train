"""Stage 1 — inventory: download PSI metadata and the current file list of every configured investigation.

Output (cached; existing files are not downloaded again):
    <raw>/inventory/<accession>.json        investigation metadata
    <raw>/inventory/<accession>_files.json  current files (versions applied)
"""
from concurrent.futures import ThreadPoolExecutor, as_completed

from configs.corpus.config_corpus import corpus_settings
from flame.common.http import get_json
from flame.common.io import read_json, rel, write_json

_s = corpus_settings


def inventory_dir():
    return _s.paths.raw / "inventory"


def current_files(versions: dict) -> list[dict]:
    """Apply added / modified / removed across versions to get the current file set."""
    def key(f: dict) -> str:
        return f.get("remote_path") or f.get("s3_path") or f.get("file_name")

    current: dict[str, dict] = {}
    for ver in versions.get("versions", []):
        for f in ver.get("added", []) + ver.get("modified", []):
            if isinstance(f, dict):
                current[key(f)] = f
        for f in ver.get("removed", []):
            if isinstance(f, dict):
                current.pop(key(f), None)
    return list(current.values())


def _fetch_one(accession: str) -> tuple[str, int]:
    meta_path = inventory_dir() / f"{accession}.json"
    files_path = inventory_dir() / f"{accession}_files.json"
    api = _s.sources.psi.api_base
    if not meta_path.exists():
        write_json(meta_path, get_json(f"{api}/repo/investigations/{accession}"))
    if not files_path.exists():
        write_json(files_path, current_files(get_json(f"{api}/repo/investigations/{accession}/versions")))
    return accession, len(read_json(files_path))


def load_metadata(accession: str) -> dict:
    return read_json(inventory_dir() / f"{accession}.json")


def load_files(accession: str) -> list[dict]:
    return read_json(inventory_dir() / f"{accession}_files.json")


def run() -> None:
    accessions = [i.accession for i in _s.sources.psi.investigations]
    print(f"  {len(accessions)} investigaciones -> {rel(inventory_dir())}")
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=_s.network.concurrency) as pool:
        futures = {pool.submit(_fetch_one, a): a for a in accessions}
        for fut in as_completed(futures):
            acc = futures[fut]
            try:
                _, n = fut.result()
                print(f"  {acc:8} {n:6} archivos")
            except Exception as e:  # keep going; report all failures at the end
                failures.append(f"{acc}: {e}")
                print(f"  {acc:8} ERROR {e}")
    if failures:
        raise RuntimeError(f"{len(failures)} investigaciones fallaron (vuelve a correr la etapa; "
                           f"las que terminaron quedan en caché): {failures}")
