"""Stage 3 — fetch: download the selected PSI files and external resources.

Files already on disk are skipped (cache), so a re-run only downloads what is missing.
PSI files: the API returns a temporary pre-signed S3 URL, which is used once and never stored.
Output: the files at each document's local_path.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed

from configs import PROJECT_ROOT
from configs.corpus.config_corpus import corpus_settings
from flame.common.http import download, get_text
from flame.common.io import read_jsonl
from flame.corpus.select import documents_path

_s = corpus_settings


def _fetch(doc: dict) -> tuple[str, str]:
    dest = PROJECT_ROOT / doc["local_path"]
    if dest.exists() and dest.stat().st_size > 0:
        return doc["doc_id"], "caché"
    if doc["source"] == "psi_file":
        url = get_text(f"{_s.sources.psi.api_base}/studies/{doc['accession']}/download",
                       {"file": doc["file_name"], "redirect": "false"})
        if not url.startswith("https://"):
            raise RuntimeError(f"respuesta inesperada de la API: {url[:120]}")
    else:
        url = doc["public_url"]
    size = download(url, dest)
    return doc["doc_id"], f"{size / 1e6:.2f} MB"


def run() -> None:
    docs = [d for d in read_jsonl(documents_path()) if d["local_path"]]
    print(f"  {len(docs)} archivos por descargar o verificar")
    failures: list[tuple[dict, Exception]] = []
    done = cached = 0
    with ThreadPoolExecutor(max_workers=_s.network.concurrency) as pool:
        futures = {pool.submit(_fetch, d): d for d in docs}
        for i, fut in enumerate(as_completed(futures), 1):
            doc = futures[fut]
            try:
                _, status = fut.result()
                cached += status == "caché"
                done += 1
            except Exception as e:
                failures.append((doc, e))
            if i % 25 == 0 or i == len(docs):
                print(f"  {i}/{len(docs)} procesados ({cached} en caché, {len(failures)} errores)")
    if failures:
        failures.sort(key=lambda item: item[0]["doc_id"])
        for doc, e in failures:
            print(f"  ERROR {doc['doc_id']} ({doc['title'][:60]}): {e}")
        # A 404 does not go away on a re-run (the file is gone from PSI): print the exact entries for the
        # selection.exclude list in corpus.yml.
        print("  Si un archivo da 404 siempre, agrégalo a selection.exclude en configs/corpus/corpus.yml:")
        for doc, _ in failures:
            print(f'    - "{doc.get("remote_path") or doc["doc_id"].removeprefix("ext:")}"')
        raise RuntimeError(f"{len(failures)} descargas fallaron. Vuelve a correr 'fetch': "
                           f"solo se descarga lo que falta.")
