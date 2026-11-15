"""Stage 7 — load: build corpus.db (SQLite + FTS5 + sqlite-vec) and manifest.json.

corpus.db is built as corpus.db.tmp and replaces the previous file ONLY if the build finishes,
so a failed build never leaves a half-built database.
"""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from configs.corpus.config_corpus import corpus_settings
from flame.common.io import read_json, read_jsonl, rel, write_json
from flame.corpus.chunk import chunks_path
from flame.corpus.embed import embeddings_path, ids_path, meta_path
from flame.corpus.inventory import load_metadata
from flame.corpus.select import documents_path

_s = corpus_settings
SCHEMA_VERSION = 1
SCHEMA_FILE = Path(__file__).with_name("schema.sql")
# metadata_json keeps ONLY these descriptive fields (an allow-list). The raw PSI record also carries file
# lists (added / removed / modified, megabytes per investigation), legacy file URLs and S3 paths, which would
# flood the LLM context through get_investigation and must never be cited.
_METADATA_FIELDS = (
    "accession", "title", "investigationAcronym", "researchArea", "subResearchArea", "projectType",
    "proposalTitle", "flightProgram", "spaceProgram", "managingNasaCenter", "missionName",
    "experimentHardware", "experimentHardwareLink", "investigationStartDate", "investigationEndDate",
    "objective", "hypothesis", "approach", "researchImpacts", "description", "funding",
    "doi", "licenseName", "licenseIdentifier", "licenseLink", "releaseDate",
)
_CONTACT_FIELDS = ("firstName", "midInitials", "lastName", "affiliation")  # no email, phone, fax or address


def connect(path: Path) -> sqlite3.Connection:
    import sqlite_vec

    conn = sqlite3.connect(path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def public_metadata(meta: dict) -> dict:
    """Descriptive PSI metadata for investigations.metadata_json (allow-list; no file lists, no contact details)."""
    out = {k: meta[k] for k in _METADATA_FIELDS if meta.get(k) not in (None, "", [])}
    people = []
    for c in meta.get("contacts", []):
        person = {k: c[k] for k in _CONTACT_FIELDS if c.get(k)}
        roles = [r.get("annotationValue") for r in c.get("roles", []) if r.get("annotationValue")]
        if roles:
            person["roles"] = roles
        if person:
            people.append(person)
    if people:
        out["investigators"] = people
    pubs = [{k: p[k] for k in ("title", "authorList", "doi") if p.get(k)} for p in meta.get("publications", [])]
    pubs = [p for p in pubs if p.get("title")]
    if pubs:
        out["publications"] = pubs
    return out


def run() -> None:
    em, se = _s.embedding, _s.search
    docs = list(read_jsonl(documents_path()))
    chunks = list(read_jsonl(chunks_path()))
    vectors = np.load(embeddings_path())
    ids = read_json(ids_path())
    emb_meta = read_json(meta_path())

    # --- consistency checks before writing anything ---
    if ids != [c["chunk_id"] for c in chunks]:
        raise RuntimeError("Los embeddings no corresponden a los fragmentos actuales. Corre 'embed' de nuevo.")
    if vectors.shape != (len(chunks), em.dimension):
        raise RuntimeError(f"Forma de embeddings {vectors.shape} no coincide con {len(chunks)} x {em.dimension}")
    current_contract = {"model_id": em.model_id, "revision": em.revision, "dimension": em.dimension,
                        "normalize": em.normalize, "max_length": em.max_length,
                        "query_prefix": em.query_prefix, "document_prefix": em.document_prefix}
    if emb_meta["contract"] != current_contract:
        raise RuntimeError("La configuración de embedding cambió desde 'embed'. Corre 'embed' de nuevo.")

    out = _s.paths.output_db
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.unlink(missing_ok=True)

    conn = connect(tmp)
    try:
        schema = SCHEMA_FILE.read_text(encoding="utf-8")
        schema = schema.replace("{fts_tokenizer}", se.fts_tokenizer).replace("{dimension}", str(em.dimension))
        conn.executescript(schema)

        url_tpl = _s.sources.psi.investigation_url
        conn.executemany(
            "INSERT INTO investigations VALUES (?, ?, ?, ?, ?, ?)",
            [(inv.accession, inv.acronym, load_metadata(inv.accession).get("title") or inv.accession,
              inv.fire_relevance, url_tpl.format(accession=inv.accession),
              json.dumps(public_metadata(load_metadata(inv.accession)), ensure_ascii=False))
             for inv in _s.sources.psi.investigations],
        )
        conn.executemany(
            "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(d["doc_id"], d["source"], d["accession"], json.dumps(d["related_accessions"]), d["doc_type"],
              d["ingest_stage"], d["title"], d["authors"], d["year"], d["file_name"], d["psi_folder"],
              d["public_url"]) for d in docs],
        )
        conn.executemany(
            "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [(i, c["chunk_id"], c["doc_id"], c["chunk_index"], c["page"], c["page_end"], c["text"], c["n_tokens"])
             for i, c in enumerate(chunks, 1)],
        )
        conn.execute("INSERT INTO chunks_fts(rowid, text) SELECT id, text FROM chunks")
        conn.executemany(
            "INSERT INTO chunks_vec(rowid, embedding) VALUES (?, ?)",
            [(i, vectors[i - 1].astype("<f4").tobytes()) for i in range(1, len(chunks) + 1)],
        )

        vec_version = conn.execute("SELECT vec_version()").fetchone()[0]
        info = {
            "schema_version": SCHEMA_VERSION,
            "corpus_name": _s.corpus.name,
            "description": _s.corpus.description,
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "build_stages": list(_s.selection.build_stages),
            "embedding": {**current_contract, "resolved_revision": emb_meta.get("resolved_revision")},
            "search": {"k": se.k, "candidates_per_method": se.candidates_per_method, "fusion": se.fusion,
                       "rrf_k": se.rrf_k, "fts_tokenizer": se.fts_tokenizer},
            "counts": {"investigations": len(_s.sources.psi.investigations), "documents": len(docs),
                       "chunks": len(chunks)},
            "sqlite_vec_version": vec_version,
            "vector_distance": "L2 on L2-normalized vectors; cosine similarity = 1 - distance^2 / 2",
        }
        conn.executemany("INSERT INTO corpus_info VALUES (?, ?)",
                         [(k, json.dumps(v, ensure_ascii=False)) for k, v in info.items()])
        conn.commit()

        # --- self-check: a stored vector must find itself first ---
        probe = vectors[0].astype("<f4").tobytes()
        hit = conn.execute("SELECT rowid FROM chunks_vec WHERE embedding MATCH ? AND k = 1", (probe,)).fetchone()
        if not hit or hit[0] != 1:
            raise RuntimeError("Autoverificación fallida: la búsqueda vectorial no se encuentra a sí misma")
        n_fts = conn.execute("SELECT count(*) FROM chunks_fts").fetchone()[0]
        if n_fts != len(chunks):
            raise RuntimeError(f"Autoverificación fallida: FTS5 tiene {n_fts} filas, se esperaban {len(chunks)}")
    finally:
        conn.close()

    tmp.replace(out)
    write_json(_s.paths.manifest, info)
    size_mb = out.stat().st_size / 1e6
    print(f"  corpus.db: {len(docs)} documentos, {len(chunks)} fragmentos, {size_mb:.1f} MB -> {rel(out)}")
    print(f"  manifest -> {rel(_s.paths.manifest)} (sqlite-vec {vec_version})")
