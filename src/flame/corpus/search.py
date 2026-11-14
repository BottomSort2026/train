"""Reference hybrid search over corpus.db (BM25 + vectors + reciprocal rank fusion).

This is the behavior the MCP (TypeScript) must reproduce. It reads its parameters from corpus_info,
not from corpus.yml, exactly as the MCP should.
"""
import json
import re
import sqlite3

from configs.corpus.config_corpus import corpus_settings
from flame.common.io import rel
from flame.corpus.embed import embedding_config
from flame.corpus.embedding import Embedder, check_compatible
from flame.corpus.load import connect

_s = corpus_settings
_TERM = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-+.]*[A-Za-z0-9+]|[A-Za-z0-9]")


def read_info(conn: sqlite3.Connection) -> dict:
    return {k: json.loads(v) for k, v in conn.execute("SELECT key, value FROM corpus_info")}


def fts_query(text: str) -> str | None:
    """Natural-language query -> FTS5 query: each term quoted (so '-', ':' etc. are literal), joined with OR."""
    terms = list(dict.fromkeys(t.lower() for t in _TERM.findall(text)))
    if not terms:
        return None
    return " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)


def bm25_search(conn, query: str, n: int) -> list[int]:
    q = fts_query(query)
    if q is None:
        return []
    rows = conn.execute(
        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?", (q, n))
    return [r[0] for r in rows]


def vector_search(conn, query_vector, n: int) -> list[tuple[int, float]]:
    rows = conn.execute(
        "SELECT rowid, distance FROM chunks_vec WHERE embedding MATCH ? AND k = ? ORDER BY distance",
        (query_vector.astype("<f4").tobytes(), n))
    return [(r[0], 1 - r[1] ** 2 / 2) for r in rows]  # cosine similarity


def rrf(rankings: list[list[int]], k: int) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, rowid in enumerate(ranking, 1):
            scores[rowid] = scores.get(rowid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def fetch(conn, rowids: list[int]) -> dict[int, dict]:
    if not rowids:
        return {}
    marks = ",".join("?" * len(rowids))
    rows = conn.execute(f"""
        SELECT c.id, c.chunk_id, c.page, c.text, d.doc_id, d.doc_type, d.title, d.authors, d.year,
               d.public_url, d.accession, i.acronym, i.investigation_url, i.fire_relevance
        FROM chunks c JOIN documents d USING (doc_id) LEFT JOIN investigations i USING (accession)
        WHERE c.id IN ({marks})""", rowids)
    cols = [c[0] for c in rows.description]
    return {r[0]: dict(zip(cols, r)) for r in rows}


def hybrid_search(conn, embedder: Embedder, query: str, info: dict, k: int | None = None) -> dict:
    s = info["search"]
    k = k or s["k"]
    n = max(s["candidates_per_method"], k)
    bm25 = bm25_search(conn, query, n)
    vec = vector_search(conn, embedder.embed_query(query), n)
    fused = rrf([bm25, [r for r, _ in vec]], s["rrf_k"])
    return {"bm25": bm25[:k], "vector": vec[:k], "hybrid": fused[:k]}


def _print_results(title: str, rowids: list[int], scores: list[float | None], chunks: dict[int, dict]) -> None:
    print(f"\n  === {title} ===")
    if not rowids:
        print("  (sin resultados)")
    for rank, (rowid, score) in enumerate(zip(rowids, scores), 1):
        c = chunks[rowid]
        where = c["acronym"] or c["accession"] or "externo"
        page = f" p.{c['page']}" if c["page"] else ""
        sc = f"{score:.3f}" if score is not None else "  -  "
        snippet = re.sub(r"\s+", " ", c["text"])[:160]
        print(f"  {rank}. [{sc}] {where} | {c['doc_type']} | {c['title'][:70]}{page}")
        print(f"     {snippet}…")
        print(f"     {c['public_url'] or c['investigation_url']}")


def run(query: str, k: int | None = None) -> None:
    db = _s.paths.output_db
    if not db.exists():
        raise FileNotFoundError(f"No existe {rel(db)}. Corre 'build' o 'load' primero.")
    conn = connect(db)
    try:
        info = read_info(conn)
        config = embedding_config()
        mismatches = check_compatible(config, info["embedding"])
        if mismatches:
            raise RuntimeError("La configuración de embedding no coincide con corpus.db: " + "; ".join(mismatches))
        embedder = Embedder(config)
        res = hybrid_search(conn, embedder, query, info, k)
        ids = set(res["bm25"]) | {r for r, _ in res["vector"]} | {r for r, _ in res["hybrid"]}
        chunks = fetch(conn, list(ids))
        print(f"  Consulta: {query!r}  (corpus {info['built_at']}, {info['counts']['chunks']} fragmentos)")
        _print_results("BM25 (palabras clave)", res["bm25"], [None] * len(res["bm25"]), chunks)
        _print_results("Vectorial (similitud coseno)", [r for r, _ in res["vector"]], [s for _, s in res["vector"]], chunks)
        _print_results("Híbrida (RRF)", [r for r, _ in res["hybrid"]], [s for _, s in res["hybrid"]], chunks)
    finally:
        conn.close()
