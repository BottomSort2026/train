"""`corpus eval`: measure search quality on eval/questions.yml.

For each question and each method (BM25, vector, hybrid), a hit means one of the expected documents appears
in the top k. Reports recall@k (share of questions with a hit) and MRR (mean of 1/rank of the first hit).
"""
from pathlib import Path

import yaml

from configs import PROJECT_ROOT
from configs.corpus.config_corpus import corpus_settings
from flame.common.io import rel
from flame.corpus.embed import embedding_config
from flame.corpus.embedding import Embedder, check_compatible
from flame.corpus.load import connect
from flame.corpus.search import fetch, hybrid_search, read_info

DEFAULT_QUESTIONS = PROJECT_ROOT / "eval" / "questions.yml"
METHODS = ("bm25", "vector", "hybrid")


def _matches(chunk: dict, expected) -> bool:
    if isinstance(expected, str):
        return chunk["doc_id"] == expected
    return all(chunk.get(k) == v for k, v in expected.items())


def first_hit_rank(chunks: list[dict], expected: list) -> int | None:
    for rank, c in enumerate(chunks, 1):
        if any(_matches(c, e) for e in expected):
            return rank
    return None


def run(questions_path: Path | None = None, k: int | None = None) -> None:
    path = Path(questions_path) if questions_path else DEFAULT_QUESTIONS
    questions = yaml.safe_load(path.read_text(encoding="utf-8"))["questions"]
    db = corpus_settings.paths.output_db
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
        k = k or info["search"]["k"]

        results = []  # (question, level, {method: rank or None})
        for q in questions:
            res = hybrid_search(conn, embedder, q["query"], info, k)
            ranked = {"bm25": res["bm25"], "vector": [r for r, _ in res["vector"]],
                      "hybrid": [r for r, _ in res["hybrid"]]}
            chunks = fetch(conn, list({r for ids in ranked.values() for r in ids}))
            ranks = {m: first_hit_rank([chunks[r] for r in ids], q["expected"]) for m, ids in ranked.items()}
            results.append((q, q.get("level", "-"), ranks))
    finally:
        conn.close()

    print(f"  {len(results)} preguntas de {rel(path)}, k={k} (corpus {info['built_at']})\n")
    print(f"  {'bm25':>5} {'vector':>6} {'hybrid':>6}  pregunta")
    for q, level, ranks in results:
        cells = [f"{ranks[m] if ranks[m] else '✗':>{w}}" for m, w in zip(METHODS, (5, 6, 6))]
        print(f"  {' '.join(cells)}  [{level}] {q['query'][:70]}")

    print(f"\n  {'nivel':10} {'método':>6}  {'recall@' + str(k):>9} {'MRR':>6}")
    for level in sorted({lv for _, lv, _ in results}) + ["TOTAL"]:
        subset = [r for _, lv, r in results if level == "TOTAL" or lv == level]
        for m in METHODS:
            hits = [r[m] for r in subset]
            recall = sum(h is not None for h in hits) / len(hits)
            mrr = sum(1 / h for h in hits if h) / len(hits)
            print(f"  {level:10} {m:>6}  {recall:>9.0%} {mrr:>6.2f}")
    misses = [q["query"] for q, _, r in results if r["hybrid"] is None]
    if misses:
        print("\n  Sin acierto en la búsqueda híbrida (revisa si falta el documento o si el ruido lo desplaza):")
        for m in misses:
            print(f"    - {m}")
