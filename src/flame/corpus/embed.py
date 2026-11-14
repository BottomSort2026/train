"""Stage 6 — embed: compute the embedding of every fragment (its embed_text).

Output:
    <processed>/embeddings.npy          float32 array (n_chunks, dimension), same order as the ids file
    <processed>/embedding_ids.json      chunk_ids in the same order
    <processed>/embedding_meta.json     contract + resolved model revision
"""
import time

import numpy as np

from configs.corpus.config_corpus import corpus_settings
from flame.common.io import read_jsonl, rel, write_json
from flame.corpus.chunk import chunks_path
from flame.corpus.embedding import Embedder, EmbeddingConfig

_s = corpus_settings


def embedding_config() -> EmbeddingConfig:
    em = _s.embedding
    return EmbeddingConfig(model_id=em.model_id, revision=em.revision, dimension=em.dimension,
                           normalize=em.normalize, max_length=em.max_length, query_prefix=em.query_prefix,
                           document_prefix=em.document_prefix, batch_size=em.batch_size, device=em.device)


def embeddings_path():
    return _s.paths.processed / "embeddings.npy"


def ids_path():
    return _s.paths.processed / "embedding_ids.json"


def meta_path():
    return _s.paths.processed / "embedding_meta.json"


def run() -> None:
    chunks = list(read_jsonl(chunks_path()))
    if not chunks:
        raise RuntimeError("No hay fragmentos. Corre 'chunk' primero.")
    config = embedding_config()
    print(f"  Cargando {config.model_id} (revision={config.revision}) en {config.device}...")
    embedder = Embedder(config)

    start, step = time.time(), 512
    parts = []
    for i in range(0, len(chunks), step):
        batch = chunks[i:i + step]
        parts.append(embedder.embed_documents([c["embed_text"] for c in batch]))
        print(f"  {min(i + step, len(chunks))}/{len(chunks)} fragmentos ({time.time() - start:.0f} s)")
    vectors = np.vstack(parts)

    if vectors.shape != (len(chunks), config.dimension) or not np.isfinite(vectors).all():
        raise RuntimeError(f"Embeddings inválidos: forma {vectors.shape}")

    embeddings_path().parent.mkdir(parents=True, exist_ok=True)
    tmp = embeddings_path().with_name("embeddings.tmp.npy")
    np.save(tmp, vectors)
    tmp.replace(embeddings_path())
    write_json(ids_path(), [c["chunk_id"] for c in chunks])
    write_json(meta_path(), {"contract": config.contract(), "resolved_revision": embedder.resolved_revision})
    print(f"  {vectors.shape[0]} vectores de {vectors.shape[1]} dimensiones -> {rel(embeddings_path())}")
    if config.revision is None and embedder.resolved_revision:
        print(f"  AVISO: embedding.revision no está fijado. La versión descargada es {embedder.resolved_revision}; "
              f"cópiala en corpus.yml (y en engine) para fijarla.")
