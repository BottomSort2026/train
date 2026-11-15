"""Embedding module SHARED by the corpus build (train repo) and the engine service (/embed).

Copy this file VERBATIM into the engine repo. It is self-contained: it depends only on
numpy and sentence-transformers, and takes its configuration as parameters.
Both sides must use the same model, revision, normalization, max length and prefixes, and the same
library versions. `check_compatible` compares a configuration against the one stored in corpus.db.
"""
from dataclasses import asdict, dataclass
from pathlib import PurePath

import numpy as np

# Fields that define the vector space. If any differs between build and query time, search breaks silently.
CONTRACT_FIELDS = ("model_id", "revision", "dimension", "normalize", "max_length", "query_prefix", "document_prefix")


@dataclass(frozen=True)
class EmbeddingConfig:
    model_id: str
    revision: str | None
    dimension: int
    normalize: bool
    max_length: int
    query_prefix: str
    document_prefix: str
    batch_size: int = 32
    device: str = "cpu"

    @classmethod
    def from_dict(cls, d: dict) -> "EmbeddingConfig":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})

    def contract(self) -> dict:
        return {k: getattr(self, k) for k in CONTRACT_FIELDS}


def check_compatible(config: EmbeddingConfig, stored_contract: dict) -> list[str]:
    """Return the list of mismatches between `config` and the contract stored in corpus.db (empty = OK).

    A stored revision of None means "not pinned": it only matches a config whose revision is also None.
    """
    mine = config.contract()
    return [f"{k}: corpus={stored_contract.get(k)!r} vs config={mine[k]!r}"
            for k in CONTRACT_FIELDS if stored_contract.get(k) != mine[k]]


class Embedder:
    def __init__(self, config: EmbeddingConfig):
        from sentence_transformers import SentenceTransformer

        self.config = config
        self.model = SentenceTransformer(config.model_id, revision=config.revision, device=config.device)
        self.model.max_seq_length = config.max_length
        # Newer sentence-transformers renamed get_sentence_embedding_dimension -> get_embedding_dimension.
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        dim = get_dim()
        if dim != config.dimension:
            raise ValueError(f"El modelo {config.model_id} produce {dim} dimensiones, la configuración dice {config.dimension}")

    @property
    def resolved_revision(self) -> str | None:
        """Commit hash of the model version actually loaded (use it to pin `revision`).

        Read from the local Hugging Face cache (works offline): a cached file lives in
        .../models--<org>--<name>/snapshots/<commit hash>/. Falls back to the transformers config attribute,
        which older library versions filled in.
        """
        try:
            from huggingface_hub import try_to_load_from_cache

            cached = try_to_load_from_cache(self.config.model_id, "config.json", revision=self.config.revision)
            if isinstance(cached, str):
                parts = PurePath(cached).parts
                if "snapshots" in parts:
                    return parts[parts.index("snapshots") + 1]
        except Exception:
            pass
        try:
            return getattr(self.model[0].auto_model.config, "_commit_hash", None)
        except Exception:
            return None

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Fragments (corpus build). Returns float32 array of shape (n, dimension)."""
        return self._encode([self.config.document_prefix + t for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        """Search query (engine /embed). The query prefix is applied HERE, never by the caller."""
        return self._encode([self.config.query_prefix + text])[0]
