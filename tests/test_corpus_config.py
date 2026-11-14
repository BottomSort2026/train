import dataclasses

import pytest

from configs.corpus.config_corpus import Chunking, ContextHeader, corpus_settings


def test_config_loads():
    s = corpus_settings
    assert s.sources.psi.investigations
    assert s.embedding.dimension > 0
    assert s.paths.output_db.name == "corpus.db"


def test_fragment_plus_header_fits_model():
    s = corpus_settings
    header = s.chunking.context_header.max_tokens if s.chunking.context_header.enabled else 0
    assert s.chunking.max_tokens + header + 2 <= s.embedding.max_length


def test_overlap_must_be_smaller_than_fragment():
    with pytest.raises(ValueError):
        Chunking(max_tokens=100, overlap_tokens=100, no_split_doc_types=(),
                 context_header=ContextHeader(enabled=False, template="", max_tokens=0))


def test_oversized_fragment_is_rejected():
    s = corpus_settings
    too_big = dataclasses.replace(s.chunking, max_tokens=s.embedding.max_length)
    with pytest.raises(ValueError):
        dataclasses.replace(s, chunking=too_big)
