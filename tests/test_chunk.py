"""Chunking tests with a whitespace token counter (no model download needed)."""
from flame.corpus.chunk import context_header, join_units, pack, to_units


def count(text: str) -> int:
    return len(text.split())


def _pages(n_sentences: int, words: int = 12) -> list[dict]:
    # Sentences start with a capital letter, as in real text (the sentence splitter relies on it).
    sentences = [" ".join(f"W{i}x{j}" for j in range(words)) + "." for i in range(n_sentences)]
    return [{"page": 1, "text": " ".join(sentences)}]


def test_no_fragment_exceeds_max_tokens():
    units = to_units(_pages(100), count, max_tokens=50)
    for group in pack(units, max_tokens=50, overlap_tokens=15):
        assert count(join_units(group)) <= 50


def test_consecutive_fragments_overlap():
    units = to_units(_pages(30), count, max_tokens=50)
    groups = pack(units, max_tokens=50, overlap_tokens=15)
    assert len(groups) > 1
    for prev, nxt in zip(groups, groups[1:]):
        assert prev[-1].text == nxt[0].text


def test_all_text_is_covered():
    pages = _pages(40)
    units = to_units(pages, count, max_tokens=50)
    covered = " ".join(join_units(g) for g in pack(units, max_tokens=50, overlap_tokens=15))
    for word in pages[0]["text"].replace(".", "").split():
        assert word in covered


def test_header_drops_missing_values():
    header = context_header("[{acronym} | {title} | p.{page}]", {"acronym": None, "title": "Doc", "page": None},
                            count, max_tokens=60)
    assert header == "[Doc]"
