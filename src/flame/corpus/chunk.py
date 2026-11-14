"""Stage 5 — chunk: split each document's text into fragments of at most `max_tokens` tokens.

- Sizes are measured with the embedding model's own tokenizer.
- Text is split into sentences (keeping paragraph breaks) and sentences are packed into fragments.
  Consecutive fragments repeat the last sentences of the previous one (`overlap_tokens`).
- `embed_text` = context header + text. It is used ONLY to compute the embedding; `text` is what gets stored.

Output: <processed>/chunks.jsonl with rows
    {chunk_id, doc_id, chunk_index, page, page_end, text, embed_text, n_tokens}
"""
import re
from dataclasses import dataclass
from typing import Callable

from configs.corpus.config_corpus import corpus_settings
from flame.common.io import read_jsonl, rel, write_jsonl
from flame.corpus.extract import pages_path
from flame.corpus.select import documents_path

_s = corpus_settings
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_PLACEHOLDER = re.compile(r"{(\w+)}")

TokenCounter = Callable[[str], int]


@dataclass
class Unit:
    text: str
    page: int | None
    para_start: bool
    n_tokens: int


def hf_token_counter() -> TokenCounter:
    """Token counter backed by the embedding model's tokenizer (same model_id and revision as the embedder)."""
    from transformers import AutoTokenizer

    em = _s.embedding
    tok = AutoTokenizer.from_pretrained(em.model_id, revision=em.revision)
    tok.model_max_length = 10**9  # we only count tokens here; avoids "sequence too long" warnings

    def count(text: str) -> int:
        return len(tok(text, add_special_tokens=False)["input_ids"])

    return count


def _split_long(text: str, count: TokenCounter, max_tokens: int) -> list[str]:
    """Split a sentence longer than max_tokens by words (rare: tables, lists without punctuation)."""
    pieces, current = [], []
    for word in text.split():
        candidate = " ".join(current + [word])
        if current and count(candidate) > max_tokens:
            pieces.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        pieces.append(" ".join(current))
    return pieces


def to_units(pages: list[dict], count: TokenCounter, max_tokens: int) -> list[Unit]:
    units: list[Unit] = []
    for page in pages:
        for paragraph in page["text"].split("\n\n"):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            first = True
            for sentence in _SENTENCE_END.split(paragraph):
                sentence = sentence.strip()
                if not sentence:
                    continue
                for piece in _split_long(sentence, count, max_tokens) if count(sentence) > max_tokens else [sentence]:
                    units.append(Unit(piece, page["page"], first, count(piece)))
                    first = False
    return units


def join_units(units: list[Unit]) -> str:
    out = []
    for i, u in enumerate(units):
        if i:
            out.append("\n\n" if u.para_start else " ")
        out.append(u.text)
    return "".join(out)


def pack(units: list[Unit], max_tokens: int, overlap_tokens: int) -> list[list[Unit]]:
    """Greedy packing of units into fragments, with a sentence-level overlap between consecutive fragments."""
    chunks: list[list[Unit]] = []
    current: list[Unit] = []
    size = 0
    for u in units:
        if current and size + u.n_tokens > max_tokens:
            chunks.append(current)
            tail, tail_size = [], 0
            for v in reversed(current):
                if tail_size + v.n_tokens > overlap_tokens:
                    break
                tail.insert(0, v)
                tail_size += v.n_tokens
            current, size = (tail, tail_size) if tail_size + u.n_tokens <= max_tokens else ([], 0)
        current.append(u)
        size += u.n_tokens
    if current:
        chunks.append(current)
    return chunks


def context_header(template: str, values: dict, count: TokenCounter, max_tokens: int) -> str:
    """Render e.g. '[{acronym} | {title} | p.{page}]', dropping segments whose values are missing."""
    inner = template.strip()
    opened, closed = inner.startswith("["), inner.endswith("]")
    inner = inner.strip("[]")
    segments = []
    for seg in inner.split(" | "):
        keys = _PLACEHOLDER.findall(seg)
        if all(values.get(k) not in (None, "") for k in keys):
            segments.append(seg.format(**{k: values[k] for k in keys}))
    header = ("[" if opened else "") + " | ".join(segments) + ("]" if closed else "")
    title = str(values.get("title") or "")
    while count(header) > max_tokens and len(title) > 10:  # shorten the title until the header fits
        title = title[: max(10, len(title) - 15)].rstrip() + "…"
        header = context_header(template, {**values, "title": title}, count, 10**9)
    return header


def chunk_document(doc: dict, pages: list[dict], count: TokenCounter, acronym: str | None) -> list[dict]:
    ch = _s.chunking
    full_text = "\n\n".join(p["text"] for p in pages)
    if doc["doc_type"] in ch.no_split_doc_types and count(full_text) <= ch.max_tokens:
        groups = [[Unit(full_text, pages[0]["page"], True, count(full_text))]]
    else:
        groups = pack(to_units(pages, count, ch.max_tokens), ch.max_tokens, ch.overlap_tokens)

    rows = []
    for i, group in enumerate(groups):
        text = join_units(group)
        page = group[0].page
        embed_text = text
        if ch.context_header.enabled:
            header = context_header(ch.context_header.template,
                                    {"acronym": acronym, "title": doc["title"], "page": page},
                                    count, ch.context_header.max_tokens)
            embed_text = f"{header}\n{text}" if header.strip("[] ") else text
        rows.append({
            "chunk_id": f"{doc['doc_id']}#{i:04d}",
            "doc_id": doc["doc_id"],
            "chunk_index": i,
            "page": page,
            "page_end": group[-1].page,
            "text": text,
            "embed_text": embed_text,
            "n_tokens": count(text),
        })
    return rows


def chunks_path():
    return _s.paths.processed / "chunks.jsonl"


def run() -> None:
    docs = {d["doc_id"]: d for d in read_jsonl(documents_path())}
    pages_by_doc: dict[str, list[dict]] = {}
    for row in read_jsonl(pages_path()):
        pages_by_doc.setdefault(row["doc_id"], []).append(row)

    print(f"  Cargando el tokenizador de {_s.embedding.model_id}...")
    count = hf_token_counter()
    limit = _s.embedding.max_length - 2  # [CLS] and [SEP]
    rows: list[dict] = []
    too_long = 0
    for doc_id, pages in pages_by_doc.items():
        doc = docs[doc_id]
        inv = _s.sources.psi.get(doc["accession"]) if doc["accession"] else None
        for row in chunk_document(doc, pages, count, inv.acronym if inv else None):
            too_long += count(row["embed_text"]) > limit
            rows.append(row)

    n = write_jsonl(chunks_path(), rows)
    sizes = sorted(r["n_tokens"] for r in rows)
    print(f"  {n} fragmentos de {len(pages_by_doc)} documentos -> {rel(chunks_path())}")
    if sizes:
        print(f"  tokens por fragmento: mín {sizes[0]}, mediana {sizes[len(sizes) // 2]}, máx {sizes[-1]}")
    if too_long:
        raise RuntimeError(f"{too_long} fragmentos superan {limit} tokens con el encabezado; "
                           f"baja chunking.max_tokens o chunking.context_header.max_tokens")
