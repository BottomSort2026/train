"""Stage 4 — extract: turn every selected document into text, page by page.

    card -> text built from the investigation's PSI metadata (contact details are never included)
    pdf  -> one entry per page, paragraphs from PyMuPDF text blocks, repeated headers/footers removed
    txt  -> one entry (encodings tried in order, line endings normalized)
    html -> one entry with the main content only

Output: <interim>/pages.jsonl with rows {doc_id, page, text}. page is null for card, txt and html.
"""
import html
import re
from collections import Counter
from html.parser import HTMLParser

from configs import PROJECT_ROOT
from configs.corpus.config_corpus import corpus_settings
from flame.common.io import read_jsonl, rel, write_jsonl
from flame.corpus.inventory import load_metadata
from flame.corpus.select import documents_path

_s = corpus_settings
_TAG = re.compile(r"<[^>]+>")


def pages_path():
    return _s.paths.interim / "pages.jsonl"


def clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Investigation card
# ---------------------------------------------------------------------------
_CARD_FIELDS = [
    ("Research area", "researchArea"),
    ("Sub-research area", "subResearchArea"),
    ("Project type", "projectType"),
    ("Flight program / platform", "flightProgram"),
    ("Managing NASA center", "managingNasaCenter"),
    ("Experiment hardware", "experimentHardware"),
    ("Investigation start date", "investigationStartDate"),
    ("Investigation end date", "investigationEndDate"),
    ("Proposal title", "proposalTitle"),
    ("Objectives", "objective"),
    ("Hypothesis", "hypothesis"),
    ("Approach", "approach"),
    ("Research impacts and Earth benefits", "researchImpacts"),
    ("Description", "description"),
]


def _plain(value) -> str:
    return clean(html.unescape(_TAG.sub(" ", str(value)))) if value else ""


def card_text(accession: str) -> str:
    meta = load_metadata(accession)
    inv = _s.sources.psi.get(accession)
    acronym = meta.get("investigationAcronym") or (inv.acronym if inv else None)
    head = f"{meta.get('title', accession)}" + (f" ({acronym})" if acronym else "")
    parts = [f"PSI investigation {accession}: {head}"]
    for label, key in _CARD_FIELDS:
        value = _plain(meta.get(key))
        if value:
            parts.append(f"{label}: {value}")
    # Investigators and publications are deliberately NOT part of the searchable card: as fragments, those lists
    # get a generic "combustion" embedding and show up in almost every search. They stay available in
    # investigations.metadata_json (get_investigation).
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------
def _pdf_pages(path) -> list[str]:
    import fitz  # PyMuPDF

    pages: list[str] = []
    with fitz.open(path) as pdf:
        for page in pdf:
            blocks = [b[4] for b in page.get_text("blocks") if b[6] == 0]  # b[6] == 0: text block
            paragraphs = []
            for b in blocks:
                b = re.sub(r"(\w)-\n(\w)", r"\1\2", b)  # join words hyphenated across lines
                b = re.sub(r"\s*\n\s*", " ", b).strip()  # one block = one paragraph
                if b:
                    paragraphs.append(b)
            pages.append("\n\n".join(paragraphs))
    return pages


def _strip_repeated_lines(pages: list[str]) -> list[str]:
    """Remove paragraphs (headers, footers, page numbers) that repeat on many pages."""
    cfg = _s.extraction.pdf
    if not cfg.strip_repeated_lines or len(pages) < 3:
        return pages

    def norm(p: str) -> str:
        return re.sub(r"\d+", "#", p.strip().lower())

    counts = Counter(n for page in pages for n in {norm(p) for p in page.split("\n\n") if p.strip()})
    repeated = {n for n, c in counts.items() if c / len(pages) >= cfg.repeated_line_min_ratio and len(n) < 200}
    return ["\n\n".join(p for p in page.split("\n\n") if norm(p) not in repeated) for page in pages]


_REFERENCES_HEADING = re.compile(
    r"^(references|reference list|bibliography|works cited|literature cited|cited literature|"
    r"selected publications|publications)\s*:?$", re.I)
_CITATION_MARKERS = re.compile(
    r"\bdoi\b|doi\.org|\bvol\.|\bpp\.|\bproceedings\b|\bjournal\b|\bet al\.|\(\d{4}\)|\b(?:19|20)\d{2}[a-z]?[.;,)]|"
    r"\bAIAA\b|NASA/(?:TM|CR|TP)|\bISBN\b|\bISSN\b|combust\.|\bsymp(?:osium)?\b", re.I)


def is_citation_paragraph(paragraph: str) -> bool:
    """A paragraph that is mostly bibliographic references (several citation markers, densely packed)."""
    hits = len(_CITATION_MARKERS.findall(paragraph))
    words = max(len(paragraph.split()), 1)
    return hits >= 3 and hits / words >= 1 / 25


def strip_references(pages: list[str]) -> tuple[list[str], int]:
    """Remove bibliography: everything after a 'References' heading on its page, plus citation-dense paragraphs
    on any page. Returns (pages, number of paragraphs removed)."""
    if not _s.extraction.pdf.drop_references:
        return pages, 0
    removed = 0
    out = []
    for page in pages:
        kept, in_refs = [], False
        for p in page.split("\n\n"):
            if _REFERENCES_HEADING.match(p.strip()):
                in_refs = True
            if in_refs or is_citation_paragraph(p):
                removed += 1
                continue
            kept.append(p)
        out.append("\n\n".join(kept))
    return out, removed


# ---------------------------------------------------------------------------
# TXT and HTML
# ---------------------------------------------------------------------------
def _txt(path) -> str:
    raw = path.read_bytes()
    for enc in _s.extraction.txt.encodings:
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"no se pudo decodificar con {_s.extraction.txt.encodings}")


class _MainText(HTMLParser):
    """Collect the readable text inside the first <tag>.

    Removed:
      - scripts, styles, navigation, headers, footers, sidebars and form controls
      - hidden elements: `hidden`, aria-hidden="true", display:none, and modal dialogs (data-reveal, role=dialog).
        sma.nasa.gov pages carry a hidden modal with text from another page.
      - link lists: blocks whose text is mostly links (menus, "related articles")
      - very short blocks (fewer than MIN_WORDS words), except h1–h3 headings: menu items, buttons, labels
    <form> itself is NOT skipped, because some sites wrap the whole page in one <form>.
    """

    SKIP = {"script", "style", "noscript", "template", "nav", "header", "footer", "aside", "svg",
            "button", "select", "option", "textarea", "label", "iframe"}
    BLOCK = {"p", "div", "section", "article", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "td", "br",
             "ul", "ol", "table", "figcaption", "blockquote", "dd", "dt"}
    HEADINGS = {"h1", "h2", "h3"}
    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}
    MIN_WORDS = 4
    MAX_LINK_RATIO = 0.6

    def __init__(self, tag: str):
        super().__init__(convert_charrefs=True)
        self.tag = tag
        self.stack: list[tuple[str, bool]] = []  # (tag, hidden-or-skipped)
        self.inside = 0      # depth inside the target tag
        self.done = False
        self.blocks: list[dict] = []
        self._new_block(None)

    def _new_block(self, tag):
        self.blocks.append({"tag": tag, "text": [], "link_chars": 0})

    @staticmethod
    def _hidden(attrs: dict) -> bool:
        style = (attrs.get("style") or "").replace(" ", "").lower()
        return ("hidden" in attrs or attrs.get("aria-hidden") == "true" or "display:none" in style
                or "data-reveal" in attrs or attrs.get("role") in ("dialog", "navigation"))

    def _skipping(self) -> bool:
        return any(skip for _, skip in self.stack)

    def _in_link(self) -> bool:
        return any(t == "a" for t, _ in self.stack)

    def handle_starttag(self, tag, attrs):
        if self.done:
            return
        attrs = dict(attrs)
        if tag == self.tag:
            self.inside += 1
        if tag in self.BLOCK and not self._skipping():
            self._new_block(tag)
        if tag not in self.VOID:
            self.stack.append((tag, tag in self.SKIP or self._hidden(attrs)))

    def handle_endtag(self, tag):
        if self.done:
            return
        # pop up to the matching open tag (tolerates unclosed tags in messy HTML)
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                break
        if tag in self.BLOCK:
            self._new_block(None)
        if tag == self.tag and self.inside:
            self.inside -= 1
            self.done = self.inside == 0

    def handle_data(self, data):
        if not self.inside or self.done or self._skipping():
            return
        block = self.blocks[-1]
        block["text"].append(data)
        if self._in_link():
            block["link_chars"] += len(data.strip())

    def text(self) -> str:
        kept: list[tuple[str, bool]] = []  # (text, is_heading)
        seen: set[str] = set()
        for b in self.blocks:
            text = re.sub(r"\s+", " ", "".join(b["text"])).strip()
            if not text:
                continue
            if b["link_chars"] / max(len(text), 1) > self.MAX_LINK_RATIO:
                continue
            is_heading = b["tag"] in self.HEADINGS
            if len(text.split()) < self.MIN_WORDS and not is_heading:
                continue
            key = text.lower()
            if key in seen:  # same paragraph repeated by the page layout
                continue
            seen.add(key)
            kept.append((text, is_heading))
        # drop headings with no body text after them (e.g. a trailing "Related Stories")
        result = [t for i, (t, h) in enumerate(kept)
                  if not h or any(not h2 for _, h2 in kept[i + 1:i + 2])]
        return "\n\n".join(result)


def html_text(source: str) -> tuple[str, str]:
    """Return (text, tag used): the first tag in content_tags with at least min_chars of text."""
    cfg = _s.extraction.html
    lengths = []
    for tag in cfg.content_tags:
        parser = _MainText(tag)
        parser.feed(source)
        text = clean(parser.text())
        if len(text) >= cfg.min_chars:
            return text, tag
        lengths.append(f"<{tag}>={len(text)}")
    raise ValueError(f"no se encontró contenido principal en el HTML ({', '.join(lengths)} caracteres)")


def _html(path) -> str:
    return html_text(path.read_text(encoding="utf-8", errors="ignore"))[0]


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------
def run() -> None:
    docs = list(read_jsonl(documents_path()))
    rows: list[dict] = []
    low_text: list[str] = []
    references_removed = 0
    failures: list[str] = []
    min_chars = _s.extraction.pdf.min_chars_per_page

    for doc in docs:
        try:
            if doc["source"] == "psi_card":
                rows.append({"doc_id": doc["doc_id"], "page": None, "text": card_text(doc["accession"])})
                continue
            path = PROJECT_ROOT / doc["local_path"]
            if not path.exists():
                raise FileNotFoundError(f"no está descargado ({doc['local_path']}). Corre 'fetch'.")
            if doc["format"] == "pdf":
                pages = _strip_repeated_lines(_pdf_pages(path))
                # pages without text BEFORE removing references = possible scans (reported, no OCR)
                empty = sum(len(clean(p)) < min_chars for p in pages)
                pages, n_refs = strip_references(pages)
                references_removed += n_refs
                for i, text in enumerate(pages, 1):
                    text = clean(text)
                    if len(text) >= min_chars:
                        rows.append({"doc_id": doc["doc_id"], "page": i, "text": text})
                if empty:
                    low_text.append(f"{doc['doc_id']} ({doc['title'][:50]}): {empty}/{len(pages)} páginas sin texto")
            elif doc["format"] == "txt":
                rows.append({"doc_id": doc["doc_id"], "page": None, "text": clean(_txt(path))})
            elif doc["format"] == "html":
                rows.append({"doc_id": doc["doc_id"], "page": None, "text": _html(path)})
            else:
                raise ValueError(f"formato no soportado: {doc['format']}")
        except Exception as e:
            failures.append(f"{doc['doc_id']}: {e}")

    rows = [r for r in rows if r["text"]]
    n = write_jsonl(pages_path(), rows)
    with_text = len({r["doc_id"] for r in rows})
    print(f"  {n} páginas de texto de {with_text}/{len(docs)} documentos -> {rel(pages_path())}")
    print(f"  {sum(len(r['text']) for r in rows) / 1e6:.2f} M caracteres en total")
    if references_removed:
        print(f"  {references_removed} párrafos de referencias bibliográficas descartados")
    if low_text:
        print(f"  {len(low_text)} PDFs con páginas sin texto (posibles escaneos, no se aplica OCR):")
        for line in low_text[:15]:
            print(f"    {line}")
    if failures:
        for f in failures[:10]:
            print(f"  ERROR {f}")
        raise RuntimeError(f"{len(failures)} documentos no se pudieron extraer")
