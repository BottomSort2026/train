"""Loads configs/corpus/corpus.yml into frozen dataclasses, once, at import time.

Usage:
    from configs.corpus.config_corpus import corpus_settings
    corpus_settings.chunking.max_tokens
"""
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from configs import PROJECT_ROOT

corpus_settings_path = PROJECT_ROOT / Path("configs/corpus/corpus.yml")

VALID_ACTIONS = ("include", "exclude")
VALID_STAGES = (0, 1, 2)
VALID_RELEVANCE = (1, 2, 3)
VALID_FORMATS = ("pdf", "txt", "html")


def _check_keys(section: dict, allowed: set[str], where: str, required: set[str] | None = None) -> None:
    """Fail on unknown keys (typos) and on missing required keys."""
    if not isinstance(section, dict):
        raise ValueError(f"corpus.yml: '{where}' debe ser un diccionario")
    unknown = set(section) - allowed
    if unknown:
        raise ValueError(f"corpus.yml: claves desconocidas en '{where}': {sorted(unknown)}")
    missing = (required if required is not None else allowed) - set(section)
    if missing:
        raise ValueError(f"corpus.yml: faltan claves en '{where}': {sorted(missing)}")


def _compile(pattern: str | None, where: str) -> re.Pattern | None:
    if pattern is None:
        return None
    try:
        return re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        raise ValueError(f"corpus.yml: expresión regular inválida en '{where}': {pattern!r} ({e})") from e


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CorpusInfo:
    name: str
    description: str


@dataclass(frozen=True)
class Paths:
    raw: Path
    interim: Path
    processed: Path
    output_db: Path
    manifest: Path


@dataclass(frozen=True)
class Investigation:
    accession: str
    acronym: str | None
    fire_relevance: int

    def __post_init__(self):
        if not re.fullmatch(r"PSI-\d+", self.accession):
            raise ValueError(f"Accession inválido: {self.accession!r}")
        if self.fire_relevance not in VALID_RELEVANCE:
            raise ValueError(f"{self.accession}: fire_relevance debe ser 1, 2 o 3")


@dataclass(frozen=True)
class PsiSource:
    api_base: str
    investigation_url: str
    investigations: tuple[Investigation, ...]

    def __post_init__(self):
        accs = [i.accession for i in self.investigations]
        dup = {a for a in accs if accs.count(a) > 1}
        if dup:
            raise ValueError(f"Accessions repetidos en sources.psi.investigations: {sorted(dup)}")
        if "{accession}" not in self.investigation_url:
            raise ValueError("sources.psi.investigation_url debe contener '{accession}'")

    def get(self, accession: str) -> Investigation | None:
        return next((i for i in self.investigations if i.accession == accession), None)


@dataclass(frozen=True)
class ExternalResource:
    id: str
    ingest_stage: int
    doc_type: str
    format: str
    title: str
    url: str
    related_accessions: tuple[str, ...]
    authors: str | None = None
    year: int | None = None

    def __post_init__(self):
        if self.ingest_stage not in VALID_STAGES:
            raise ValueError(f"Recurso {self.id}: ingest_stage debe ser 0, 1 o 2")
        if self.format not in VALID_FORMATS:
            raise ValueError(f"Recurso {self.id}: format debe ser uno de {VALID_FORMATS}")
        if not self.url.startswith("https://"):
            raise ValueError(f"Recurso {self.id}: la url debe empezar con https://")


@dataclass(frozen=True)
class Sources:
    psi: PsiSource
    external: tuple[ExternalResource, ...]

    def __post_init__(self):
        ids = [r.id for r in self.external]
        dup = {i for i in ids if ids.count(i) > 1}
        if dup:
            raise ValueError(f"Ids repetidos en sources.external: {sorted(dup)}")
        known = {i.accession for i in self.psi.investigations}
        for r in self.external:
            bad = set(r.related_accessions) - known
            if bad:
                raise ValueError(f"Recurso {r.id}: related_accessions no están en sources.psi: {sorted(bad)}")


@dataclass(frozen=True)
class Rule:
    name: str
    action: str
    doc_type: str | None
    ingest_stage: int | None
    ext: str | None
    folder: str | None
    file_regex: re.Pattern | None
    subcategory_regex: re.Pattern | None
    accessions: tuple[str, ...] | None
    max_fire_relevance: int | None

    def __post_init__(self):
        if self.action not in VALID_ACTIONS:
            raise ValueError(f"Regla {self.name}: action debe ser 'include' o 'exclude'")
        if self.action == "include":
            if self.doc_type is None or self.ingest_stage is None:
                raise ValueError(f"Regla {self.name}: una regla 'include' necesita doc_type e ingest_stage")
            if self.ingest_stage not in VALID_STAGES:
                raise ValueError(f"Regla {self.name}: ingest_stage debe ser 0, 1 o 2")
        if self.max_fire_relevance is not None and self.max_fire_relevance not in VALID_RELEVANCE:
            raise ValueError(f"Regla {self.name}: max_fire_relevance debe ser 1, 2 o 3")


@dataclass(frozen=True)
class InvestigationCards:
    ingest_stage: int
    doc_type: str


@dataclass(frozen=True)
class Selection:
    build_stages: tuple[int, ...]
    investigation_cards: InvestigationCards
    allowed_extensions: tuple[str, ...]
    max_file_mb: float
    rules: tuple[Rule, ...]
    exclude: tuple[str, ...]

    def __post_init__(self):
        if not self.build_stages or set(self.build_stages) - set(VALID_STAGES):
            raise ValueError("selection.build_stages debe contener valores entre 0 y 2")
        names = [r.name for r in self.rules]
        dup = {n for n in names if names.count(n) > 1}
        if dup:
            raise ValueError(f"Nombres de reglas repetidos: {sorted(dup)}")


@dataclass(frozen=True)
class PdfExtraction:
    min_chars_per_page: int
    strip_repeated_lines: bool
    repeated_line_min_ratio: float
    drop_references: bool

    def __post_init__(self):
        if not 0 < self.repeated_line_min_ratio <= 1:
            raise ValueError("extraction.pdf.repeated_line_min_ratio debe estar entre 0 y 1")


@dataclass(frozen=True)
class TxtExtraction:
    encodings: tuple[str, ...]


@dataclass(frozen=True)
class HtmlExtraction:
    content_tags: tuple[str, ...]
    min_chars: int

    def __post_init__(self):
        if not self.content_tags:
            raise ValueError("extraction.html.content_tags no puede estar vacío")
        bad = [t for t in self.content_tags if not re.fullmatch(r"[a-z][a-z0-9]*", t)]
        if bad:
            raise ValueError(f"extraction.html.content_tags debe contener nombres de etiqueta HTML: {bad}")


@dataclass(frozen=True)
class Extraction:
    pdf: PdfExtraction
    txt: TxtExtraction
    html: HtmlExtraction


@dataclass(frozen=True)
class ContextHeader:
    enabled: bool
    template: str
    max_tokens: int


@dataclass(frozen=True)
class Chunking:
    max_tokens: int
    overlap_tokens: int
    no_split_doc_types: tuple[str, ...]
    context_header: ContextHeader

    def __post_init__(self):
        if self.max_tokens <= 0:
            raise ValueError("chunking.max_tokens debe ser mayor que 0")
        if not 0 <= self.overlap_tokens < self.max_tokens:
            raise ValueError("chunking.overlap_tokens debe ser >= 0 y menor que max_tokens")


@dataclass(frozen=True)
class Embedding:
    model_id: str
    revision: str | None
    dimension: int
    normalize: bool
    max_length: int
    query_prefix: str
    document_prefix: str
    batch_size: int
    device: str


@dataclass(frozen=True)
class Search:
    k: int
    candidates_per_method: int
    fusion: str
    rrf_k: int
    fts_tokenizer: str

    def __post_init__(self):
        if self.fusion != "rrf":
            raise ValueError("search.fusion: por ahora solo se soporta 'rrf'")
        if self.k > self.candidates_per_method:
            raise ValueError("search.k no puede ser mayor que search.candidates_per_method")


@dataclass(frozen=True)
class Network:
    timeout_api_s: float
    timeout_download_s: float
    retries: int
    backoff_s: float
    concurrency: int
    delay_between_requests_s: float
    user_agent: str


@dataclass(frozen=True)
class CorpusSettings:
    corpus: CorpusInfo
    paths: Paths
    sources: Sources
    selection: Selection
    extraction: Extraction
    chunking: Chunking
    embedding: Embedding
    search: Search
    network: Network

    def __post_init__(self):
        # Fragment + context header + 2 special tokens ([CLS], [SEP]) must fit in the model's max length.
        header = self.chunking.context_header.max_tokens if self.chunking.context_header.enabled else 0
        needed = self.chunking.max_tokens + header + 2
        if needed > self.embedding.max_length:
            raise ValueError(
                f"chunking.max_tokens ({self.chunking.max_tokens}) + encabezado ({header}) + 2 = {needed} "
                f"supera embedding.max_length ({self.embedding.max_length})"
            )
        known = {i.accession for i in self.sources.psi.investigations}
        for r in self.selection.rules:
            if r.accessions and set(r.accessions) - known:
                raise ValueError(f"Regla {r.name}: accessions no están en sources.psi: {sorted(set(r.accessions) - known)}")


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
_RULE_KEYS = {"name", "action", "doc_type", "ingest_stage", "ext", "folder", "file_regex",
              "subcategory_regex", "accessions", "max_fire_relevance"}
_EXTERNAL_KEYS = {"id", "ingest_stage", "doc_type", "format", "title", "url", "related_accessions", "authors", "year"}


def _rule(r: dict) -> Rule:
    _check_keys(r, _RULE_KEYS, f"selection.rules[{r.get('name', '?')}]", required={"name", "action"})
    return Rule(
        name=r["name"],
        action=r["action"],
        doc_type=r.get("doc_type"),
        ingest_stage=r.get("ingest_stage"),
        ext=r["ext"].lower() if r.get("ext") else None,
        folder=r.get("folder"),
        file_regex=_compile(r.get("file_regex"), f"regla {r['name']}.file_regex"),
        subcategory_regex=_compile(r.get("subcategory_regex"), f"regla {r['name']}.subcategory_regex"),
        accessions=tuple(r["accessions"]) if r.get("accessions") else None,
        max_fire_relevance=r.get("max_fire_relevance"),
    )


def _external(r: dict) -> ExternalResource:
    _check_keys(r, _EXTERNAL_KEYS, f"sources.external[{r.get('id', '?')}]",
                required=_EXTERNAL_KEYS - {"authors", "year"})
    return ExternalResource(
        id=r["id"],
        ingest_stage=r["ingest_stage"],
        doc_type=r["doc_type"],
        format=r["format"],
        title=r["title"],
        url=r["url"],
        related_accessions=tuple(r["related_accessions"]),
        authors=r.get("authors"),
        year=r.get("year"),
    )


def init_corpus_settings(path: Path) -> CorpusSettings:
    s = yaml.safe_load(path.read_text(encoding="utf-8"))
    _check_keys(s, {"corpus", "paths", "sources", "selection", "extraction", "chunking", "embedding", "search", "network"}, "raíz")

    _check_keys(s["corpus"], {"name", "description"}, "corpus")
    _check_keys(s["paths"], {"raw", "interim", "processed", "output_db", "manifest"}, "paths")
    _check_keys(s["sources"], {"psi", "external"}, "sources")
    _check_keys(s["sources"]["psi"], {"api_base", "investigation_url", "investigations"}, "sources.psi")
    for inv in s["sources"]["psi"]["investigations"]:
        _check_keys(inv, {"accession", "acronym", "fire_relevance"}, f"sources.psi.investigations[{inv.get('accession', '?')}]")
    _check_keys(s["selection"], {"build_stages", "investigation_cards", "allowed_extensions", "max_file_mb", "rules", "exclude"}, "selection")
    _check_keys(s["selection"]["investigation_cards"], {"ingest_stage", "doc_type"}, "selection.investigation_cards")
    _check_keys(s["extraction"], {"pdf", "txt", "html"}, "extraction")
    _check_keys(s["extraction"]["pdf"], {"min_chars_per_page", "strip_repeated_lines", "repeated_line_min_ratio", "drop_references"}, "extraction.pdf")
    _check_keys(s["extraction"]["txt"], {"encodings"}, "extraction.txt")
    _check_keys(s["extraction"]["html"], {"content_tags", "min_chars"}, "extraction.html")
    _check_keys(s["chunking"], {"max_tokens", "overlap_tokens", "no_split_doc_types", "context_header"}, "chunking")
    _check_keys(s["chunking"]["context_header"], {"enabled", "template", "max_tokens"}, "chunking.context_header")
    _check_keys(s["embedding"], {"model_id", "revision", "dimension", "normalize", "max_length", "query_prefix",
                                 "document_prefix", "batch_size", "device"}, "embedding")
    _check_keys(s["search"], {"k", "candidates_per_method", "fusion", "rrf_k", "fts_tokenizer"}, "search")
    _check_keys(s["network"], {"timeout_api_s", "timeout_download_s", "retries", "backoff_s", "concurrency",
                               "delay_between_requests_s", "user_agent"}, "network")

    p, psi, sel, ex, ch, em, se, nw = (s["paths"], s["sources"]["psi"], s["selection"], s["extraction"],
                                       s["chunking"], s["embedding"], s["search"], s["network"])
    return CorpusSettings(
        corpus=CorpusInfo(name=s["corpus"]["name"], description=s["corpus"]["description"]),
        paths=Paths(
            raw=PROJECT_ROOT / Path(p["raw"]),
            interim=PROJECT_ROOT / Path(p["interim"]),
            processed=PROJECT_ROOT / Path(p["processed"]),
            output_db=PROJECT_ROOT / Path(p["output_db"]),
            manifest=PROJECT_ROOT / Path(p["manifest"]),
        ),
        sources=Sources(
            psi=PsiSource(
                api_base=psi["api_base"].rstrip("/"),
                investigation_url=psi["investigation_url"],
                investigations=tuple(
                    Investigation(accession=i["accession"], acronym=i["acronym"], fire_relevance=i["fire_relevance"])
                    for i in psi["investigations"]
                ),
            ),
            external=tuple(_external(r) for r in s["sources"]["external"]),
        ),
        selection=Selection(
            build_stages=tuple(sel["build_stages"]),
            investigation_cards=InvestigationCards(**sel["investigation_cards"]),
            allowed_extensions=tuple(e.lower() for e in sel["allowed_extensions"]),
            max_file_mb=float(sel["max_file_mb"]),
            rules=tuple(_rule(r) for r in sel["rules"]),
            exclude=tuple(sel["exclude"] or ()),
        ),
        extraction=Extraction(
            pdf=PdfExtraction(**ex["pdf"]),
            txt=TxtExtraction(encodings=tuple(ex["txt"]["encodings"])),
            html=HtmlExtraction(content_tags=tuple(t.lower() for t in ex["html"]["content_tags"]),
                                min_chars=ex["html"]["min_chars"]),
        ),
        chunking=Chunking(
            max_tokens=ch["max_tokens"],
            overlap_tokens=ch["overlap_tokens"],
            no_split_doc_types=tuple(ch["no_split_doc_types"]),
            context_header=ContextHeader(**ch["context_header"]),
        ),
        embedding=Embedding(**em),
        search=Search(**se),
        network=Network(**nw),
    )


corpus_settings = init_corpus_settings(corpus_settings_path)
