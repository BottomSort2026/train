"""Stage 2 — select: decide automatically which documents enter the corpus.

Sources:
    - one text card per configured investigation (from its PSI metadata)
    - PSI files that match an `include` rule (first matching rule wins)
    - external resources declared in corpus.yml
Only documents whose ingest_stage is in selection.build_stages are kept.

Output: <interim>/documents.jsonl (one row per document) + a summary printed to the console.
"""
import hashlib
import html
import re
from collections import Counter
from pathlib import PurePosixPath

from configs.corpus.config_corpus import Rule, corpus_settings
from flame.common.io import rel, write_jsonl
from flame.corpus.inventory import load_files, load_metadata

_s = corpus_settings
_SUBCATEGORY = re.compile(r"^Legacy/(\d{4}) - (.+)$", re.S)


def documents_path():
    return _s.paths.interim / "documents.jsonl"


def files_dir():
    return _s.paths.raw / "files"


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)[:120]


def parse_subcategory(sub: str | None) -> dict:
    """'Legacy/2015 - TJ Shah, FJ Miller. "Title..." Journal' -> {year, authors, title}."""
    if not sub:
        return {}
    m = _SUBCATEGORY.match(html.unescape(sub).strip())
    if not m:
        return {}
    year, rest = int(m.group(1)), m.group(2).strip()
    quoted = re.search(r'"([^"]+)"', rest)
    if quoted:
        authors = rest[: quoted.start()].strip().rstrip(".").strip() or None
        return {"year": year, "authors": authors, "title": quoted.group(1).strip()}
    return {"year": year, "authors": None, "title": rest}


def _folder(remote_path: str) -> str | None:
    parts = PurePosixPath(remote_path).parts  # ('/', 'Home', 'psi-data', 'PSI-22', 'Raw Data', 'x.pdf')
    return parts[4] if len(parts) > 5 else None


def match_rule(f: dict, accession: str, fire_relevance: int) -> Rule | None:
    name = f.get("file_name", "")
    ext = PurePosixPath(name).suffix.lower()
    folder = _folder(f.get("remote_path", ""))
    sub = html.unescape(f.get("subcategory") or "")
    for r in _s.selection.rules:
        if r.ext and ext != r.ext:
            continue
        if r.folder and (folder or "").lower() != r.folder.lower():
            continue
        if r.file_regex and not r.file_regex.search(name):
            continue
        if r.subcategory_regex and not r.subcategory_regex.search(sub):
            continue
        if r.accessions and accession not in r.accessions:
            continue
        if r.max_fire_relevance and fire_relevance > r.max_fire_relevance:
            continue
        return r
    return None


def _card_doc(inv) -> dict:
    meta = load_metadata(inv.accession)
    cards = _s.selection.investigation_cards
    return {
        "doc_id": f"{inv.accession}:card",
        "source": "psi_card",
        "accession": inv.accession,
        "related_accessions": [],
        "doc_type": cards.doc_type,
        "ingest_stage": cards.ingest_stage,
        "title": f"{meta.get('title') or inv.accession} — investigation summary",
        "authors": None,
        "year": None,
        "file_name": None,
        "psi_folder": None,
        "remote_path": None,
        "format": "card",
        "public_url": None,
        "local_path": None,
        "rule": "investigation-card",
    }


def _psi_doc(f: dict, inv, rule: Rule) -> dict:
    remote_path = f["remote_path"]
    digest = hashlib.sha1(remote_path.encode()).hexdigest()[:10]
    parsed = parse_subcategory(f.get("subcategory"))
    file_name = f["file_name"]
    return {
        "doc_id": f"{inv.accession}:{digest}",
        "source": "psi_file",
        "accession": inv.accession,
        "related_accessions": [],
        "doc_type": rule.doc_type,
        "ingest_stage": rule.ingest_stage,
        "title": parsed.get("title") or PurePosixPath(file_name).stem,
        "authors": parsed.get("authors"),
        "year": parsed.get("year"),
        "file_name": file_name,
        "psi_folder": _folder(remote_path),
        "remote_path": remote_path,
        "format": PurePosixPath(file_name).suffix.lower().lstrip("."),
        "public_url": None,
        "local_path": rel(files_dir() / inv.accession / f"{digest}_{_safe(file_name)}"),
        "rule": rule.name,
    }


def _external_doc(r) -> dict:
    return {
        "doc_id": f"ext:{r.id}",
        "source": "external",
        "accession": None,
        "related_accessions": list(r.related_accessions),
        "doc_type": r.doc_type,
        "ingest_stage": r.ingest_stage,
        "title": r.title,
        "authors": r.authors,
        "year": r.year,
        "file_name": None,
        "psi_folder": None,
        "remote_path": None,
        "format": r.format,
        "public_url": r.url,
        "local_path": rel(files_dir() / "external" / f"{_safe(r.id)}.{r.format}"),
        "rule": "external",
    }


def run() -> None:
    sel = _s.selection
    stages = set(sel.build_stages)
    excluded = set(sel.exclude)
    docs: list[dict] = []
    by_rule: Counter = Counter()
    skipped: Counter = Counter()

    for inv in _s.sources.psi.investigations:
        if sel.investigation_cards.ingest_stage in stages:
            docs.append(_card_doc(inv))
            by_rule["investigation-card"] += 1

        seen_names: dict[str, str] = {}
        for f in load_files(inv.accession):
            name, path = f.get("file_name", ""), f.get("remote_path", "")
            if not name or not path or path in excluded:
                skipped["sin nombre / excluido"] += 1
                continue
            if PurePosixPath(name).suffix.lower() not in sel.allowed_extensions:
                skipped["extensión no permitida"] += 1
                continue
            if (f.get("file_size") or 0) > sel.max_file_mb * 1e6:
                skipped["supera max_file_mb"] += 1
                continue
            rule = match_rule(f, inv.accession, inv.fire_relevance)
            if rule is None or rule.action == "exclude":
                skipped[f"regla {rule.name}" if rule else "sin regla"] += 1
                continue
            if rule.ingest_stage not in stages:
                skipped[f"etapa {rule.ingest_stage} no construida"] += 1
                continue
            # The PSI download endpoint identifies files by name, so duplicate names would be ambiguous.
            if name in seen_names:
                print(f"  AVISO {inv.accession}: nombre repetido '{name}', se omite {path}")
                skipped["nombre repetido"] += 1
                continue
            seen_names[name] = path
            docs.append(_psi_doc(f, inv, rule))
            by_rule[rule.name] += 1

    for r in _s.sources.external:
        if r.ingest_stage in stages and r.id not in excluded:
            docs.append(_external_doc(r))
            by_rule["external"] += 1

    n = write_jsonl(documents_path(), docs)
    print(f"  {n} documentos seleccionados (etapas {sorted(stages)}) -> {rel(documents_path())}")
    print("  Por regla:     " + ", ".join(f"{k}={v}" for k, v in by_rule.most_common()))
    print("  Por tipo:      " + ", ".join(f"{k}={v}" for k, v in Counter(d["doc_type"] for d in docs).most_common()))
    per_acc = Counter(d["accession"] or "externo" for d in docs)
    print("  Por accession: " + ", ".join(f"{k}={v}" for k, v in per_acc.most_common()))
    print("  Descartados:   " + ", ".join(f"{k}={v}" for k, v in skipped.most_common()))
