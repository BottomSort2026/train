-- corpus.db schema (SQLite + FTS5 + sqlite-vec). CONTRACT between the corpus build (Python) and the MCP (TypeScript).
-- {fts_tokenizer} and {dimension} are filled in from corpus.yml at build time.
--
-- Reading guide for the MCP:
--   BM25 search:   SELECT rowid, bm25(chunks_fts) AS score FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?
--                  (lower bm25 = better; quote each term, e.g. '"SAFFIRE-III" OR "PMMA"')
--   Vector search: SELECT rowid, distance FROM chunks_vec WHERE embedding MATCH ? AND k = ? ORDER BY distance
--                  (query vector = float32 little-endian bytes from engine /embed; L2 distance on normalized
--                   vectors, so cosine similarity = 1 - distance^2 / 2)
--   rowid of chunks_fts and chunks_vec = chunks.id
--   Citation:      chunks JOIN documents USING (doc_id) LEFT JOIN investigations USING (accession)

-- Key/value metadata. value is JSON. Keys: schema_version, corpus_name, description, built_at, build_stages,
-- embedding (contract + resolved_revision), search (defaults), counts, sqlite_vec_version, vector_distance.
CREATE TABLE corpus_info (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- One row per configured PSI investigation. metadata_json = PSI metadata without contact details.
CREATE TABLE investigations (
    accession         TEXT PRIMARY KEY,
    acronym           TEXT,
    title             TEXT NOT NULL,
    fire_relevance    INTEGER NOT NULL CHECK (fire_relevance BETWEEN 1 AND 3),
    investigation_url TEXT NOT NULL,
    metadata_json     TEXT NOT NULL
);

-- One row per source document (PDF, TXT, HTML page or investigation card).
CREATE TABLE documents (
    doc_id             TEXT PRIMARY KEY,
    source             TEXT NOT NULL CHECK (source IN ('psi_card', 'psi_file', 'external')),
    accession          TEXT REFERENCES investigations(accession),  -- NULL for external resources
    related_accessions TEXT NOT NULL DEFAULT '[]',                 -- JSON list, for external resources
    doc_type           TEXT NOT NULL,
    ingest_stage       INTEGER NOT NULL,
    title              TEXT NOT NULL,
    authors            TEXT,
    year               INTEGER,
    file_name          TEXT,
    psi_folder         TEXT,
    public_url         TEXT                                        -- set for external resources only
);
CREATE INDEX idx_documents_accession ON documents(accession);

-- One row per fragment. text is what the LLM receives (without the context header).
CREATE TABLE chunks (
    id          INTEGER PRIMARY KEY,           -- rowid shared with chunks_fts and chunks_vec
    chunk_id    TEXT NOT NULL UNIQUE,          -- '<doc_id>#<index>'
    doc_id      TEXT NOT NULL REFERENCES documents(doc_id),
    chunk_index INTEGER NOT NULL,              -- order inside the document
    page        INTEGER,                       -- first page (NULL for non-paginated sources)
    page_end    INTEGER,                       -- last page
    text        TEXT NOT NULL,
    n_tokens    INTEGER NOT NULL
);
CREATE INDEX idx_chunks_doc ON chunks(doc_id, chunk_index);

-- Keyword index (external content: the text lives in chunks).
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    text,
    content='chunks',
    content_rowid='id',
    tokenize='{fts_tokenizer}'
);

-- Vector index (sqlite-vec).
CREATE VIRTUAL TABLE chunks_vec USING vec0(
    embedding float[{dimension}]
);
