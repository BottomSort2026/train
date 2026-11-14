# train

Offline, local pipelines for the *Flame in Freefall* project (NASA Space Apps 2026):

- **Model training:** produces the model bundle loaded by the ML service (`/predict`).
- **Corpus build:** produces `corpus.db` (SQLite, read-only), loaded by the MCP service.

## Setup

```bash
conda env create -f environment.yml
conda activate flame-train
```

## Usage

```bash
python main.py --help
python main.py train --help
python main.py corpus --help
```

## Corpus (`corpus.db`)

Configuration: `configs/corpus/corpus.yml`, loaded once into dataclasses by `configs/corpus/config_corpus.py`.

```bash
python main.py corpus build            # all stages, stops at the first one that fails
# or stage by stage:
python main.py corpus inventory        # PSI metadata + file lists (cached)
python main.py corpus select           # automatic selection with the rules -> documents.jsonl
python main.py corpus fetch            # downloads (cached; re-run downloads only what is missing)
python main.py corpus extract          # text per page
python main.py corpus chunk            # fragments (≤ chunking.max_tokens)
python main.py corpus embed            # embeddings (BAAI/bge-small-en-v1.5)
python main.py corpus load             # corpus.db + manifest.json

python main.py corpus search "how does CO2 dilution suppress droplet flames"   # query in English
```

Add `-v` to any command to see the full traceback when it fails.

Start over from zero (deletes the PSI cache, downloads, intermediate files and `corpus.db`; training data is untouched):

```bash
python main.py corpus clean            # asks for confirmation (-y to skip)
python main.py corpus build --fresh    # clean + build
```

### Updating the conda environment

After `environment.yml` changes, the safest way is to recreate the environment:

```bash
conda deactivate
conda env remove -n flame-train                         # or: conda env remove -p ~/.envs/flame-train
conda env create -f environment.yml                     # or: conda env create -f environment.yml -p ~/.envs/flame-train
conda activate flame-train                              # or: conda activate ~/.envs/flame-train
```

For small changes (adding a package), `conda env update -f environment.yml --prune` is enough.
It does not uninstall packages that were installed with pip, so recreate the environment when a package
moves from pip to conda.

- `src/flame/corpus/schema.sql` is the `corpus.db` contract for the MCP (TypeScript).
- `src/flame/corpus/embedding.py` is copied **verbatim** into the engine service (`/embed`).

## Layout

```
configs/training/   YAML configs, one per training experiment
configs/corpus/     corpus build config (sources, tiers, chunking, embedding model)
src/flame/common/   shared utilities
src/flame/training/ training pipeline
src/flame/corpus/   corpus pipeline: fetch → extract → chunk → embed → corpus.db
eval/               RAG evaluation question set
data/               raw/ → interim/ → processed/ (not committed)
artifacts/models/   exported model bundles (not committed)
artifacts/corpus/   built corpus.db + manifest (not committed)
tests/
```
