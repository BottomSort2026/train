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
