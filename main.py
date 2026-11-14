"""CLI entry point for the offline pipelines: model training and corpus build.

Usage:
    python main.py <group> <command> [options]

Commands are added as they are needed. Each command maps to a function in src/flame/<group>/.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from flame.training.cli import register_commands as register_train_commands  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="main.py", description="Flame in Freefall offline pipelines")
    groups = parser.add_subparsers(dest="group", required=True)

    # --- train: model training (commands added as needed) ---
    train = groups.add_parser("train", help="Model training pipeline")
    register_train_commands(train.add_subparsers(dest="command", required=True))

    # --- corpus: corpus.db build (commands added as needed) ---
    corpus = groups.add_parser("corpus", help="Corpus (corpus.db) build pipeline")
    corpus.add_subparsers(dest="command", required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args) if hasattr(args, "func") else 0


if __name__ == "__main__":
    sys.exit(main())
