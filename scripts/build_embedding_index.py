from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

from services.embedding_index_service import build_embedding_index


def main():
    parser = argparse.ArgumentParser(description="Build embedding vector store.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for testing. Example: --limit 1000",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size.",
    )

    args = parser.parse_args()

    build_embedding_index(
        limit=args.limit,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()