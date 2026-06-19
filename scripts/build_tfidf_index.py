from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

from services.tfidf_index_service import build_tfidf_index


def main():
    parser = argparse.ArgumentParser(description="Build TF-IDF index.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for testing. Example: --limit 5000",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=100000,
        help="Maximum TF-IDF vocabulary size.",
    )

    args = parser.parse_args()

    build_tfidf_index(
        limit=args.limit,
        max_features=args.max_features,
    )


if __name__ == "__main__":
    main()