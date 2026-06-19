import argparse

from services.bm25_index_service import build_bm25_index


def main():
    parser = argparse.ArgumentParser(description="Build BM25 index.")
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
        help="Maximum vocabulary size.",
    )

    args = parser.parse_args()

    build_bm25_index(
        limit=args.limit,
        max_features=args.max_features,
    )


if __name__ == "__main__":
    main()