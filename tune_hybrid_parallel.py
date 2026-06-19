import argparse

from services.hybrid_parallel_tuning_service import run_hybrid_parallel_tuning


def main():
    parser = argparse.ArgumentParser(description="Tune Hybrid Parallel weighted RRF.")
    parser.add_argument(
        "--bm25-weights",
        nargs="+",
        type=float,
        default=[0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
        help="BM25 weights to test. TF-IDF weight will be 1 - BM25 weight.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Top K documents to retrieve for evaluation.",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="RRF constant k.",
    )
    parser.add_argument(
        "--k1",
        type=float,
        default=1.5,
        help="BM25 k1 parameter.",
    )
    parser.add_argument(
        "--b",
        type=float,
        default=0.75,
        help="BM25 b parameter.",
    )

    args = parser.parse_args()

    run_hybrid_parallel_tuning(
        bm25_weights=args.bm25_weights,
        top_k=args.top_k,
        rrf_k=args.rrf_k,
        k1=args.k1,
        b=args.b,
    )


if __name__ == "__main__":
    main()