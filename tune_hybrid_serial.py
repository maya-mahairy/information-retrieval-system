import argparse

from services.hybrid_serial_tuning_service import run_hybrid_serial_tuning


def main():
    parser = argparse.ArgumentParser(description="Tune Hybrid Serial alpha values.")
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[0.5, 0.7, 0.8, 0.9],
        help="Alpha values to test.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Top K documents to retrieve for evaluation.",
    )
    parser.add_argument(
        "--bm25-candidates",
        type=int,
        default=100,
        help="Number of BM25 candidates to rerank with embeddings.",
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

    run_hybrid_serial_tuning(
        alphas=args.alphas,
        top_k=args.top_k,
        bm25_candidates=args.bm25_candidates,
        k1=args.k1,
        b=args.b,
    )


if __name__ == "__main__":
    main()