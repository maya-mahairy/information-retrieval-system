from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

from services.evaluation_service import evaluate_models


def main():
    parser = argparse.ArgumentParser(description="Evaluate IR models using qrels.")
    parser.add_argument(
        "--models",
    nargs="+",
    default=["tfidf", "bm25", "hybrid_serial", "hybrid_parallel"],
    choices=[
    "tfidf",
    "bm25",
    "hybrid_serial",
    "hybrid_parallel",
    "bm25_refined_corrected",
    "bm25_refined_expanded",
],
    help="Models to evaluate.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Top K documents to retrieve for evaluation.",
    )

    args = parser.parse_args()

    evaluate_models(
        model_names=args.models,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()