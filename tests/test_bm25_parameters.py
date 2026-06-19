from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.bm25_search_service import BM25SearchService


def print_results(title, results):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    if not results:
        print("No results found.")
        return

    print("Cleaned query:", results[0]["cleaned_query"])
    print("BM25 parameters: k1 =", results[0]["k1"], ", b =", results[0]["b"])

    for result in results:
        print("\n" + "-" * 70)
        print("Rank:", result["rank"])
        print("Score:", round(result["score"], 4))
        print("Doc ID:", result["doc_id"])
        print("Title:", result["title"])
        print("Stance:", result["stance"])
        print("Text Preview:")
        print(result["original_text"][:300])


def main():
    query = "Should teachers get tenure?"

    search_service = BM25SearchService()

    experiments = [
        {
            "title": "BM25 Default Parameters",
            "k1": 1.5,
            "b": 0.75,
        },
        {
            "title": "BM25 Lower Length Normalization",
            "k1": 1.2,
            "b": 0.25,
        },
        {
            "title": "BM25 Higher Length Normalization",
            "k1": 2.0,
            "b": 1.0,
        },
    ]

    print("=" * 70)
    print("BM25 PARAMETER COMPARISON TEST")
    print("=" * 70)
    print("Query:", query)

    for experiment in experiments:
        results = search_service.search(
            query_text=query,
            top_k=3,
            k1=experiment["k1"],
            b=experiment["b"],
        )

        print_results(experiment["title"], results)


if __name__ == "__main__":
    main()