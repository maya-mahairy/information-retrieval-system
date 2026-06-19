from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.hybrid_serial_search_service import HybridSerialSearchService


def print_results(query_text, results):
    print("\n" + "=" * 70)
    print("HYBRID SERIAL SEARCH RESULTS")
    print("=" * 70)
    print("Query:", query_text)

    if not results:
        print("No results found.")
        return

    first_result = results[0]

    print("BM25 candidates:", first_result["bm25_candidates"])
    print("BM25 parameters: k1 =", first_result["k1"], ", b =", first_result["b"])
    print("Hybrid alpha:", first_result["alpha"])

    for result in results:
        print("\n" + "-" * 70)
        print("Rank:", result["rank"])
        print("Hybrid Score:", round(result["hybrid_score"], 4))
        print("BM25 Score:", round(result["bm25_score"], 4))
        print("Embedding Score:", round(result["embedding_score"], 4))
        print("Doc ID:", result["doc_id"])
        print("Title:", result["title"])
        print("Stance:", result["stance"])
        print("URL:", result["url"])
        print("Text Preview:")
        print(result["original_text"][:500])


def main():
    search_service = HybridSerialSearchService()

    queries = [
        "Should teachers get tenure?",
        "Should teachers have job protection and tenure?",
        "Should students be rewarded for going to school?",
    ]

    for query_text in queries:
        results = search_service.search(
            query_text=query_text,
            top_k=5,
            bm25_candidates=100,
            k1=1.5,
            b=0.75,
            alpha=0.5,
        )

        print_results(query_text, results)


if __name__ == "__main__":
    main()