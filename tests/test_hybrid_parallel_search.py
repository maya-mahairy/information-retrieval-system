from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.hybrid_parallel_search_service import HybridParallelSearchService


def print_results(query_text, results):
    print("\n" + "=" * 70)
    print("HYBRID PARALLEL SEARCH RESULTS")
    print("=" * 70)
    print("Query:", query_text)

    if not results:
        print("No results found.")
        return

    first_result = results[0]

    print("Fusion method:", first_result["fusion_method"])
    print("RRF k:", first_result["rrf_k"])
    print("BM25 weight:", first_result["bm25_weight"])
    print("TF-IDF weight:", first_result["tfidf_weight"])
    print("BM25 parameters: k1 =", first_result["k1"], ", b =", first_result["b"])

    for result in results:
        print("\n" + "-" * 70)
        print("Rank:", result["rank"])
        print("Fusion Score:", round(result["fusion_score"], 6))
        print("BM25 Rank:", result["bm25_rank"])
        print("BM25 Score:", result["bm25_score"])
        print("TF-IDF Rank:", result["tfidf_rank"])
        print("TF-IDF Score:", result["tfidf_score"])
        print("Doc ID:", result["doc_id"])
        print("Title:", result["title"])
        print("Stance:", result["stance"])
        print("URL:", result["url"])
        print("Text Preview:")
        print(result["original_text"][:500])


def main():
    search_service = HybridParallelSearchService()

    queries = [
        "Should teachers get tenure?",
        "Should teachers have job protection and tenure?",
        "Should students be rewarded for going to school?",
    ]

    for query_text in queries:
        results = search_service.search(
            query_text=query_text,
            top_k=5,
            bm25_top_k=100,
            tfidf_top_k=100,
            rrf_k=60,
            bm25_weight=0.8,
            tfidf_weight=0.2,
            k1=1.5,
            b=0.75,
        )

        print_results(query_text, results)


if __name__ == "__main__":
    main()