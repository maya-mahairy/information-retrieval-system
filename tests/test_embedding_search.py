from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.embedding_search_service import EmbeddingSearchService


def print_results(query, results):
    print("\n" + "=" * 70)
    print("EMBEDDING SEARCH RESULTS")
    print("=" * 70)

    print("Query:", query)

    if not results:
        print("No results found.")
        return

    print("Model:", results[0]["model_name"])

    for result in results:
        print("\n" + "-" * 70)
        print("Rank:", result["rank"])
        print("Score:", round(result["score"], 4))
        print("Doc ID:", result["doc_id"])
        print("Title:", result["title"])
        print("Stance:", result["stance"])
        print("URL:", result["url"])
        print("Text Preview:")
        print(result["original_text"][:500])


def main():
    search_service = EmbeddingSearchService()

    queries = [
    "Should teachers carry guns at school?",
    "Should educators be armed in classrooms?",
    "Should students be rewarded for going to school?",
]

    for query in queries:
        results = search_service.search(query, top_k=5)
        print_results(query, results)


if __name__ == "__main__":
    main()