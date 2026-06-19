from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.bm25_search_service import BM25SearchService


def main():
    query = "Should teachers get tenure?"

    print("=" * 70)
    print("BM25 SEARCH TEST")
    print("=" * 70)

    print("Query:", query)

    search_service = BM25SearchService()

    results = search_service.search(
        query_text=query,
        top_k=5,
        k1=1.5,
        b=0.75,
    )

    if not results:
        print("No results found.")
        return

    print("Cleaned query:", results[0]["cleaned_query"])
    print("Query tokens:", results[0]["query_tokens"])
    print("BM25 parameters: k1 =", results[0]["k1"], ", b =", results[0]["b"])

    print("\nTop results:")

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


if __name__ == "__main__":
    main()