from services.tfidf_search_service import TFIDFSearchService


def main():
    query = "Should teachers get tenure?"

    print("=" * 70)
    print("TF-IDF SEARCH TEST")
    print("=" * 70)

    print("Query:", query)

    search_service = TFIDFSearchService()
    results = search_service.search(query, top_k=5)

    if not results:
        print("No results found.")
        return

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