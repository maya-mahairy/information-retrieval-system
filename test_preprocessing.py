from services.dataset_service import load_docs_stream, load_queries
from services.preprocessing_service import preprocess_document, preprocess_query


def main():
    print("=" * 70)
    print("PREPROCESSING TEST")
    print("=" * 70)

    print("\nLoading first query...")
    queries = load_queries()
    first_query = dict(queries[0])

    processed_query = preprocess_query(first_query)

    print("\n" + "-" * 70)
    print("Original Query")
    print("-" * 70)
    print(processed_query["original_text"])

    print("\nCleaned Query")
    print("-" * 70)
    print(processed_query["cleaned_text"])

    print("\nQuery Tokens")
    print("-" * 70)
    print(processed_query["tokens"])

    print("\nLoading first document...")
    docs_stream = load_docs_stream()
    first_doc = dict(next(iter(docs_stream)))

    processed_doc = preprocess_document(first_doc)

    print("\n" + "-" * 70)
    print("Document Metadata")
    print("-" * 70)
    print("Doc ID:", processed_doc["doc_id"])
    print("Title:", processed_doc["title"])
    print("Stance:", processed_doc["stance"])
    print("URL:", processed_doc["url"])

    print("\nOriginal Document Text Preview")
    print("-" * 70)
    print(processed_doc["original_text"][:500])

    print("\nCleaned Document Text Preview")
    print("-" * 70)
    print(processed_doc["cleaned_text"][:500])

    print("\nFirst 30 Document Tokens")
    print("-" * 70)
    print(processed_doc["tokens"][:30])

    print("\nPreprocessing test completed successfully.")


if __name__ == "__main__":
    main()