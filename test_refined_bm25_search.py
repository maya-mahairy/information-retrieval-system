from services.bm25_search_service import BM25SearchService
from services.query_refinement_service import QueryRefinementService


def print_search_results(title, query_text, results):
    print("\n" + "-" * 70)
    print(title)
    print("-" * 70)
    print("Query:", query_text)

    if not results:
        print("No results found.")
        return

    for result in results[:5]:
        print("\nRank:", result["rank"])
        print("Score:", round(result["score"], 4))
        print("Doc ID:", result["doc_id"])
        print("Title:", result["title"])
        print("Stance:", result["stance"])
        print("Text Preview:")
        print(result["original_text"][:300])


def compare_original_corrected_expanded(query_text, bm25_service, refinement_service):
    print("\n" + "=" * 70)
    print("REFINED BM25 SEARCH COMPARISON")
    print("=" * 70)
    print("Original User Query:", query_text)

    refinement = refinement_service.refine_query(query_text)

    print("\nCorrected Query:")
    print(refinement["corrected_query"])

    print("\nExpanded Query:")
    print(refinement["expanded_query"])

    print("\nCorrections:")
    if refinement["corrections"]:
        for correction in refinement["corrections"]:
            print("-", correction["original"], "=>", correction["corrected"])
    else:
        print("No spelling corrections.")

    print("\nAdded Terms:")
    if refinement["added_terms"]:
        for item in refinement["added_terms"]:
            print("-", item["source_token"], "=>", item["added_term"])
    else:
        print("No added terms.")

    original_results = bm25_service.search(
        query_text=query_text,
        top_k=5,
        k1=1.5,
        b=0.75,
    )

    corrected_results = bm25_service.search(
        query_text=refinement["corrected_query"],
        top_k=5,
        k1=1.5,
        b=0.75,
    )

    expanded_results = bm25_service.search(
        query_text=refinement["expanded_query"],
        top_k=5,
        k1=1.5,
        b=0.75,
    )

    print_search_results(
        title="Original Query Results",
        query_text=query_text,
        results=original_results,
    )

    print_search_results(
        title="Corrected Query Results",
        query_text=refinement["corrected_query"],
        results=corrected_results,
    )

    print_search_results(
        title="Expanded Query Results",
        query_text=refinement["expanded_query"],
        results=expanded_results,
    )


def main():
    bm25_service = BM25SearchService()
    refinement_service = QueryRefinementService()

    queries = [
        "Should techers get tenur?",
        "Should educators be armed in classrooms?",
        "Should students be rewarded for going to school?",
        "Should college professors have job security?",
    ]

    for query in queries:
        compare_original_corrected_expanded(
            query_text=query,
            bm25_service=bm25_service,
            refinement_service=refinement_service,
        )


if __name__ == "__main__":
    main()