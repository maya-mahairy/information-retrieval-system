from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.query_refinement_service import QueryRefinementService


def print_refinement_result(result):
    print("\n" + "=" * 70)
    print("QUERY REFINEMENT RESULT")
    print("=" * 70)

    print("Original Query:")
    print(result["original_query"])

    print("\nNormalized Query:")
    print(result["normalized_query"])

    print("\nCorrected Query:")
    print(result["corrected_query"])

    print("\nExpanded Query:")
    print(result["expanded_query"])

    print("\nCorrections:")
    if result["corrections"]:
        for correction in result["corrections"]:
            print(
                "-",
                correction["original"],
                "=>",
                correction["corrected"],
            )
    else:
        print("No spelling corrections.")

    print("\nAdded Terms:")
    if result["added_terms"]:
        for item in result["added_terms"]:
            print(
                "-",
                item["source_token"],
                "=>",
                item["added_term"],
            )
    else:
        print("No synonym expansion terms.")

    print("\nSuggestions:")
    if result["suggestions"]:
        for suggestion in result["suggestions"]:
            print("-", suggestion)
    else:
        print("No suggestions generated.")


def main():
    refinement_service = QueryRefinementService()

    queries = [
        "Should techers get tenur?",
        "Should educators be armed in classrooms?",
        "Should students be rewarded for going to school?",
        "Should college professors have job security?",
    ]

    for query in queries:
        result = refinement_service.refine_query(query)
        print_refinement_result(result)


if __name__ == "__main__":
    main()