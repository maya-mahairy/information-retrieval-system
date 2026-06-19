from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.bm25_search_service import BM25SearchService
from services.result_clustering_service import ResultClusteringService


def print_cluster_summary(cluster_output):
    print("\n" + "=" * 70)
    print("RESULT CLUSTERING TEST")
    print("=" * 70)

    print(f"Cluster count: {cluster_output['cluster_count']}")
    print(f"Message: {cluster_output['message']}")

    for cluster in cluster_output["clusters"]:
        print("\n" + "-" * 70)
        print(f"Cluster {cluster['cluster_id']}")
        print(f"Label: {cluster['label']}")
        print(f"Size: {cluster['size']}")
        print(f"Top terms: {', '.join(cluster['top_terms'])}")

        print("\nDocuments:")
        for document in cluster["documents"][:5]:
            print(
                f"- Rank {document['rank']} | "
                f"{document['title']} | "
                f"Score: {document['score']}"
            )


def main():
    query = "Should teachers get tenure?"

    search_service = BM25SearchService()
    clustering_service = ResultClusteringService(max_clusters=3)

    results = search_service.search(
        query_text=query,
        top_k=20,
        k1=1.5,
        b=0.75,
    )

    cluster_output = clustering_service.cluster_results(
        results=results,
        max_clusters=3,
    )

    print(f"Query: {query}")
    print_cluster_summary(cluster_output)

    assert cluster_output["cluster_count"] > 0
    assert len(cluster_output["clusters"]) > 0

    total_clustered_documents = sum(
        cluster["size"] for cluster in cluster_output["clusters"]
    )

    assert total_clustered_documents == len(results)


if __name__ == "__main__":
    main()