from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.bm25_search_service import BM25SearchService
from services.topic_detection_service import TopicDetectionService


def print_topic_detection_result(query, topic_output):
    print("\n" + "=" * 70)
    print("TOPIC DETECTION TEST")
    print("=" * 70)

    print(f"Query: {query}")
    print(f"Detected Topic Label: {topic_output['topic_label']}")
    print(f"Document Count: {topic_output['document_count']}")
    print(f"Method: {topic_output['method']}")
    print(f"Message: {topic_output['message']}")

    print("\nTop Topic Terms:")

    for item in topic_output["top_terms"]:
        print(f"- {item['term']} | score: {item['score']}")


def main():
    query = "Should teachers get tenure?"

    search_service = BM25SearchService()
    topic_service = TopicDetectionService(top_terms=10)

    results = search_service.search(
        query_text=query,
        top_k=20,
        k1=1.5,
        b=0.75,
    )

    topic_output = topic_service.detect_topic(
        query_text=query,
        results=results,
        top_n=10,
    )

    print_topic_detection_result(query, topic_output)

    assert topic_output["topic_label"] != "No topic detected"
    assert topic_output["document_count"] == len(results)
    assert len(topic_output["top_terms"]) > 0


if __name__ == "__main__":
    main()