from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


class ResultClusteringService:
    def __init__(
        self,
        max_clusters: int = 3,
        max_features: int = 700,
        top_terms_per_cluster: int = 5,
    ):
        self.max_clusters = max_clusters
        self.max_features = max_features
        self.top_terms_per_cluster = top_terms_per_cluster

    def cluster_results(
        self,
        results: list[dict[str, Any]],
        max_clusters: int | None = None,
    ) -> dict[str, Any]:
        if not results:
            return {
                "cluster_count": 0,
                "clusters": [],
                "message": "No results available for clustering.",
            }

        texts = [self._build_result_text(result) for result in results]
        valid_items = [
            (index, result, text)
            for index, (result, text) in enumerate(zip(results, texts), start=1)
            if text.strip()
        ]

        if len(valid_items) < 2:
            return self._single_cluster(valid_items)

        cluster_count = self._resolve_cluster_count(
            item_count=len(valid_items),
            requested_clusters=max_clusters,
        )
        extra_stop_words = {
    "http",
    "https",
    "www",
    "com",
    "org",
    "net",
    "tinyurl",
    "html",
    "php",
    "amp",
    "nbsp",
    "debate",
    "round",
    "argument",
    "opponent",
}
        
        english_stop_words = list(TfidfVectorizer(stop_words="english").get_stop_words())
        custom_stop_words = english_stop_words + list(extra_stop_words)


        vectorizer = TfidfVectorizer(
    stop_words=custom_stop_words,
    max_features=self.max_features,
    ngram_range=(1, 2),
    min_df=1,
)
        matrix = vectorizer.fit_transform([item[2] for item in valid_items])

        if matrix.shape[0] < 2 or matrix.shape[1] == 0:
            return self._single_cluster(valid_items)

        model = KMeans(
            n_clusters=cluster_count,
            random_state=42,
            n_init=10,
        )

        labels = model.fit_predict(matrix)
        feature_names = np.array(vectorizer.get_feature_names_out())

        grouped: dict[int, list[tuple[int, dict[str, Any], str]]] = defaultdict(list)

        for item, label in zip(valid_items, labels):
            grouped[int(label)].append(item)

        clusters = []

        for cluster_id, items in grouped.items():
            top_terms = self._extract_top_terms(
                center=model.cluster_centers_[cluster_id],
                feature_names=feature_names,
            )

            documents = []

            for original_rank, result, _ in items:
                documents.append(
                    {
                        "rank": original_rank,
                        "doc_id": result.get("doc_id", ""),
                        "title": result.get("title", "Untitled Document"),
                        "score": self._safe_float(result.get("score")),
                        "stance": result.get("stance", ""),
                    }
                )

            clusters.append(
                {
                    "cluster_id": cluster_id + 1,
                    "label": self._make_cluster_label(top_terms),
                    "top_terms": top_terms,
                    "size": len(items),
                    "documents": documents,
                }
            )

        clusters.sort(key=lambda cluster: cluster["size"], reverse=True)

        for new_id, cluster in enumerate(clusters, start=1):
            cluster["cluster_id"] = new_id

        return {
            "cluster_count": len(clusters),
            "clusters": clusters,
            "message": "Search results were clustered successfully.",
        }

    def _resolve_cluster_count(
        self,
        item_count: int,
        requested_clusters: int | None,
    ) -> int:
        cluster_count = requested_clusters or self.max_clusters
        cluster_count = max(1, min(cluster_count, self.max_clusters, item_count))

        if item_count <= 3:
            cluster_count = min(cluster_count, item_count)

        return cluster_count

    def _build_result_text(self, result: dict[str, Any]) -> str:
        title = result.get("title", "") or ""
        original_text = result.get("original_text", "") or ""
        text = result.get("text", "") or ""
        cleaned_text = result.get("cleaned_text", "") or ""

        return f"{title} {original_text} {text} {cleaned_text}".strip()

    def _extract_top_terms(
        self,
        center: np.ndarray,
        feature_names: np.ndarray,
    ) -> list[str]:
        if center.size == 0:
            return []

        top_indices = center.argsort()[::-1][: self.top_terms_per_cluster]
        terms = [feature_names[index] for index in top_indices if center[index] > 0]

        return terms

    def _make_cluster_label(self, top_terms: list[str]) -> str:
        if not top_terms:
            return "General search results"

        return " / ".join(top_terms[:3])

    def _safe_float(self, value: Any) -> float | None:
        try:
            return float(value)
        except Exception:
            return None

    def _single_cluster(
        self,
        valid_items: list[tuple[int, dict[str, Any], str]],
    ) -> dict[str, Any]:
        documents = []

        for original_rank, result, _ in valid_items:
            documents.append(
                {
                    "rank": original_rank,
                    "doc_id": result.get("doc_id", ""),
                    "title": result.get("title", "Untitled Document"),
                    "score": self._safe_float(result.get("score")),
                    "stance": result.get("stance", ""),
                }
            )

        return {
            "cluster_count": 1 if documents else 0,
            "clusters": [
                {
                    "cluster_id": 1,
                    "label": "General search results",
                    "top_terms": [],
                    "size": len(documents),
                    "documents": documents,
                }
            ]
            if documents
            else [],
            "message": "Not enough results for multi-cluster grouping.",
        }