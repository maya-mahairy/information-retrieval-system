from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class TopicDetectionService:
    def __init__(
        self,
        max_features: int = 1000,
        top_terms: int = 10,
    ):
        self.max_features = max_features
        self.top_terms = top_terms

    def detect_topic(
        self,
        query_text: str,
        results: list[dict[str, Any]],
        top_n: int | None = None,
    ) -> dict[str, Any]:
        top_n = top_n or self.top_terms

        if not query_text.strip() and not results:
            return {
                "topic_label": "No topic detected",
                "top_terms": [],
                "document_count": 0,
                "method": "TF-IDF topic term extraction",
                "message": "No query or search results were provided.",
            }

        documents = self._build_documents(query_text=query_text, results=results)

        if not documents:
            return {
                "topic_label": "No topic detected",
                "top_terms": [],
                "document_count": 0,
                "method": "TF-IDF topic term extraction",
                "message": "No valid text was available for topic detection.",
            }

        vectorizer = TfidfVectorizer(
            stop_words=self._build_stop_words(),
            max_features=self.max_features,
            ngram_range=(1, 2),
            min_df=1,
        )

        matrix = vectorizer.fit_transform(documents)

        if matrix.shape[1] == 0:
            return {
                "topic_label": "No topic detected",
                "top_terms": [],
                "document_count": len(results),
                "method": "TF-IDF topic term extraction",
                "message": "No informative terms were found.",
            }

        feature_names = np.array(vectorizer.get_feature_names_out())
        average_scores = np.asarray(matrix.mean(axis=0)).ravel()

        ranked_indices = average_scores.argsort()[::-1]

        top_terms = []

        for index in ranked_indices:
            score = float(average_scores[index])

            if score <= 0:
                continue

            term = feature_names[index]

            top_terms.append(
                {
                    "term": term,
                    "score": round(score, 6),
                }
            )

            if len(top_terms) >= top_n:
                break

        topic_label = self._make_topic_label(top_terms)

        return {
            "topic_label": topic_label,
            "top_terms": top_terms,
            "document_count": len(results),
            "method": "TF-IDF topic term extraction",
            "message": "Topic terms were extracted successfully.",
        }

    def _build_documents(
        self,
        query_text: str,
        results: list[dict[str, Any]],
    ) -> list[str]:
        documents = []

        if query_text.strip():
            documents.append(query_text.strip())

        for result in results:
            text_parts = [
                result.get("title", "") or "",
                result.get("original_text", "") or "",
                result.get("text", "") or "",
                result.get("cleaned_text", "") or "",
            ]

            combined_text = " ".join(text_parts).strip()

            if combined_text:
                documents.append(combined_text)

        return documents

    def _build_stop_words(self) -> list[str]:
        extra_stop_words = {
            "new",
"old",
"know",
"want",
"need",
"good",
"bad",
"yes",
"no",
            "http",
            "https",
            "www",
            "com",
            "org",
            "net",
            "html",
            "php",
            "amp",
            "nbsp",
            "tinyurl",
            "debate",
            "round",
            "argument",
            "opponent",
            "pro",
            "con",
            "said",
            "say",
            "says",
            "think",
            "people",
            "thing",
            "things",
            "just",
            "like",
            "make",
            "does",
            "did",
            "don",
            "isn",
            "aren",
            "would",
            "could",
            "should",
        }

        english_stop_words = TfidfVectorizer(
            stop_words="english"
        ).get_stop_words()

        return list(english_stop_words) + list(extra_stop_words)

    def _make_topic_label(self, top_terms: list[dict[str, Any]]) -> str:
        if not top_terms:
            return "No topic detected"

        terms = [item["term"] for item in top_terms[:4]]

        return " / ".join(terms)