import json
import sqlite3
from typing import Dict, List, Tuple

import joblib
import numpy as np
from scipy import sparse

from config import (
    BM25_COUNT_MATRIX_PATH,
    BM25_COUNT_VECTORIZER_PATH,
    BM25_DOC_LENGTHS_PATH,
    BM25_IDF_PATH,
    BM25_METADATA_PATH,
    DOCUMENT_STORE_DB_PATH,
)
from services.preprocessing_service import preprocess_text


class BM25SearchService:
    """
    Search service using BM25 scoring.

    BM25 parameters k1 and b can be changed per query.
    """

    def __init__(self):
        self.vectorizer = joblib.load(BM25_COUNT_VECTORIZER_PATH)
        self.count_matrix = sparse.load_npz(BM25_COUNT_MATRIX_PATH).tocsc()
        self.doc_lengths = np.load(BM25_DOC_LENGTHS_PATH)
        self.idf = np.load(BM25_IDF_PATH)

        with open(BM25_METADATA_PATH, "r", encoding="utf-8") as file:
            self.metadata = json.load(file)

        self.avg_doc_length = float(self.metadata["avg_doc_length"])
        self.default_k1 = float(self.metadata.get("default_k1", 1.5))
        self.default_b = float(self.metadata.get("default_b", 0.75))
        self.docs_count = int(self.metadata["docs_count"])

    def _get_documents_by_row_ids(self, row_ids: List[int]) -> Dict[int, Dict]:
        """
        Fetches document metadata and original text from SQLite by row_id.
        """
        if not row_ids:
            return {}

        placeholders = ",".join(["?"] * len(row_ids))

        connection = sqlite3.connect(DOCUMENT_STORE_DB_PATH)
        connection.row_factory = sqlite3.Row
        cursor = connection.cursor()

        cursor.execute(
            f"""
            SELECT row_id, doc_id, title, stance, url, original_text
            FROM documents
            WHERE row_id IN ({placeholders})
            """,
            row_ids,
        )

        rows = cursor.fetchall()
        connection.close()

        return {
            int(row["row_id"]): {
                "doc_id": row["doc_id"],
                "title": row["title"],
                "stance": row["stance"],
                "url": row["url"],
                "original_text": row["original_text"],
            }
            for row in rows
        }

    def _get_query_term_indices(self, query_text: str) -> Tuple[List[int], str, List[str]]:
        """
        Preprocesses the query and maps its tokens to vocabulary indices.
        """
        processed_query = preprocess_text(query_text)
        cleaned_query = processed_query["cleaned_text"]
        query_tokens = processed_query["tokens"]

        vocabulary = self.vectorizer.vocabulary_

        term_indices = []
        seen = set()

        for token in query_tokens:
            index = vocabulary.get(token)

            if index is None:
                continue

            if index in seen:
                continue

            seen.add(index)
            term_indices.append(index)

        return term_indices, cleaned_query, query_tokens

    def search(
        self,
        query_text: str,
        top_k: int = 10,
        k1: float = None,
        b: float = None,
    ) -> List[Dict]:
        """
        Searches documents using BM25.

        k1 controls term frequency saturation.
        b controls document length normalization.
        """
        if k1 is None:
            k1 = self.default_k1

        if b is None:
            b = self.default_b

        k1 = float(k1)
        b = float(b)

        term_indices, cleaned_query, query_tokens = self._get_query_term_indices(query_text)

        if not term_indices:
            return []

        scores = np.zeros(self.docs_count, dtype=np.float32)

        length_normalization = k1 * (
            1.0 - b + b * (self.doc_lengths / self.avg_doc_length)
        )

        for term_index in term_indices:
            start = self.count_matrix.indptr[term_index]
            end = self.count_matrix.indptr[term_index + 1]

            doc_indices = self.count_matrix.indices[start:end]
            term_frequencies = self.count_matrix.data[start:end].astype(np.float32)

            numerator = term_frequencies * (k1 + 1.0)
            denominator = term_frequencies + length_normalization[doc_indices]

            term_scores = self.idf[term_index] * (numerator / (denominator + 1e-9))
            scores[doc_indices] += term_scores

        if scores.max() <= 0:
            return []

        top_k = min(top_k, len(scores))

        candidate_indices = np.argpartition(scores, -top_k)[-top_k:]
        ranked_indices = candidate_indices[np.argsort(scores[candidate_indices])[::-1]]

        row_ids = [int(index) for index in ranked_indices]
        documents = self._get_documents_by_row_ids(row_ids)

        results = []

        for rank, row_id in enumerate(row_ids, start=1):
            doc = documents.get(row_id, {})

            results.append(
                {
                    "rank": rank,
                    "score": float(scores[row_id]),
                    "row_id": row_id,
                    "doc_id": doc.get("doc_id", ""),
                    "title": doc.get("title", ""),
                    "stance": doc.get("stance", ""),
                    "url": doc.get("url", ""),
                    "original_text": doc.get("original_text", ""),
                    "cleaned_query": cleaned_query,
                    "query_tokens": query_tokens,
                    "k1": k1,
                    "b": b,
                }
            )

        return results