import sqlite3
from typing import Dict, List

import joblib
import numpy as np
from scipy import sparse

from config import DOCUMENT_STORE_DB_PATH, TFIDF_MATRIX_PATH, TFIDF_VECTORIZER_PATH
from services.preprocessing_service import preprocess_text


class TFIDFSearchService:
    """
    Search service using TF-IDF representation and cosine similarity.
    """

    def __init__(self):
        self.vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
        self.tfidf_matrix = sparse.load_npz(TFIDF_MATRIX_PATH)

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

    def search(self, query_text: str, top_k: int = 10) -> List[Dict]:
        """
        Searches documents using TF-IDF cosine similarity.
        """
        processed_query = preprocess_text(query_text)
        cleaned_query = processed_query["cleaned_text"]

        if not cleaned_query:
            return []

        query_vector = self.vectorizer.transform([cleaned_query])

        scores = self.tfidf_matrix.dot(query_vector.T).toarray().ravel()

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
                }
            )

        return results