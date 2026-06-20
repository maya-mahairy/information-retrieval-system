import json
import sqlite3
import time
from typing import List, Optional, Tuple

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from config import (
    DOCUMENT_STORE_DB_PATH,
    PROCESSED_DOCS_PATH,
    TFIDF_MATRIX_PATH,
    TFIDF_VECTORIZER_PATH,
)


def read_processed_docs(limit: Optional[int] = None):
    count = 0

    with open(PROCESSED_DOCS_PATH, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            yield json.loads(line)
            count += 1

            if limit is not None and count >= limit:
                break


def create_document_store(limit: Optional[int] = None, batch_size: int = 1000) -> int:
    if DOCUMENT_STORE_DB_PATH.exists():
        DOCUMENT_STORE_DB_PATH.unlink()

    connection = sqlite3.connect(DOCUMENT_STORE_DB_PATH)
    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE documents (
            row_id INTEGER PRIMARY KEY,
            doc_id TEXT,
            title TEXT,
            stance TEXT,
            url TEXT,
            original_text TEXT,
            cleaned_text TEXT
        )
        """
    )

    cursor.execute("CREATE INDEX idx_documents_doc_id ON documents(doc_id)")

    batch = []
    row_id = 0

    for doc in tqdm(read_processed_docs(limit=limit), desc="Creating document store"):
        batch.append(
            (
                row_id,
                doc.get("doc_id", ""),
                doc.get("title", ""),
                doc.get("stance", ""),
                doc.get("url", ""),
                doc.get("original_text", ""),
                doc.get("cleaned_text", ""),
            )
        )

        row_id += 1

        if len(batch) >= batch_size:
            cursor.executemany(
                """
                INSERT INTO documents (
                    row_id, doc_id, title, stance, url, original_text, cleaned_text
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                batch,
            )
            connection.commit()
            batch = []

    if batch:
        cursor.executemany(
            """
            INSERT INTO documents (
                row_id, doc_id, title, stance, url, original_text, cleaned_text
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            batch,
        )
        connection.commit()

    connection.close()

    return row_id


def load_cleaned_texts(limit: Optional[int] = None) -> List[str]:
    texts = []

    for doc in tqdm(read_processed_docs(limit=limit), desc="Loading cleaned texts"):
        cleaned_text = doc.get("cleaned_text", "")
        texts.append(cleaned_text)

    return texts


def build_tfidf_index(
    limit: Optional[int] = None,
    max_features: int = 100000,
    min_df: int = 2,
    max_df: float = 0.95,
):
    start_time = time.time()

    print("=" * 70)
    print("BUILDING TF-IDF INDEX")
    print("=" * 70)

    print("\nStep 1: Creating SQLite document store...")
    docs_count = create_document_store(limit=limit)
    print("Documents stored:", docs_count)
    print("Document store path:", DOCUMENT_STORE_DB_PATH)

    print("\nStep 2: Loading cleaned texts...")
    texts = load_cleaned_texts(limit=limit)
    print("Texts loaded:", len(texts))

    print("\nStep 3: Building TF-IDF matrix...")

    vectorizer = TfidfVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )

    tfidf_matrix = vectorizer.fit_transform(texts)

    print("TF-IDF matrix shape:", tfidf_matrix.shape)
    print("Vocabulary size:", len(vectorizer.vocabulary_))
    print("Non-zero values:", tfidf_matrix.nnz)

    print("\nStep 4: Saving TF-IDF index...")
    joblib.dump(vectorizer, TFIDF_VECTORIZER_PATH)
    sparse.save_npz(TFIDF_MATRIX_PATH, tfidf_matrix)

    elapsed = time.time() - start_time

    print("\nTF-IDF vectorizer saved to:", TFIDF_VECTORIZER_PATH)
    print("TF-IDF matrix saved to:", TFIDF_MATRIX_PATH)
    print("Total time:", round(elapsed, 2), "seconds")
    print("\nTF-IDF index built successfully.")

    return {
        "docs_count": docs_count,
        "matrix_shape": tfidf_matrix.shape,
        "vocabulary_size": len(vectorizer.vocabulary_),
        "non_zero_values": int(tfidf_matrix.nnz),
        "time_seconds": round(elapsed, 2),
    }