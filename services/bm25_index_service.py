import json
import time
from typing import List, Optional

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

from config import (
    BM25_COUNT_MATRIX_PATH,
    BM25_COUNT_VECTORIZER_PATH,
    BM25_DOC_LENGTHS_PATH,
    BM25_IDF_PATH,
    BM25_METADATA_PATH,
    DOCUMENT_STORE_DB_PATH,
)
from services.tfidf_index_service import create_document_store, read_processed_docs


def ensure_document_store_exists(limit: Optional[int] = None):
    if DOCUMENT_STORE_DB_PATH.exists():
        print("Using existing document store:", DOCUMENT_STORE_DB_PATH)
        return

    print("Document store not found. Creating it now...")
    create_document_store(limit=limit)


def load_cleaned_texts(limit: Optional[int] = None) -> List[str]:
    texts = []

    for doc in tqdm(read_processed_docs(limit=limit), desc="Loading cleaned texts"):
        texts.append(doc.get("cleaned_text", ""))

    return texts


def build_bm25_index(
    limit: Optional[int] = None,
    max_features: int = 100000,
    min_df: int = 2,
    max_df: float = 0.95,
    default_k1: float = 1.5,
    default_b: float = 0.75,
):
    start_time = time.time()

    print("=" * 70)
    print("BUILDING BM25 INDEX")
    print("=" * 70)

    print("\nStep 1: Checking document store...")
    ensure_document_store_exists(limit=limit)

    print("\nStep 2: Loading cleaned texts...")
    texts = load_cleaned_texts(limit=limit)
    print("Texts loaded:", len(texts))

    print("\nStep 3: Building count matrix...")

    vectorizer = CountVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        dtype=np.float32,
    )

    count_matrix_csr = vectorizer.fit_transform(texts)

    n_docs = count_matrix_csr.shape[0]
    doc_lengths = np.asarray(count_matrix_csr.sum(axis=1)).ravel().astype(np.float32)
    avg_doc_length = float(doc_lengths.mean())

    document_frequencies = np.asarray(
        count_matrix_csr.getnnz(axis=0),
        dtype=np.float32,
    )

    idf = np.log(
        1.0 + ((n_docs - document_frequencies + 0.5) / (document_frequencies + 0.5))
    ).astype(np.float32)

    count_matrix_csc = count_matrix_csr.tocsc().astype(np.float32)

    print("Count matrix shape:", count_matrix_csc.shape)
    print("Vocabulary size:", len(vectorizer.vocabulary_))
    print("Non-zero values:", count_matrix_csc.nnz)
    print("Average document length:", round(avg_doc_length, 2))

    print("\nStep 4: Saving BM25 index...")

    joblib.dump(vectorizer, BM25_COUNT_VECTORIZER_PATH)
    sparse.save_npz(BM25_COUNT_MATRIX_PATH, count_matrix_csc)
    np.save(BM25_DOC_LENGTHS_PATH, doc_lengths)
    np.save(BM25_IDF_PATH, idf)

    metadata = {
        "docs_count": int(n_docs),
        "matrix_shape": list(count_matrix_csc.shape),
        "vocabulary_size": len(vectorizer.vocabulary_),
        "non_zero_values": int(count_matrix_csc.nnz),
        "avg_doc_length": avg_doc_length,
        "default_k1": default_k1,
        "default_b": default_b,
        "max_features": max_features,
        "min_df": min_df,
        "max_df": max_df,
        "is_full_dataset": limit is None,
    }

    with open(BM25_METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    elapsed = time.time() - start_time

    print("BM25 vectorizer saved to:", BM25_COUNT_VECTORIZER_PATH)
    print("BM25 count matrix saved to:", BM25_COUNT_MATRIX_PATH)
    print("BM25 document lengths saved to:", BM25_DOC_LENGTHS_PATH)
    print("BM25 IDF saved to:", BM25_IDF_PATH)
    print("BM25 metadata saved to:", BM25_METADATA_PATH)
    print("Total time:", round(elapsed, 2), "seconds")
    print("\nBM25 index built successfully.")

    return metadata