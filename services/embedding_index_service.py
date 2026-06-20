import json
import time
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import (
    DOCUMENT_STORE_DB_PATH,
    EMBEDDING_MATRIX_PATH,
    EMBEDDING_MAX_CHARS,
    EMBEDDING_METADATA_PATH,
    EMBEDDING_MODEL_NAME,
    EXPECTED_DOCS_COUNT,
)
from services.tfidf_index_service import create_document_store, read_processed_docs


def ensure_document_store_exists(limit: Optional[int] = None):
    if DOCUMENT_STORE_DB_PATH.exists():
        print("Using existing document store:", DOCUMENT_STORE_DB_PATH)
        return

    print("Document store not found. Creating it now...")
    create_document_store(limit=limit)


def prepare_embedding_text(doc: dict) -> str:
    text = doc.get("searchable_text", "")

    if not text:
        title = doc.get("title", "")
        original_text = doc.get("original_text", "")
        text = f"{title} {original_text}".strip()

    text = " ".join(str(text).split())

    return text[:EMBEDDING_MAX_CHARS]


def get_docs_count_for_embedding(limit: Optional[int] = None) -> int:
    if limit is not None:
        return int(limit)

    return int(EXPECTED_DOCS_COUNT)


def iter_embedding_batches(limit: Optional[int] = None, batch_size: int = 64):
    batch_texts = []
    processed_count = 0

    for doc in read_processed_docs(limit=limit):
        batch_texts.append(prepare_embedding_text(doc))
        processed_count += 1

        if len(batch_texts) >= batch_size:
            yield batch_texts
            batch_texts = []

    if batch_texts:
        yield batch_texts


def build_embedding_index(
    limit: Optional[int] = None,
    batch_size: int = 64,
):
    start_time = time.time()

    print("=" * 70)
    print("BUILDING EMBEDDING VECTOR STORE")
    print("=" * 70)

    print("\nStep 1: Checking document store...")
    ensure_document_store_exists(limit=limit)

    print("\nStep 2: Loading embedding model...")
    print("Model:", EMBEDDING_MODEL_NAME)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    n_docs = get_docs_count_for_embedding(limit=limit)
    embedding_dim = model.get_sentence_embedding_dimension()

    print("Expected documents:", n_docs)
    print("Embedding dimension:", embedding_dim)
    print("Batch size:", batch_size)
    print("Max chars per document:", EMBEDDING_MAX_CHARS)

    print("\nStep 3: Encoding documents and saving vectors...")

    embedding_matrix = np.lib.format.open_memmap(
        EMBEDDING_MATRIX_PATH,
        mode="w+",
        dtype=np.float32,
        shape=(n_docs, embedding_dim),
    )

    row_start = 0

    progress_bar = tqdm(total=n_docs, desc="Encoding documents")

    for batch_texts in iter_embedding_batches(limit=limit, batch_size=batch_size):
        embeddings = model.encode(
            batch_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        row_end = row_start + embeddings.shape[0]
        embedding_matrix[row_start:row_end] = embeddings

        row_start = row_end
        progress_bar.update(len(batch_texts))

    progress_bar.close()
    embedding_matrix.flush()

    if row_start != n_docs:
        raise ValueError(
            f"Expected {n_docs} embeddings, but created {row_start}. "
            "Check processed_docs.jsonl and EXPECTED_DOCS_COUNT."
        )

    elapsed = time.time() - start_time

    metadata = {
        "model_name": EMBEDDING_MODEL_NAME,
        "docs_count": int(n_docs),
        "embedding_dim": int(embedding_dim),
        "embedding_matrix_path": str(EMBEDDING_MATRIX_PATH),
        "document_store_path": str(DOCUMENT_STORE_DB_PATH),
        "max_chars": EMBEDDING_MAX_CHARS,
        "batch_size": batch_size,
        "is_full_dataset": limit is None,
        "time_seconds": round(elapsed, 2),
    }

    with open(EMBEDDING_METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print("\nEmbedding matrix saved to:", EMBEDDING_MATRIX_PATH)
    print("Embedding metadata saved to:", EMBEDDING_METADATA_PATH)
    print("Total time:", round(elapsed, 2), "seconds")
    print("\nEmbedding vector store built successfully.")

    return metadata