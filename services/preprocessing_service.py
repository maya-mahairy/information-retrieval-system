import html
import re
from typing import Any, Dict, List

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer


STOP_WORDS = set(stopwords.words("english"))

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


def normalize_text(text: str) -> str:
    if text is None:
        return ""

    text = str(text)
    text = html.unescape(text)
    text = text.lower()

    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    text = re.sub(r"[^a-z0-9\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_text(normalized_text: str) -> List[str]:
    return re.findall(r"\b[a-z0-9]+\b", normalized_text)


def preprocess_tokens(tokens: List[str], use_stemming: bool = False) -> List[str]:
    processed_tokens = []

    for token in tokens:
        if len(token) <= 2:
            continue

        if token in STOP_WORDS:
            continue

        if use_stemming:
            token = stemmer.stem(token)
        else:
            token = lemmatizer.lemmatize(token)

        processed_tokens.append(token)

    return processed_tokens


def preprocess_text(text: str, use_stemming: bool = False) -> Dict[str, Any]:
    original_text = "" if text is None else str(text)
    normalized_text = normalize_text(original_text)
    raw_tokens = tokenize_text(normalized_text)
    processed_tokens = preprocess_tokens(raw_tokens, use_stemming=use_stemming)
    cleaned_text = " ".join(processed_tokens)

    return {
        "original_text": original_text,
        "normalized_text": normalized_text,
        "tokens": processed_tokens,
        "cleaned_text": cleaned_text,
    }


def build_document_search_text(doc: Dict[str, Any]) -> str:
    title = doc.get("title", "")
    text = doc.get("text", "")

    return f"{title} {text}".strip()


def preprocess_document(doc: Dict[str, Any], use_stemming: bool = False) -> Dict[str, Any]:
    searchable_text = build_document_search_text(doc)
    processed = preprocess_text(searchable_text, use_stemming=use_stemming)

    return {
        "doc_id": str(doc.get("doc_id", "")),
        "title": doc.get("title", ""),
        "stance": doc.get("stance", ""),
        "url": doc.get("url", ""),
        "original_text": doc.get("text", ""),
        "searchable_text": searchable_text,
        "normalized_text": processed["normalized_text"],
        "tokens": processed["tokens"],
        "cleaned_text": processed["cleaned_text"],
    }


def build_query_search_text(query: Dict[str, Any], use_description: bool = False) -> str:
    query_text = query.get("text", "")

    if use_description:
        description = query.get("description", "")
        query_text = f"{query_text} {description}".strip()

    return query_text


def preprocess_query(
    query: Dict[str, Any],
    use_stemming: bool = False,
    use_description: bool = False,
) -> Dict[str, Any]:
    query_text = build_query_search_text(query, use_description=use_description)
    processed = preprocess_text(query_text, use_stemming=use_stemming)

    return {
        "query_id": str(query.get("query_id", "")),
        "original_text": query_text,
        "normalized_text": processed["normalized_text"],
        "tokens": processed["tokens"],
        "cleaned_text": processed["cleaned_text"],
    }