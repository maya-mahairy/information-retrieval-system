import html
import re
from typing import Any, Dict, List

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer


STOP_WORDS = set(stopwords.words("english"))

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


def normalize_text(text: str) -> str:
    """
    Normalizes raw text before tokenization.

    Steps:
    - Convert HTML entities to normal characters.
    - Lowercase the text.
    - Remove URLs.
    - Replace non-alphanumeric characters with spaces.
    - Remove extra spaces.
    """
    if text is None:
        return ""

    text = str(text)
    text = html.unescape(text)
    text = text.lower()

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Keep letters and numbers only; replace punctuation/symbols with spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Normalize repeated spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_text(normalized_text: str) -> List[str]:
    """
    Splits normalized text into word tokens.

    We use regex tokenization to keep the preprocessing simple and stable.
    """
    return re.findall(r"\b[a-z0-9]+\b", normalized_text)


def preprocess_tokens(tokens: List[str], use_stemming: bool = False) -> List[str]:
    """
    Removes stopwords and applies lemmatization by default.

    Stemming is optional. We keep lemmatization as default because it keeps
    words more readable and suitable for explanation in the report/interview.
    """
    processed_tokens = []

    for token in tokens:
        # Remove very short tokens and stopwords
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
    """
    Full preprocessing pipeline for any text.

    Returns both the original and cleaned forms to support:
    - Original text display in UI.
    - Cleaned text/tokens for retrieval models.
    """
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
    """
    Builds the searchable text for a document.

    We use title + text because:
    - title gives the general topic.
    - text contains the actual argument/content.

    We keep stance and url as metadata for display, not as core searchable text.
    """
    title = doc.get("title", "")
    text = doc.get("text", "")

    return f"{title} {text}".strip()


def preprocess_document(doc: Dict[str, Any], use_stemming: bool = False) -> Dict[str, Any]:
    """
    Preprocesses a dataset document while preserving its original metadata.
    """
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
    """
    Builds query text for retrieval.

    By default, we use query['text'] only because this represents the user's actual query.
    The description/narrative fields are useful for understanding relevance, but using them
    in retrieval may give the system extra information not available in a real user query.
    """
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
    """
    Preprocesses a query using the same pipeline used for documents.
    This is required to keep query/document representations compatible.
    """
    query_text = build_query_search_text(query, use_description=use_description)
    processed = preprocess_text(query_text, use_stemming=use_stemming)

    return {
        "query_id": str(query.get("query_id", "")),
        "original_text": query_text,
        "normalized_text": processed["normalized_text"],
        "tokens": processed["tokens"],
        "cleaned_text": processed["cleaned_text"],
    }