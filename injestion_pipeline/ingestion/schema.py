"""LanceDB / PyArrow schema definition for the ingestion pipeline.

This schema is used for both table creation and validation.
The vector dimension (384) matches intfloat/multilingual-e5-small output.
"""

import pyarrow as pa

# E5-small embedding dimension
EMBEDDING_DIM = 384

# Valid chunk strategy labels
CHUNK_STRATEGIES = ("fixed", "semantic", "metadata_aware")

# Language code to formatted locale mapping (e.g. 'ta' -> 'ta-IN', 'hi' -> 'hi-IN')
LANGUAGE_MAP = {
    "en": "en-IN",
    "hi": "hi-IN",
    "bn": "bn-IN",
    "kn": "kn-IN",
    "ml": "ml-IN",
    "mr": "mr-IN",
    "or": "or-IN",
    "pa": "pa-IN",
    "ta": "ta-IN",
    "te": "te-IN",
    "gu": "gu-IN",
    "as": "as-IN",
    "ur": "ur-IN",
    "ne": "ne-IN",
    "kok": "kok-IN",
    "ks": "ks-IN",
    "sd": "sd-IN",
    "sa": "sa-IN",
    "sat": "sat-IN",
    "mni": "mni-IN",
    "brx": "brx-IN",
    "mai": "mai-IN",
    "doi": "doi-IN",
}

# Reverse mapping: 'ta-IN' -> 'ta', 'ta-in' -> 'ta', 'ta' -> 'ta'
LOCALE_TO_CODE = {}
for code, locale in LANGUAGE_MAP.items():
    LOCALE_TO_CODE[code.lower()] = code
    LOCALE_TO_CODE[locale.lower()] = code
    LOCALE_TO_CODE[locale] = code


def to_locale_table_name(lang: str) -> str:
    """Map a short language code or existing locale to standard locale format (e.g. 'ta' -> 'ta-IN')."""
    cleaned = lang.strip().lower()
    return LANGUAGE_MAP.get(cleaned, lang)


def normalize_lang_code(lang: str) -> str:
    """Normalize locale or code to short dataset code (e.g. 'ta-IN' -> 'ta', 'ta' -> 'ta')."""
    cleaned = lang.strip()
    return LOCALE_TO_CODE.get(cleaned.lower(), cleaned)


# Explicit PyArrow schema — never let LanceDB infer it
TABLE_SCHEMA = pa.schema([
    pa.field("chunk_id", pa.string()),
    pa.field("source_doc_id", pa.string()),
    pa.field("chunk_strategy", pa.string()),   # "fixed" | "semantic" | "metadata_aware"
    pa.field("language", pa.string()),
    pa.field("text", pa.string()),
    pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIM)),
])

TABLE_NAME = "passages"
