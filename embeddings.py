# backend/embeddings.py
"""Text-embedding helper (google-genai SDK on Vertex AI).

Claude/Anthropic does not provide an embeddings API, so the RAG vector layer
uses Google's text-embedding model via the google-genai SDK, pointed at Vertex
AI. This lives in one place so both the ingestion script (ingest_corpus.py) and
the API (main.py) embed text the same way — same model and dimension.

Migrated off the deprecated `vertexai.language_models` SDK (removal ~2026-06-24)
to `google-genai`. The Vertex-backed client uses Application Default Credentials.
"""
import os
from google import genai
from google.genai import types

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "bgita-teacher")
EMBED_REGION = os.getenv("VERTEX_EMBED_REGION", "us-central1")
EMBED_MODEL = os.getenv("VERTEX_EMBED_MODEL", "text-embedding-004")

# Vertex-backed google-genai client (reads ADC for auth). Created once at import
# so both embed helpers share a single client.
_client = genai.Client(vertexai=True, project=PROJECT_ID, location=EMBED_REGION)


def embed_documents(texts, batch_size=50):
    """Embed a list of passages for storage. Returns a list of float vectors.

    Uses task_type RETRIEVAL_DOCUMENT so the vectors are optimized for being
    searched against (the query side uses RETRIEVAL_QUERY).
    """
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = _client.models.embed_content(
            model=EMBED_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        out.extend(e.values for e in resp.embeddings)
    return out


def embed_query(text_str):
    """Embed a single user message for retrieval. Returns one float vector."""
    resp = _client.models.embed_content(
        model=EMBED_MODEL,
        contents=[text_str],
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return resp.embeddings[0].values


def to_pgvector_literal(vec):
    """Format a float vector as a pgvector text literal: '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


def parse_pgvector(value):
    """Parse a pgvector value from the DB into a list of floats.

    pg8000 returns the vector column as the text literal '[0.1,0.2,...]'. Returns
    an empty list on anything unparseable so callers can fall back gracefully.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    s = str(value).strip().strip("[]")
    if not s:
        return []
    try:
        return [float(x) for x in s.split(",")]
    except ValueError:
        return []
