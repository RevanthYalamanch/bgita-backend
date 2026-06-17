# backend/embeddings.py
"""Vertex AI text-embedding helper.

Claude/Anthropic does not provide an embeddings API, so the RAG vector layer
uses Google's Vertex AI text-embedding model. This lives in one place so both
the ingestion script (ingest_corpus.py) and the API (main.py) embed text the
same way — with the same model and dimension.

NOTE: embeddings run through the *google* vertexai SDK (us-central1 by default),
which is independent of the AnthropicVertex client used for chat in main.py.
"""
import os
import vertexai
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "bgita-teacher")
EMBED_REGION = os.getenv("VERTEX_EMBED_REGION", "us-central1")
EMBED_MODEL = os.getenv("VERTEX_EMBED_MODEL", "text-embedding-004")

vertexai.init(project=PROJECT_ID, location=EMBED_REGION)
_model = TextEmbeddingModel.from_pretrained(EMBED_MODEL)


def embed_documents(texts, batch_size=50):
    """Embed a list of passages for storage. Returns a list of float vectors.

    Uses task_type RETRIEVAL_DOCUMENT so the vectors are optimized for being
    searched against (the query side uses RETRIEVAL_QUERY).
    """
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = [TextEmbeddingInput(t, "RETRIEVAL_DOCUMENT") for t in batch]
        embeddings = _model.get_embeddings(inputs)
        out.extend(e.values for e in embeddings)
    return out


def embed_query(text_str):
    """Embed a single user message for retrieval. Returns one float vector."""
    inp = TextEmbeddingInput(text_str, "RETRIEVAL_QUERY")
    return _model.get_embeddings([inp])[0].values


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
