# backend/ingest_corpus.py
"""One-time ingestion: chunk the public-domain Gita corpus, embed each chunk via
Vertex (google-genai), and load the vectors into a pgvector table (`gita_chunks`).

Corpus is Edwin Arnold's public-domain translation "The Song Celestial"
(Project Gutenberg #2388), saved as gita_public_domain.txt. Per Claude.md, ONLY
public-domain translations may be ingested — do NOT point CORPUS_PATH at the
copyrighted Easwaran text.

Run once (re-run safely — it rebuilds the table from scratch):

    cd backend
    python ingest_corpus.py

After this, main.py's get_clinical_context() does semantic search over the
vectors this script produced.
"""
import os
from sqlalchemy import text

from database import engine
from embeddings import embed_documents, to_pgvector_literal, EMBED_MODEL

CORPUS_PATH = os.getenv("CORPUS_PATH", "gita_public_domain.txt")
TABLE = "gita_chunks"
MIN_WORDS = 50  # merge paragraphs shorter than this into the previous chunk


def load_chunks(path):
    """Split the corpus into paragraph-sized chunks.

    The text is cleanly separated by blank lines, so each paragraph is a
    coherent unit of commentary. Very short fragments are merged into the
    previous chunk so we don't embed near-empty snippets.
    """
    with open(path, encoding="utf-8") as f:
        raw = f.read()

    paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]

    chunks = []
    for para in paragraphs:
        if chunks and len(para.split()) < MIN_WORDS:
            chunks[-1] = chunks[-1] + " " + para
        else:
            chunks.append(para)
    return chunks


def main():
    print(f"Reading {CORPUS_PATH} ...")
    chunks = load_chunks(CORPUS_PATH)
    print(f"  -> {len(chunks)} chunks")

    print(f"Embedding with Vertex model '{EMBED_MODEL}' ...")
    vectors = embed_documents(chunks)
    dim = len(vectors[0])
    print(f"  -> {len(vectors)} vectors of dimension {dim}")

    print(f"Rebuilding table '{TABLE}' (dimension {dim}) ...")
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.execute(text(f"DROP TABLE IF EXISTS {TABLE}"))
        conn.execute(text(f"""
            CREATE TABLE {TABLE} (
                id          SERIAL PRIMARY KEY,
                chunk_index INTEGER,
                content     TEXT,
                embedding   vector({dim})
            )
        """))

        insert = text(f"""
            INSERT INTO {TABLE} (chunk_index, content, embedding)
            VALUES (:i, :c, CAST(:e AS vector))
        """)
        for i, (content, vec) in enumerate(zip(chunks, vectors)):
            conn.execute(insert, {"i": i, "c": content, "e": to_pgvector_literal(vec)})

        # Stamp which embedding model built this corpus. Query vectors MUST come
        # from the same model or cosine search compares vectors from different
        # spaces and silently returns garbage. main.py reads this at boot and
        # warns loudly on a mismatch, turning a silent retrieval bug into a log line.
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS gita_corpus_meta (
                id          SERIAL PRIMARY KEY,
                embed_model TEXT,
                dim         INTEGER,
                chunk_count INTEGER,
                ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.execute(text("DELETE FROM gita_corpus_meta"))
        conn.execute(
            text("INSERT INTO gita_corpus_meta (embed_model, dim, chunk_count) VALUES (:m, :d, :c)"),
            {"m": EMBED_MODEL, "d": dim, "c": len(chunks)},
        )
        print(f"  -> stamped corpus meta: model={EMBED_MODEL} dim={dim}")

        # Cosine-distance index. Optional for ~4k rows (brute force is instant),
        # but cheap to add and future-proofs a larger corpus. Don't fail the
        # ingest if the pgvector build doesn't support hnsw.
        try:
            conn.execute(text(
                f"CREATE INDEX ON {TABLE} USING hnsw (embedding vector_cosine_ops)"
            ))
            print("  -> created hnsw cosine index")
        except Exception as e:
            print(f"  -> skipped index (non-fatal): {e}")

        # GIN index backing the hybrid keyword search in main.py
        # (to_tsvector('english', content) @@ plainto_tsquery(...)). Without it
        # Postgres recomputes the tsvector for every row on each query.
        try:
            conn.execute(text(
                f"CREATE INDEX IF NOT EXISTS {TABLE}_fts ON {TABLE} "
                f"USING GIN (to_tsvector('english', content))"
            ))
            print("  -> created GIN full-text index")
        except Exception as e:
            print(f"  -> skipped fts index (non-fatal): {e}")

    print(f"Done. Loaded {len(chunks)} chunks into '{TABLE}'.")


if __name__ == "__main__":
    main()
