from fastapi import FastAPI, HTTPException, Request, Depends, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
import os
import re
from sqlalchemy import text
from database import engine, get_db
from embeddings import embed_query, to_pgvector_literal, parse_pgvector
from rerank import mmr_select
from auth import (
    hash_password, verify_password, needs_rehash, ADMIN_SIGNUP_CODE,
    create_access_token, decode_access_token,
)
from safety import (detect_crisis, CRISIS_RESPONSE,
                    detect_harm_to_others, HARM_TO_OTHERS_RESPONSE)
from ratelimit import SlidingWindowLimiter
from sqlalchemy.orm import Session
from typing import Optional, List, Literal
import time
import traceback


def _log_exc(label: str, e: Exception):
    """Print a rich, greppable error line + full traceback.

    Bare `print(f"... {e}")` loses the exception *type* and stack, which is why
    config errors (e.g. a Vertex 404 on a bad model id) looked like transient
    blips. This always surfaces type, repr, and traceback so Cloud Run logs tell
    the whole story.
    """
    print(f"🚨 {label}: [{type(e).__name__}] {e!r}")
    traceback.print_exc()


app = FastAPI()

class LogCreate(BaseModel):
    # email is ignored server-side (identity comes from the auth token); kept
    # optional for backward compatibility with older clients that still send it.
    email: Optional[str] = None
    mood_score: int
    diary_text: str
    # Rich mood-logging fields (all optional so older clients keep working and
    # the overall mood_score alone is still a valid check-in). Lists are capped
    # to bound payload size; energy is a 1–5 self-rating parallel to mood.
    emotions: Optional[List[str]] = Field(default=None, max_length=20)
    energy: Optional[int] = Field(default=None, ge=1, le=5)
    sleep: Optional[str] = Field(default=None, max_length=50)
    activities: Optional[List[str]] = Field(default=None, max_length=20)

class AssessmentCreate(BaseModel):
    # A completed standardized screening (PHQ-9 depression or GAD-7 anxiety).
    # The client scores it too, but we recompute server-side from `answers` so a
    # tampered/mismatched total can't land bad clinical data. Identity comes
    # from the auth token, never the body.
    assessment_type: Literal["phq9", "gad7"]
    # One 0–3 response per item (PHQ-9 = 9 items, GAD-7 = 7). Validated by length
    # + range in the endpoint against the expected item count for the type.
    answers: List[int] = Field(min_length=1, max_length=27)
    session_id: Optional[str] = Field(default="onboarding", max_length=128)

class AuthRequest(BaseModel):
    email: str
    password: str
    name: str = ""
    role: str = "user"

class RegisterRequest(BaseModel):
    email: str
    password: str
    name: str
    role: str
    admin_code: Optional[str] = ""

# 2. Model for Login (Only needs 2 fields)
class LoginRequest(BaseModel):
    email: str
    password: str

class LessonComplete(BaseModel):
    # email is ignored server-side (identity comes from the token).
    email: Optional[str] = None
    lesson_id: int
    exercise_data: Optional[str] = ""
    blueprint_data: Optional[str] = ""

class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)

class ChatRequest(BaseModel):
    # Input caps bound prompt size and cost, and reject empty/oversized payloads.
    message: str = Field(min_length=1, max_length=4000)
    context: Optional[str] = Field(default=None, max_length=8000)
    session_id: Optional[str] = Field(default="anonymous_session", max_length=128)
    email: Optional[str] = Field(default="unknown_user", max_length=254)
    # Prior turns of the conversation, oldest first. Lets the model maintain
    # context across messages instead of treating each turn as standalone.
    # Capped here so a client can't send an unbounded transcript.
    history: Optional[List[ChatTurn]] = Field(default=None, max_length=50)

    @field_validator("message")
    @classmethod
    def _strip_message(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("message must not be empty")
        return v

class LessonAnalyze(BaseModel):
    # The user's typed Practice-step answers (human-readable, already serialized
    # by the client) plus light lesson framing. We stream back a short reflection
    # the user reads before writing their Commit-step takeaway.
    lesson_id: int
    skill: Optional[str] = Field(default="", max_length=200)
    title: Optional[str] = Field(default="", max_length=300)
    answers: str = Field(min_length=1, max_length=8000)
    # Internal coaching context for this lesson (data/curriculum.js ai_prompt_context
    # plus the lesson goal and what the exercise asked). Never shown to the user;
    # steers the model's framing so the reflection isn't generic/nonsensical.
    context: Optional[str] = Field(default=None, max_length=8000)
    # "reflect" → short reflection on the Practice-step answers (default).
    # "takeaway" → a warm response to the commitment the user wrote on the Commit step.
    mode: Optional[str] = Field(default="reflect", max_length=20)

    @field_validator("answers")
    @classmethod
    def _strip_answers(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("answers must not be empty")
        return v

class ToolEvent(BaseModel):
    # One use of an SOS/"Reset" coping tool (breathing, grounding, urge surfing,
    # TIPP). Best-effort telemetry: powers outcome data for the clinician portal
    # and future user insights. Identity comes from the auth token, never here.
    # pre_/post_distress are the optional SUDS 0–10 self-ratings; their delta is
    # the outcome signal (how much distress the tool took off).
    tool_id: str = Field(min_length=1, max_length=50)
    session_id: Optional[str] = Field(default="anonymous_session", max_length=128)
    duration_sec: Optional[int] = Field(default=None, ge=0, le=7200)
    completed: Optional[bool] = False
    pre_distress: Optional[int] = Field(default=None, ge=0, le=10)
    post_distress: Optional[int] = Field(default=None, ge=0, le=10)

# Cap how many prior turns we replay to the model, to bound prompt size/cost.
MAX_HISTORY_TURNS = int(os.getenv("CHAT_MAX_HISTORY_TURNS", "12"))

# Allow your frontend to talk to this backend.
#   - Web (Vercel): the *.vercel.app regex + localhost:3000 for `next dev`.
#   - Native (Capacitor iOS/Android): the app's WebView loads from a local
#     origin and calls this backend directly (no Next.js proxy on device), so
#     those origins must be allowed too — iOS is capacitor://localhost, Android
#     (default androidScheme=https) is https://localhost. http://localhost
#     covers `cap run` / older schemes. See frontend lib/api.js.
_NATIVE_ORIGINS = [
    "capacitor://localhost",
    "https://localhost",
    "http://localhost",
]
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex = r"https://.*\.vercel\.app",
    allow_origins=[
        "http://localhost:3000",
        *_NATIVE_ORIGINS,
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# RATE LIMITING
# ---------------------------------------------------------
_chat_limiter = SlidingWindowLimiter(
    max_events=int(os.getenv("CHAT_RATE_LIMIT", "20")),
    window_sec=float(os.getenv("CHAT_RATE_WINDOW_SEC", "60")),
)


def _client_ip(request: Request) -> str:
    """Best-effort client IP, honoring the proxy's X-Forwarded-For when present."""
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def rate_limit_chat(request: Request):
    """Throttle chat requests per client IP to guard against abuse and cost."""
    if not _chat_limiter.allow(_client_ip(request)):
        raise HTTPException(
            status_code=429,
            detail="Too many messages in a short time. Please wait a moment and try again.",
        )


# ---------------------------------------------------------
# AUTH DEPENDENCY
# ---------------------------------------------------------
def require_user(authorization: Optional[str] = Header(default=None)):
    """FastAPI dependency: require any valid (non-expired) bearer token.

    Returns the decoded payload, whose `sub` is the authenticated email. Routes
    should trust this identity rather than any email in the request body.
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Authentication required.")
    payload = decode_access_token(authorization.split(" ", 1)[1].strip())
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired session.")
    return payload


def require_admin(payload: dict = Depends(require_user)):
    """FastAPI dependency: require a valid bearer token with the 'admin' role."""
    if "admin" not in (payload.get("roles") or []):
        raise HTTPException(status_code=403, detail="Admin access required.")
    return payload


# ---------------------------------------------------------
# GEMINI (google-genai SDK on Vertex AI)
# ---------------------------------------------------------
# Chat runs on Gemini 2.5 Pro via Vertex AI — billed through GCP and authenticated
# with Application Default Credentials (the Cloud Run service account; no API key).
# Same SDK/client pattern as embeddings.py. Region us-central1 (where Cloud SQL and
# the embedding model also live). Override with GEMINI_MODEL / GEMINI_REGION.
#
# ⚠️ Gemini 2.5 Pro is a THINKING model: hidden reasoning tokens count against
# max_output_tokens, and Pro (unlike Flash) CANNOT disable thinking
# (thinking_budget=0 → 400). A small cap gets fully consumed by thinking and the
# reply comes back EMPTY. So every config below BOUNDS thinking_budget and sizes
# max_output_tokens = budget + generous visible headroom. See the GenerateContentConfig
# objects built just after SYSTEM_PROMPT.
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "bgita-teacher")
GEMINI_REGION = os.getenv("GEMINI_REGION", "us-central1")
MODEL_ID = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")

# Main chat: rich multi-paragraph CBT replies + room for Pro's reasoning.
CHAT_MAX_TOKENS = int(os.getenv("GEMINI_MAX_TOKENS", "4096"))
CHAT_THINKING_BUDGET = int(os.getenv("GEMINI_THINKING_BUDGET", "1024"))

# Log the resolved AI config at startup so a bad model id / region is obvious the
# moment the container boots — no need to wait for a failed chat to find out.
print(
    f"🔧 AI config: model={MODEL_ID} region={GEMINI_REGION} "
    f"project={PROJECT_ID} max_tokens={CHAT_MAX_TOKENS} thinking_budget={CHAT_THINKING_BUDGET}"
)

# Vertex-backed google-genai client (reads ADC for auth — no API key). Project and
# region live on the client; the system prompt + token caps are passed per-call via
# a GenerateContentConfig.
try:
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=GEMINI_REGION)
    print("✅ Gemini (Vertex AI) client initialized.")
except Exception as e:
    _log_exc("GEMINI VERTEX CLIENT INIT ERROR", e)
    client = None

# System prompt: defines the assistant's role and hard rules. This is passed to
# Gemini as `system_instruction` (not interleaved with the user message), so the
# model treats it as standing instructions rather than user-provided text.
SYSTEM_PROMPT = """You are an empathetic, highly skilled Cognitive Behavioral Therapy (CBT) therapist.
You draw upon time-tested psychological frameworks from ancient wisdom traditions, but you MUST present them using modern, accessible, secular western terminology. You MUST NOT name, quote, cite, or allude to any specific scripture, religious or spiritual text, tradition, deity, teacher, or their characters — present every idea as a plain, secular psychological principle.

SAFETY (HIGHEST PRIORITY, overrides all formatting rules below):
- If the user expresses any intent or desire to harm themselves or others, or to end their life, your FIRST priority is their safety, not therapy techniques.
- Respond with warmth and without judgment, gently encourage them to reach out for immediate human help, and surface crisis resources: in the US, call or text 988 (Suicide & Crisis Lifeline) or call 911 in immediate danger; elsewhere, direct them to local emergency services or https://findahelpline.com.
- Never minimize, argue with, or shame these feelings, and never provide instructions that could facilitate self-harm.

Guidelines for your response Content:
1. Translate ancient concepts into universal psychological principles.
2. Never use Sanskrit terms, character names, religious references, or culturally specific metaphors — even if the user asks about a source, redirect to the plain secular principle without naming it.
3. Validate the user's feelings first using standard CBT empathy.
4. You may be given text retrieved from our clinical database. IF it is relevant, paraphrase its insight into plain secular language — never quote it verbatim and never carry over any names, places, or archaic wording from it.
5. IF the database text is not relevant, DO NOT mention the database.
6. Never say "According to the database".
7. Offer the user the ability to ask for the response in a simpler format.
8. Use CBT-adjacent questions to engage users further.
9. When a user asks a question, ground your answer in concrete, practical examples — without citing any source text, tradition, or historical or religious figures.

Guidelines for your response Formatting (CRITICAL):
- NEVER output a single wall of text. Break your responses into short, easily digestible paragraphs (maximum 2-3 sentences per paragraph).
- Use Markdown formatting to make your response visually structured and scannable.
- Use bullet points or numbered lists when explaining multiple concepts, actionable steps, or cognitive reframing exercises.
- Use **bold text** to gently emphasize key psychological terms or core takeaways."""


# Lesson reflections reuse the main voice but with a tighter mandate. The main
# SYSTEM_PROMPT's guideline #9 ("provide historical background with specific
# examples") pulls against the secular no-naming rule and could make the guide
# name the source story mid-reflection — bad here, where the whole point is to
# stay inside the user's own example. This override neutralizes that and pins the
# format to short plain paragraphs (a short bulleted chain is fine when mirroring
# their answers back as the lesson's framework).
ANALYSIS_SYSTEM_PROMPT = SYSTEM_PROMPT + (
    "\n\nLESSON-REFLECTION OVERRIDE (takes precedence over the content guidelines "
    "above): You are reflecting on one person's exercise answers, not answering a "
    "question. Do NOT provide historical background, and never name or allude to any "
    "source text, story, teacher, or characters. Stay entirely within the user's own "
    "words and plain modern language. No headings; keep to short paragraphs, with at "
    "most one short bulleted chain when you mirror their answers back."
)

REFLECT_MAX_TOKENS = int(os.getenv("REFLECT_MAX_TOKENS", "1536"))
TAKEAWAY_MAX_TOKENS = int(os.getenv("ANALYSIS_MAX_TOKENS", "1536"))
LESSON_THINKING_BUDGET = int(os.getenv("LESSON_THINKING_BUDGET", "512"))

CHAT_CONFIG = types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,
    max_output_tokens=CHAT_MAX_TOKENS,
    thinking_config=types.ThinkingConfig(thinking_budget=CHAT_THINKING_BUDGET),
)
REFLECT_CONFIG = types.GenerateContentConfig(
    system_instruction=ANALYSIS_SYSTEM_PROMPT,
    max_output_tokens=REFLECT_MAX_TOKENS,
    thinking_config=types.ThinkingConfig(thinking_budget=LESSON_THINKING_BUDGET),
)
TAKEAWAY_CONFIG = types.GenerateContentConfig(
    system_instruction=ANALYSIS_SYSTEM_PROMPT,
    max_output_tokens=TAKEAWAY_MAX_TOKENS,
    thinking_config=types.ThinkingConfig(thinking_budget=LESSON_THINKING_BUDGET),
)



# ---------------------------------------------------------
# THE SEMANTIC SEARCH ENGINE (RAG)
# ---------------------------------------------------------
TOP_K = int(os.getenv("RAG_TOP_K", "3"))
# Pull a wider candidate pool by vector distance, then rerank down to TOP_K.
RAG_CANDIDATES = max(int(os.getenv("RAG_CANDIDATES", "10")), TOP_K)
# Extra candidates from lexical (keyword) search, merged with the vector pool.
RAG_KEYWORD_CANDIDATES = int(os.getenv("RAG_KEYWORD_CANDIDATES", "10"))
# MMR diversity tradeoff: 1.0 = pure relevance, lower = more diverse passages.
RAG_MMR_LAMBDA = float(os.getenv("RAG_MMR_LAMBDA", "0.7"))


def _keyword_search(conn, user_message: str, limit: int):
    """Lexical full-text search over the corpus (best-effort).

    Complements vector search: catches exact terms/proper nouns the embedding
    might miss ("Arjuna", "dharma", a specific verse). Reuses the caller's
    connection (the vector query already ran on it, so a failure here can't
    disturb those results), and returns [] on any error (e.g. an empty/
    stopword-only query) so the caller degrades to vector-only retrieval.
    """
    if limit <= 0:
        return []
    try:
        sql = text("""
            SELECT content, embedding
            FROM gita_chunks
            WHERE to_tsvector('english', content) @@ plainto_tsquery('english', :q)
            LIMIT :k
        """)
        return conn.execute(sql, {"q": user_message, "k": limit}).fetchall()
    except Exception as e:
        _log_exc("KEYWORD SEARCH FAILED (non-fatal)", e)
        return []


def get_clinical_context(user_message: str) -> str:
    """Hybrid (vector + keyword) search over the Gita corpus, with MMR reranking.

    Pulls candidates two ways — RAG_CANDIDATES nearest by pgvector cosine
    distance (semantic) plus RAG_KEYWORD_CANDIDATES by Postgres full-text match
    (lexical) — merges and de-duplicates them, then reranks with Maximal Marginal
    Relevance to pick TOP_K that are relevant and non-redundant. Hybrid retrieval
    catches both paraphrased intent (vector) and exact terms/names (keyword).
    Vectors are loaded by ingest_corpus.py.
    """
    try:
        qvec = embed_query(user_message)
        qvec_literal = to_pgvector_literal(qvec)

        with engine.connect() as conn:
            # `<=>` is pgvector's cosine-distance operator (smaller = closer).
            query = text("""
                SELECT content, embedding
                FROM gita_chunks
                ORDER BY embedding <=> CAST(:qvec AS vector)
                LIMIT :n
            """)
            vector_rows = conn.execute(query, {"qvec": qvec_literal, "n": RAG_CANDIDATES}).fetchall()
            # Reuse the same connection for the keyword pass (saves a second
            # acquire per turn). Safe because the vector rows are already fetched.
            keyword_rows = _keyword_search(conn, user_message, RAG_KEYWORD_CANDIDATES)

        # Merge both pools, de-duplicating on content. Vector hits come first so
        # they win ties and anchor the fallback order if reranking can't run.
        candidates, seen = [], set()
        for row in list(vector_rows) + list(keyword_rows):
            content = row[0]
            if content in seen:
                continue
            seen.add(content)
            candidates.append({"content": content, "embedding": parse_pgvector(row[1])})

        if not candidates:
            return "No relevant passages found in the clinical database for this topic."

        selected = mmr_select(qvec, candidates, k=TOP_K, lambda_mult=RAG_MMR_LAMBDA)

        db_text = "Here is the most relevant text retrieved from our clinical_db:\n"
        for content in selected:
            db_text += f"- {content}\n"
        return db_text

    except Exception as e:
        _log_exc("SEMANTIC SEARCH FAILED", e)
        return "Warning: Could not retrieve from the clinical database."

# ---------------------------------------------------------
# THE NEW CHAT ENDPOINT
# ---------------------------------------------------------
def _ensure_metrics_table():
    """Create the AI telemetry table once at startup (idempotent).

    Previously this DDL ran inside _log_chat_metrics on *every* chat request — a
    needless round-trip per message. Hoisted here so it runs once at boot.
    """
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS ai_metrics_log (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(255),
                    email VARCHAR(255),
                    prompt_time_sec FLOAT,
                    input_tokens INT,
                    output_tokens INT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
        print("✅ ai_metrics_log table ensured.")
    except Exception as e:
        _log_exc("METRICS TABLE INIT FAILED (non-fatal)", e)


def _ensure_lesson_progress_table():
    """Create the lesson-progress table once at startup (idempotent).

    /api/lesson/complete inserts here and /api/admin/metrics reads it, but the
    table was never created — so completions silently 500'd. Schema matches both
    the full and minimal inserts in complete_lesson() and the admin read.
    """
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS lesson_progress (
                    id SERIAL PRIMARY KEY,
                    email VARCHAR(255),
                    lesson_id INTEGER,
                    exercise_data TEXT,
                    blueprint_data TEXT,
                    completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
        print("✅ lesson_progress table ensured.")
    except Exception as e:
        _log_exc("LESSON_PROGRESS TABLE INIT FAILED (non-fatal)", e)


def _ensure_crisis_events_table():
    """Create the crisis-events table once at startup (idempotent).

    The chat endpoint's safety layer (detect_crisis) only printed to logs, so
    at-risk sessions were never persisted. This table captures them so the admin
    (clinician) portal can surface flagged sessions for triage.
    """
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS crisis_events (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(255),
                    email VARCHAR(255),
                    message_excerpt TEXT,
                    category VARCHAR(32) DEFAULT 'self_harm',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            # Idempotent upgrade for tables created before the harm-to-others
            # screen existed: add the category column if it's missing.
            conn.execute(text(
                "ALTER TABLE crisis_events "
                "ADD COLUMN IF NOT EXISTS category VARCHAR(32) DEFAULT 'self_harm'"
            ))
        print("✅ crisis_events table ensured.")
    except Exception as e:
        _log_exc("CRISIS_EVENTS TABLE INIT FAILED (non-fatal)", e)


def _ensure_tool_events_table():
    """Create the SOS/coping-tool telemetry table once at startup (idempotent).

    Records each use of a Reset-toolkit tool plus the optional pre/post SUDS
    (0–10) self-ratings. The pre−post delta is the outcome signal surfaced in the
    clinician portal ("breathing takes ~N points off distress on average").
    """
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS tool_events (
                    id SERIAL PRIMARY KEY,
                    email VARCHAR(255),
                    tool_id VARCHAR(50),
                    session_id VARCHAR(128),
                    duration_sec INTEGER,
                    completed BOOLEAN DEFAULT FALSE,
                    pre_distress INTEGER,
                    post_distress INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
        print("✅ tool_events table ensured.")
    except Exception as e:
        _log_exc("TOOL_EVENTS TABLE INIT FAILED (non-fatal)", e)


def _ensure_logs_columns():
    """Add the rich mood-logging columns to the existing `logs` table (idempotent).

    The daily check-in used to be just mood (1–5) + a free-text reflection. Rich
    logging adds emotions/energy/sleep/activity context. We ALTER rather than
    recreate so historical rows are untouched (they read back as NULL) and the
    admin portal — which reads `mood`/`reflection` — is unaffected. `emotions`
    and `activities` hold delimiter-joined controlled-vocabulary labels.
    """
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                ALTER TABLE logs
                    ADD COLUMN IF NOT EXISTS emotions TEXT,
                    ADD COLUMN IF NOT EXISTS energy INTEGER,
                    ADD COLUMN IF NOT EXISTS sleep VARCHAR(50),
                    ADD COLUMN IF NOT EXISTS activities TEXT
            """))
        print("✅ logs rich-mood columns ensured.")
    except Exception as e:
        _log_exc("LOGS COLUMNS INIT FAILED (non-fatal)", e)


def _ensure_assessments_table():
    """Create the standardized-screening table once at startup (idempotent).

    Stores each completed PHQ-9 (depression) / GAD-7 (anxiety) screening: the
    per-item answers, the recomputed total score, and the severity band. The
    first pair is the onboarding baseline; later ones power the symptom-trend
    view. Score/severity are derived server-side in the endpoint.
    """
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS assessments (
                    id SERIAL PRIMARY KEY,
                    email VARCHAR(255),
                    assessment_type VARCHAR(20),
                    score INTEGER,
                    severity VARCHAR(40),
                    answers TEXT,
                    session_id VARCHAR(128),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
        print("✅ assessments table ensured.")
    except Exception as e:
        _log_exc("ASSESSMENTS TABLE INIT FAILED (non-fatal)", e)


def _verify_embedding_model():
    """Warn loudly at boot if the serving embedding model differs from the one
    that built the corpus.

    A mismatch is silent and catastrophic for RAG: query vectors and stored
    vectors then live in different embedding spaces, so cosine search returns
    effectively random passages. ingest_corpus.py stamps the model it used into
    gita_corpus_meta; here we compare it to the model this process will embed
    queries with. If the meta row is missing (corpus predates the stamp), we say
    so and point at a re-ingest rather than guessing.
    """
    try:
        from embeddings import EMBED_MODEL
        with engine.connect() as conn:
            # Check existence first so the expected pre-ingest state (table absent
            # because ingest_corpus.py hasn't stamped it yet) yields a clean
            # warning instead of a DatabaseError + traceback.
            exists = conn.execute(text("SELECT to_regclass('public.gita_corpus_meta')")).scalar()
            row = conn.execute(text(
                "SELECT embed_model, dim FROM gita_corpus_meta ORDER BY ingested_at DESC LIMIT 1"
            )).fetchone() if exists else None
        if not row:
            print(
                f"⚠️ Corpus embedding model unknown (gita_corpus_meta empty/absent). "
                f"Serving queries with '{EMBED_MODEL}'. Re-run ingest_corpus.py to stamp the corpus."
            )
            return
        stored_model, stored_dim = row[0], row[1]
        if stored_model != EMBED_MODEL:
            print(
                f"🚨 EMBEDDING MODEL MISMATCH: corpus built with '{stored_model}' but queries "
                f"embed with '{EMBED_MODEL}'. RAG retrieval is degraded until these match — "
                f"re-ingest, or set VERTEX_EMBED_MODEL='{stored_model}'."
            )
        else:
            print(f"✅ Embedding model matches corpus: {EMBED_MODEL} (dim {stored_dim}).")
    except Exception as e:
        _log_exc("EMBED MODEL VERIFY FAILED (non-fatal)", e)


_ensure_metrics_table()
_ensure_lesson_progress_table()
_ensure_crisis_events_table()
_ensure_tool_events_table()
_ensure_logs_columns()
_ensure_assessments_table()
_verify_embedding_model()

# Make admin-signup availability obvious in the boot logs. If ADMIN_SIGNUP_CODE is
# unset in this environment (e.g. not configured on Cloud Run), EVERY signup —
# even one that submits an admin code — is created as a plain "user", which is the
# usual cause of a new "admin" account hitting "Access Denied". This one line tells
# you from the logs whether admin signup is even possible here.
print(
    "🔐 Admin signup ENABLED (a matching ADMIN_SIGNUP_CODE grants admin)."
    if ADMIN_SIGNUP_CODE else
    "🔐 Admin signup DISABLED — ADMIN_SIGNUP_CODE is unset, so all signups become regular users."
)


def _log_crisis_event(request: ChatRequest, category: str = "self_harm"):
    """Persist a detected crisis event. Best-effort: never raises into the request path.

    category distinguishes the self-harm screen ('self_harm') from the
    harm-to-others screen ('harm_to_others') so the clinician portal can triage
    them differently.
    """
    try:
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO crisis_events (session_id, email, message_excerpt, category)
                    VALUES (:session_id, :email, :excerpt, :category)
                """),
                {
                    "session_id": request.session_id,
                    "email": request.email,
                    "excerpt": (request.message or "")[:280],
                    "category": category,
                },
            )
    except Exception as e:
        _log_exc("CRISIS EVENT LOG FAILED (non-fatal)", e)


def _log_chat_metrics(request: ChatRequest, prompt_time: float,
                      input_tokens: int, output_tokens: int):
    """Persist AI telemetry. Best-effort: never raises into the request path.

    The table is created once at startup by _ensure_metrics_table(), so the hot
    path here is just the INSERT.
    """
    try:
        with engine.begin() as conn:
            # Insert the stats
            conn.execute(text("""
                INSERT INTO ai_metrics_log (session_id, email, prompt_time_sec, input_tokens, output_tokens)
                VALUES (:session_id, :email, :prompt_time, :input_tokens, :output_tokens)
            """), {
                "session_id": request.session_id,
                "email": request.email,
                "prompt_time": prompt_time,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            })
    except Exception as db_err:
        _log_exc("METRICS LOG FAILED (non-fatal)", db_err)


# Acknowledgment / continuation phrases that carry no retrievable content. On
# these turns RAG (an embedding API call + 2 DB queries + injected context
# tokens) is pure waste, so we skip it and let the conversation history carry
# the thread.
_LOW_SIGNAL_PHRASES = {
    "yes", "no", "ok", "okay", "k", "kk", "sure", "yeah", "yep", "yup", "nope",
    "nah", "thanks", "thank you", "thx", "ty", "cool", "nice", "great", "good",
    "right", "true", "exactly", "agreed", "i agree", "fine", "alright",
    "all right", "got it", "i see", "makes sense", "that makes sense",
    "sounds good", "tell me more", "go on", "continue", "more",
    "please continue", "ok thanks", "okay thanks", "thank you so much",
    "perfect", "awesome", "hmm", "oh", "ah", "wow", "i understand", "understood",
}

# Filler/stopword tokens; an ultra-short message made only of these is low-signal.
_FILLER_WORDS = {
    "a", "an", "the", "i", "you", "it", "that", "this", "so", "well", "and",
    "but", "ok", "okay", "yes", "no", "yeah", "yep", "nope", "sure", "thanks",
    "thank", "please", "more", "continue", "go", "on", "right", "true", "cool",
    "nice", "great", "good", "fine", "hmm", "oh", "ah", "wow", "really", "very",
    "just", "too", "to", "is", "am", "are", "do",
}


def _should_retrieve(message: str) -> bool:
    """Decide whether a turn is substantive enough to warrant RAG retrieval.

    Skips clear acknowledgments / continuations ("yes", "tell me more",
    "ok thanks") and filler-only messages, where embedding + DB lookups would
    only burn latency, an embedding API call, and prompt tokens on irrelevant
    passages. Conversation history still carries context, so the model isn't
    left blind. Conservative by design: anything with real content (including
    short questions like "why?") still retrieves.
    """
    cleaned = re.sub(r"[^a-z\s]", "", message.lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return False  # pure punctuation / emoji
    if cleaned in _LOW_SIGNAL_PHRASES:
        return False
    words = cleaned.split()
    if len(words) <= 3 and all(w in _FILLER_WORDS for w in words):
        return False
    return True


def _build_messages(request: ChatRequest) -> list:
    """Assemble the Gemini `contents` list: prior turns + the current turn.

    The RAG/lesson context is attached only to the *current* user message so it
    doesn't bloat every historical turn. Gemini uses the roles "user" and "model"
    (assistant → "model") and expects the conversation to start with a user turn,
    so any leading model turns (e.g. the UI's greeting) are dropped.
    """
    # Gate RAG on low-signal turns: skip the embedding call + DB lookups (and the
    # injected context tokens) for acknowledgments/continuations like "yes" or
    # "tell me more", where retrieval adds cost but no value.
    retrieve = _should_retrieve(request.message)
    print(f"🔎 RAG retrieval: {'on' if retrieve else 'skipped (low-signal turn)'}")
    clinical_data = get_clinical_context(request.message) if retrieve else ""

    lesson_instructions = ""
    if request.context:
        lesson_instructions = f"\n[CURRENT LESSON CONTEXT]\n{request.context}\nFocus your entire response on guiding the user through this specific lesson and do not change the subject.\n"

    # Only attach the database block when we actually retrieved something, so
    # skipped turns don't carry an empty (or token-wasting) context section.
    db_block = ""
    if clinical_data:
        db_block = f"""
        [DATABASE CONTEXT START]
        {clinical_data}
        [DATABASE CONTEXT END]
"""

    user_content = f"""{lesson_instructions}{db_block}
        User Message:
        {request.message}"""

    messages = []
    for turn in (request.history or [])[-MAX_HISTORY_TURNS:]:
        role = "model" if turn.role == "assistant" else "user"
        messages.append(types.Content(role=role, parts=[types.Part.from_text(text=turn.content)]))
    # Drop leading model turn(s) so the conversation starts with the user.
    while messages and messages[0].role != "user":
        messages.pop(0)
    messages.append(types.Content(role="user", parts=[types.Part.from_text(text=user_content)]))
    return messages


@app.post("/api/chat")
def chat_with_gita(request: ChatRequest, user: dict = Depends(require_user),
                   _rl: None = Depends(rate_limit_chat)):
    # Trust the authenticated identity from the token, not the client-supplied
    # body email — so telemetry/crisis logs can't be spoofed and anonymous
    # callers can't run up the (paid) model. session_id stays client-supplied.
    request.email = user["sub"]
    print(
        f"💬 /api/chat session={request.session_id} email={request.email} "
        f"msg_len={len(request.message)} history_turns={len(request.history or [])} "
        f"has_lesson_ctx={bool(request.context)}"
    )
    # Safety first: if the user expresses self-harm/suicidal intent, bypass the
    # model and respond with crisis resources directly. Streamed as text/plain so
    # the frontend's streaming reader handles it identically to a normal reply.
    if detect_crisis(request.message):
        print(f"⚠️ Crisis language detected for session={request.session_id}")
        # Persist the flagged session so the clinician portal can surface it.
        # Done before returning so it isn't tied to the stream being consumed.
        _log_crisis_event(request)

        def crisis_stream():
            yield CRISIS_RESPONSE
            _log_chat_metrics(request, 0.0, 0, 0)

        return StreamingResponse(crisis_stream(), media_type="text/plain")

    # Second safety screen: explicit intent to seriously harm OTHERS. Separate,
    # higher-precision detector (see safety.py) so trauma survivors, Harm-OCD
    # intrusive thoughts, and everyday hyperbole are NOT misflagged. Surfaces the
    # session to the clinician portal; it does NOT auto-report to anyone.
    if detect_harm_to_others(request.message):
        print(f"⚠️ Harm-to-others language detected for session={request.session_id}")
        _log_crisis_event(request, category="harm_to_others")

        def harm_others_stream():
            yield HARM_TO_OTHERS_RESPONSE
            _log_chat_metrics(request, 0.0, 0, 0)

        return StreamingResponse(harm_others_stream(), media_type="text/plain")

    if client is None:
        print("🚨 /api/chat called but Gemini (Vertex) client is None (init failed — check ADC / project / region).")
        raise HTTPException(status_code=503, detail="AI engine is unavailable.")

    try:
        messages = _build_messages(request)
    except Exception as e:
        # _build_messages calls get_clinical_context (DB/embeddings) — if that
        # blows up before streaming starts, surface it instead of a blank 500.
        _log_exc("CHAT BUILD-MESSAGES ERROR", e)
        raise HTTPException(status_code=500, detail="Could not prepare your message.") from e

    def stream_reply():
        start_time = time.time()
        input_tokens = output_tokens = 0
        try:
            print(f"🤖 Calling Gemini model={MODEL_ID} with {len(messages)} messages…")
            for chunk in client.models.generate_content_stream(
                model=MODEL_ID,
                contents=messages,
                config=CHAT_CONFIG,
            ):
                if chunk.text:
                    yield chunk.text
                # Usage arrives on chunks (last one is authoritative); some
                # intermediate chunks report None, so guard every read.
                usage = getattr(chunk, "usage_metadata", None)
                if usage:
                    if usage.prompt_token_count:
                        input_tokens = usage.prompt_token_count
                    if usage.candidates_token_count:
                        output_tokens = usage.candidates_token_count
            print(f"✅ Gemini reply OK in={input_tokens} out={output_tokens} tokens")
        except Exception as e:
            # Surface type + traceback, and flag the usual suspects so the next
            # failure is diagnosable straight from the log line.
            _log_exc("AI STREAMING ERROR", e)
            status = getattr(e, "code", None) or getattr(e, "status_code", None)
            print(
                f"   ↳ context: status={status} model={MODEL_ID} "
                f"region={GEMINI_REGION} project={PROJECT_ID}"
            )
            if status in (401, 403):
                print("   ↳ HINT: auth/permission — the Cloud Run service account may lack roles/aiplatform.user, "
                      "or the Vertex AI API isn't enabled for this project.")
            elif status == 404:
                print("   ↳ HINT: 404 usually means the model id isn't valid/available in this region. "
                      "Check GEMINI_MODEL and GEMINI_REGION.")
            elif status == 429:
                print("   ↳ HINT: Vertex quota/rate limit for Gemini in this region. Request a quota increase or back off.")
            yield "\n\n_Sorry — I lost my train of thought there. Could you say that again?_"

        prompt_time = round(time.time() - start_time, 2)
        _log_chat_metrics(request, prompt_time, input_tokens, output_tokens)

    return StreamingResponse(stream_reply(), media_type="text/plain")



@app.post("/api/register")
def register_user(request: RegisterRequest):
    try:
        # engine.begin() automatically commits the data to the database!
        with engine.begin() as conn:
            # Salted bcrypt hash — never store the plain or unsalted password.
            hashed_pw = hash_password(request.password)

            # Admin role requires the env-configured signup code (blank => denied).
            assigned_role = "admin" if ADMIN_SIGNUP_CODE and request.admin_code == ADMIN_SIGNUP_CODE else "user"

            # Notice the ARRAY[:role] down below! This fixes the Postgres error.
            query = text("""
                INSERT INTO users (username, email, password, name, roles)
                VALUES (:email, :email, :password, :name, ARRAY[:role])
            """)

            conn.execute(query, {
                "email": request.email,
                "password": hashed_pw,
                "name": request.name,
                "role": assigned_role
            })
            # Return the ACTUAL assigned role so the client can tell the user when
            # an entered admin code wasn't accepted (root cause of the "I made an
            # admin account but get Access Denied" confusion): a wrong/blank code
            # silently downgrades to "user", and the UI can now say so at signup.
            return {
                "status": "success",
                "role": assigned_role,
                "message": f"User registered successfully as {assigned_role}!",
            }
            
    except HTTPException:
        raise
    except Exception as e:
        # Log the full error server-side only — never leak DB internals (SQL,
        # schema, password hash) to the client.
        _log_exc("REGISTER ERROR", e)
        msg = str(e).lower()
        if "duplicate key" in msg or "23505" in msg or "unique constraint" in msg:
            raise HTTPException(status_code=409, detail="An account with that email already exists.") from e
        # Generic failure: still a true HTTP error so the frontend won't fake success.
        raise HTTPException(status_code=500, detail="Could not create the account. Please try again.") from e


@app.delete("/api/account")
def delete_account(user: dict = Depends(require_user), db: Session = Depends(get_db)):
    """Permanently delete the authenticated user's account and all their data.

    Required for the Google Play / App Store account-deletion policy and to honor
    the deletion right promised in the privacy policy. Identity comes from the
    token (`sub` = email), never the request body, so a user can only delete
    their own account.

    All personal data lives keyed by the user's email. Note the `logs` table
    stores the email in its `username` column (historical naming); every other
    table uses an `email` column. Deletion is atomic: engine.begin() commits only
    if every statement succeeds, so we never leave a half-deleted account.
    """
    email = user["sub"]

    # (table, user-key column). Ordered children-first, users last, though the
    # single transaction makes ordering immaterial here.
    targets = [
        ("logs", "username"),
        ("assessments", "email"),
        ("lesson_progress", "email"),
        ("tool_events", "email"),
        ("crisis_events", "email"),
        ("ai_metrics_log", "email"),
        ("users", "email"),
    ]

    try:
        with engine.begin() as conn:
            for table, col in targets:
                # Table/column names are hardcoded above (never user input), so
                # interpolating them into the SQL is safe; the value is bound.
                conn.execute(
                    text(f"DELETE FROM {table} WHERE {col} = :email"),
                    {"email": email},
                )
        return {"status": "success", "message": "Your account and all associated data have been deleted."}
    except Exception as e:
        _log_exc("ACCOUNT DELETION FAILED", e)
        raise HTTPException(status_code=500, detail="Could not delete your account. Please try again.") from e


@app.post("/api/login")
def login_user(request: LoginRequest):
    try:
        with engine.begin() as conn:
            # Fetch the account by email, then verify the password in Python so we
            # can support both bcrypt and legacy SHA-256 hashes.
            query = text("""
                SELECT name, roles, email, password
                FROM users
                WHERE email = :email
            """)
            result = conn.execute(query, {"email": request.email}).fetchone()

            if not result or not verify_password(request.password, result[3]):
                # Same message for "no such user" and "wrong password" — don't
                # reveal which emails are registered.
                raise HTTPException(status_code=401, detail="Invalid email or password.")

            # Transparently upgrade weak/legacy hashes on successful login.
            if needs_rehash(result[3]):
                conn.execute(
                    text("UPDATE users SET password = :pw WHERE email = :email"),
                    {"pw": hash_password(request.password), "email": request.email},
                )

            return {
                "status": "success",
                "access_token": create_access_token(result[2], result[1]),
                "user": {"name": result[0], "role": result[1], "email": result[2]}
            }

    except HTTPException:
        # Let intended HTTP errors (e.g. 401) propagate as-is.
        raise
    except Exception as e:
        # Log full detail server-side; never echo the raw exception (DB/schema)
        # back to the client.
        _log_exc("LOGIN ERROR", e)
        raise HTTPException(status_code=500, detail="Could not log you in. Please try again.") from e


# Rich mood-logging lists (emotions/activities) are stored as delimiter-joined
# text in single columns. `|` avoids colliding with commas inside labels and is
# never part of the controlled vocabulary the client sends.
_LABEL_DELIM = " | "


def _join_labels(items) -> Optional[str]:
    """Join a client-sent label list into one column value (None if empty)."""
    if not items:
        return None
    cleaned = [str(x).strip() for x in items if str(x).strip()]
    return _LABEL_DELIM.join(cleaned) if cleaned else None


def _split_labels(value) -> list:
    """Inverse of _join_labels for reads; tolerant of NULL and legacy commas."""
    if not value:
        return []
    raw = value.split("|") if "|" in value else value.split(",")
    return [p.strip() for p in raw if p.strip()]


# Standardized-screening scoring. Each item is answered 0–3; the total maps to a
# severity band using the published clinical cutoffs. PHQ-9 (depression) has 9
# items (total 0–27), GAD-7 (anxiety) has 7 (total 0–21). The score is always
# recomputed server-side from the raw answers so a tampered client total can't
# persist bad clinical data.
_ASSESSMENT_SPEC = {
    "phq9": {"items": 9, "max": 27},
    "gad7": {"items": 7, "max": 21},
}


def _assessment_severity(atype: str, score: int) -> str:
    """Map a screening total to its severity band."""
    if atype == "phq9":
        if score <= 4:
            return "Minimal"
        if score <= 9:
            return "Mild"
        if score <= 14:
            return "Moderate"
        if score <= 19:
            return "Moderately severe"
        return "Severe"
    # gad7
    if score <= 4:
        return "Minimal"
    if score <= 9:
        return "Mild"
    if score <= 14:
        return "Moderate"
    return "Severe"


@app.post("/api/logs")
async def save_daily_log(log: LogCreate, db: Session = Depends(get_db),
                         user: dict = Depends(require_user)):
    try:
        # Trust the authenticated identity from the token, NOT the request body,
        # so a user can only write journal entries under their own account.
        username = user["sub"]

        # `mood`/`reflection` keep their original meaning (admin AVG(mood) + the
        # mood sparkline still work); the rich fields land in the columns added
        # by _ensure_logs_columns(). Lists are flattened to delimited text.
        query = text("""
            INSERT INTO logs (username, timestamp, mood, activity, reflection,
                              emotions, energy, sleep, activities)
            VALUES (:username, NOW(), :mood, :activity, :reflection,
                    :emotions, :energy, :sleep, :activities)
        """)

        db.execute(query, {
            "username": username,          # Saving their email into the 'username' column
            "mood": str(log.mood_score),   # Converting the 1-5 number into TEXT
            "activity": "Daily Journal",   # Filling the required activity column with a default label
            "reflection": log.diary_text,  # Saving the paragraph into the 'reflection' column
            "emotions": _join_labels(log.emotions),
            "energy": log.energy,
            "sleep": (log.sleep or None),
            "activities": _join_labels(log.activities),
        })
        db.commit()

        return {"status": "success", "message": "Journal entry saved to database!"}
    except Exception as e:
        db.rollback()
        # Log server-side only; return a true error status with a generic message
        # (a 200 here would make the client falsely believe the entry was saved).
        _log_exc("LOG SAVE ERROR", e)
        raise HTTPException(status_code=500, detail="Could not save your entry. Please try again.") from e


@app.post("/api/tools/log")
def log_tool_event(event: ToolEvent, db: Session = Depends(get_db),
                   user: dict = Depends(require_user)):
    """Record one use of an SOS/"Reset" coping tool (best-effort telemetry).

    Powers outcome data for the clinician portal + future user insights: which
    tools get used and, when the user rates it, how much distress dropped
    (pre − post). Identity comes from the token, never the body.

    Logging must NEVER break a coping session, so a DB failure is swallowed and
    still returns 200 — the client fires this and ignores the result.
    """
    try:
        db.execute(text("""
            INSERT INTO tool_events
                (email, tool_id, session_id, duration_sec, completed, pre_distress, post_distress, created_at)
            VALUES (:email, :tool_id, :session_id, :duration_sec, :completed, :pre_distress, :post_distress, NOW())
        """), {
            "email": user["sub"],
            "tool_id": event.tool_id,
            "session_id": event.session_id,
            "duration_sec": event.duration_sec,
            "completed": bool(event.completed),
            "pre_distress": event.pre_distress,
            "post_distress": event.post_distress,
        })
        db.commit()
        return {"status": "success"}
    except Exception as e:
        db.rollback()
        _log_exc("TOOL EVENT LOG ERROR (non-fatal)", e)
        return {"status": "skipped"}


@app.get("/api/logs")
def get_daily_logs(user: dict = Depends(require_user), db: Session = Depends(get_db)):
    """Return the authenticated user's past daily check-ins, most recent first.

    Powers the "view previous diaries" history (6/29 #1). The frontend also uses
    the most-recent timestamp to tell whether the user has already checked in
    today, so the Save Entry button only acts once per day.
    """
    try:
        rows = db.execute(text("""
            SELECT mood, reflection, timestamp, emotions, energy, sleep, activities
            FROM logs
            WHERE username = :username AND activity = 'Daily Journal'
            ORDER BY timestamp DESC
            LIMIT 90
        """), {"username": user["sub"]}).fetchall()
        return {
            "entries": [
                {
                    "mood": r[0],
                    "reflection": r[1],
                    "timestamp": str(r[2]),
                    "emotions": _split_labels(r[3]),
                    "energy": r[4],
                    "sleep": r[5],
                    "activities": _split_labels(r[6]),
                }
                for r in rows
            ]
        }
    except Exception as e:
        _log_exc("LOG FETCH ERROR", e)
        raise HTTPException(status_code=500, detail="Could not load your entries.") from e


@app.post("/api/assessment")
def save_assessment(assessment: AssessmentCreate, db: Session = Depends(get_db),
                    user: dict = Depends(require_user)):
    """Record a completed PHQ-9 / GAD-7 screening and return the scored result.

    Unlike the coping-tool telemetry this is clinical data, so a DB failure is a
    real error (not swallowed). The score + severity band are recomputed here
    from the raw per-item answers — never trusted from the client — so a tampered
    total can't land bad data. Identity comes from the token, not the body.
    """
    spec = _ASSESSMENT_SPEC[assessment.assessment_type]
    # Exactly one answer per screening item, each in the 0–3 response range.
    if len(assessment.answers) != spec["items"]:
        raise HTTPException(
            status_code=422,
            detail=f"{assessment.assessment_type} expects {spec['items']} answers.",
        )
    if any(a < 0 or a > 3 for a in assessment.answers):
        raise HTTPException(status_code=422, detail="Each answer must be between 0 and 3.")

    score = sum(assessment.answers)
    severity = _assessment_severity(assessment.assessment_type, score)

    # PHQ-9 item 9 screens for thoughts of self-harm; any non-zero answer is a
    # safety signal. Surface crisis resources to the user (via `alert`) and
    # persist a crisis event for the clinician portal — best-effort, mirroring
    # the chat crisis path so it never blocks saving the screening.
    alert = assessment.assessment_type == "phq9" and assessment.answers[8] > 0
    if alert:
        try:
            with engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO crisis_events (session_id, email, message_excerpt)
                    VALUES (:session_id, :email, :excerpt)
                """), {
                    "session_id": f"assessment-{assessment.session_id}",
                    "email": user["sub"],
                    "excerpt": f"PHQ-9 item 9 (self-harm) = {assessment.answers[8]}; total {score}.",
                })
        except Exception as e:
            _log_exc("ASSESSMENT CRISIS LOG FAILED (non-fatal)", e)

    try:
        db.execute(text("""
            INSERT INTO assessments
                (email, assessment_type, score, severity, answers, session_id, created_at)
            VALUES (:email, :atype, :score, :severity, :answers, :session_id, NOW())
        """), {
            "email": user["sub"],
            "atype": assessment.assessment_type,
            "score": score,
            "severity": severity,
            # Raw per-item answers kept as comma-joined text for later review.
            "answers": ",".join(str(a) for a in assessment.answers),
            "session_id": assessment.session_id,
        })
        db.commit()
    except Exception as e:
        db.rollback()
        _log_exc("ASSESSMENT SAVE ERROR", e)
        raise HTTPException(status_code=500, detail="Could not save your responses. Please try again.") from e

    return {
        "status": "success",
        "assessment_type": assessment.assessment_type,
        "score": score,
        "max_score": spec["max"],
        "severity": severity,
        "alert": alert,
    }


@app.get("/api/assessment")
def get_assessments(user: dict = Depends(require_user), db: Session = Depends(get_db)):
    """Return the user's past screenings (most recent first) for the trend view."""
    try:
        rows = db.execute(text("""
            SELECT assessment_type, score, severity, created_at
            FROM assessments
            WHERE email = :email
            ORDER BY created_at DESC
            LIMIT 90
        """), {"email": user["sub"]}).fetchall()
        return {
            "assessments": [
                {
                    "assessment_type": r[0],
                    "score": r[1],
                    "severity": r[2],
                    "max_score": _ASSESSMENT_SPEC.get(r[0], {}).get("max"),
                    "timestamp": str(r[3]),
                }
                for r in rows
            ]
        }
    except Exception as e:
        _log_exc("ASSESSMENT FETCH ERROR", e)
        raise HTTPException(status_code=500, detail="Could not load your assessments.") from e


@app.post("/api/lesson/complete")
def complete_lesson(request: LessonComplete, user: dict = Depends(require_user)):
    """Record that the authenticated user completed a lesson.

    The minimal schema is lesson_progress(email, lesson_id, completed_at). If the
    table also has exercise_data/blueprint_data columns we persist those too;
    otherwise we fall back to the minimal insert so a richer client still works
    against the older schema.
    """
    email = user["sub"]  # trust the token, not request.email

    try:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO lesson_progress (email, lesson_id, exercise_data, blueprint_data, completed_at)
                VALUES (:email, :lesson_id, :exercise_data, :blueprint_data, NOW())
            """), {
                "email": email,
                "lesson_id": request.lesson_id,
                "exercise_data": request.exercise_data,
                "blueprint_data": request.blueprint_data,
            })
        return {"status": "success", "message": "Lesson progress saved."}
    except Exception as full_err:
        # Most likely the extra columns don't exist — retry with the minimal set.
        _log_exc("LESSON FULL INSERT FAILED (retrying minimal)", full_err)

    try:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO lesson_progress (email, lesson_id, completed_at)
                VALUES (:email, :lesson_id, NOW())
            """), {"email": email, "lesson_id": request.lesson_id})
        return {"status": "success", "message": "Lesson progress saved."}
    except Exception as e:
        _log_exc("LESSON INSERT FAILED", e)
        raise HTTPException(status_code=500, detail="Could not save lesson progress.") from e


@app.get("/api/lesson/progress")
def get_lesson_progress(user: dict = Depends(require_user)):
    """Return the authenticated user's lesson progress.

    Lets the frontend restore unlock state from the server (durable + synced
    across devices) instead of relying solely on the localStorage mirror.
    `unlocked_level` follows the client's convention: highest completed lesson
    id + 1 (so a fresh user with no completions starts at 1).
    """
    email = user["sub"]  # trust the token, not any client-supplied identity
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT DISTINCT lesson_id FROM lesson_progress WHERE email = :email"),
                {"email": email},
            ).fetchall()
        completed = sorted({r[0] for r in rows if r[0] is not None})
        unlocked_level = (max(completed) + 1) if completed else 1
        return {
            "status": "success",
            "completed_lessons": completed,
            "unlocked_level": unlocked_level,
        }
    except Exception as e:
        _log_exc("LESSON PROGRESS READ FAILED", e)
        raise HTTPException(status_code=500, detail="Could not load your progress.") from e


@app.get("/api/lesson/answers")
def get_lesson_answers(user: dict = Depends(require_user)):
    """Return the user's most recent saved answers for each lesson they finished.

    Lets the frontend show "your previous response" and pre-fill the wizard when a
    user reviews/redoes a lesson. A lesson can have several completion rows (each
    redo inserts a new one); DISTINCT ON keeps only the latest per lesson_id.

    Degrades to an empty map if the optional exercise_data/blueprint_data columns
    don't exist on this deployment, so an older schema doesn't 500 the client.
    """
    email = user["sub"]  # trust the token, not any client-supplied identity
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT DISTINCT ON (lesson_id)
                       lesson_id, exercise_data, blueprint_data, completed_at
                FROM lesson_progress
                WHERE email = :email
                ORDER BY lesson_id, completed_at DESC
            """), {"email": email}).fetchall()
        answers = {
            str(r[0]): {
                "exercise_data": r[1] or "",
                "blueprint_data": r[2] or "",
                "completed_at": str(r[3]) if r[3] is not None else None,
            }
            for r in rows if r[0] is not None
        }
        return {"status": "success", "answers": answers}
    except Exception as e:
        # Most likely the extra columns are absent on this schema — not fatal.
        _log_exc("LESSON ANSWERS READ FAILED (non-fatal)", e)
        return {"status": "success", "answers": {}}


@app.post("/api/lesson/analyze")
def analyze_lesson(request: LessonAnalyze, user: dict = Depends(require_user),
                   _rl: None = Depends(rate_limit_chat)):
    """Stream a short reflection on the user's Practice-step answers.

    Mirrors /api/chat: the deterministic crisis layer runs first, then the answers
    are sent to the model with a reflection-specific system prompt. Streamed as
    text/plain so the frontend reuses the same streaming reader.
    """
    print(
        f"🪞 /api/lesson/analyze lesson={request.lesson_id} email={user['sub']} "
        f"answers_len={len(request.answers)}"
    )

    # Safety first — never route self-harm content through the model.
    if detect_crisis(request.answers):
        print(f"⚠️ Crisis language detected in lesson analysis for {user['sub']}")

        def crisis_stream():
            yield CRISIS_RESPONSE

        return StreamingResponse(crisis_stream(), media_type="text/plain")

    # Second safety screen: explicit intent to seriously harm others.
    if detect_harm_to_others(request.answers):
        print(f"⚠️ Harm-to-others language detected in lesson analysis for {user['sub']}")

        def harm_others_stream():
            yield HARM_TO_OTHERS_RESPONSE

        return StreamingResponse(harm_others_stream(), media_type="text/plain")

    if client is None:
        print("🚨 /api/lesson/analyze called but Gemini (Vertex) client is None (init failed — check ADC / project / region).")
        raise HTTPException(status_code=503, detail="AI engine is unavailable.")

    is_takeaway = (request.mode or "reflect").lower() == "takeaway"

    framing = ""
    if request.skill or request.title:
        framing = f"This is part of a lesson on \"{request.skill or request.title}\".\n"
    if request.context:
        # Internal coaching context — lesson goal + what the exercise asked +
        # curriculum ai_prompt_context. Guides tone/focus, never echoed verbatim.
        framing += f"[GUIDE CONTEXT — do not quote this to the user]\n{request.context}\n\n"

    if is_takeaway:
        user_content = (
            f"{framing}"
            "Here is the takeaway the person wrote to close the lesson:\n\n"
            f"{request.answers}\n\n"
            "Write your warm response to their takeaway now."
        )
        reply_config = TAKEAWAY_CONFIG
    else:
        # Give the reflection real material to build on (like /api/chat): retrieve
        # relevant corpus passages keyed off the lesson topic + the person's own
        # words, and inject only when we actually got passages back.
        db_block = ""
        retrieval_query = f"{request.skill or request.title or ''} {request.answers}".strip()
        clinical_data = get_clinical_context(retrieval_query)
        if clinical_data.startswith("Here is the most relevant"):
            db_block = (
                "[DATABASE CONTEXT — weave in only if relevant, never mention the database]\n"
                f"{clinical_data}\n\n"
            )
        user_content = (
            f"{framing}"
            f"{db_block}"
            "Here are the answers the person typed for this exercise:\n\n"
            f"{request.answers}\n\n"
            "Reflect their answers back through the lens of what THIS lesson taught "
            "(see the insight and worked example in the guide context above). In a warm, "
            "plain voice:\n"
            "1. Briefly acknowledge the feeling, using their own words.\n"
            "2. Lay their answers out as the lesson's framework — name each part using the "
            "exact structure and terms from the insight and worked example, quoting what "
            "they actually wrote.\n"
            "3. Spotlight the single link or shift that is the heart of the lesson, so they "
            "see it inside their own example.\n"
            "Stay grounded in their words. Don't add new exercises, don't correct or grade "
            "them, and don't end on a separate question. Keep it under 150 words."
        )
        reply_config = REFLECT_CONFIG

    messages = [types.Content(role="user", parts=[types.Part.from_text(text=user_content)])]

    def stream_reply():
        start_time = time.time()
        input_tokens = output_tokens = 0
        try:
            print(f"🤖 Analyze({'takeaway' if is_takeaway else 'reflect'}): calling model={MODEL_ID}…")
            for chunk in client.models.generate_content_stream(
                model=MODEL_ID,
                contents=messages,
                config=reply_config,
            ):
                if chunk.text:
                    yield chunk.text
                usage = getattr(chunk, "usage_metadata", None)
                if usage:
                    if usage.prompt_token_count:
                        input_tokens = usage.prompt_token_count
                    if usage.candidates_token_count:
                        output_tokens = usage.candidates_token_count
            print(f"✅ Analyze reply OK in={input_tokens} out={output_tokens} tokens")
        except Exception as e:
            _log_exc("LESSON ANALYZE STREAMING ERROR", e)
            yield "\n\n_Sorry — I couldn't put my thoughts together just now. You can still continue to the next step._"

        # Reuse the chat telemetry table; session id marks this as a lesson reflection.
        analyze_metrics = ChatRequest(
            message=request.answers[:4000],
            session_id=f"lesson-{'takeaway' if is_takeaway else 'analyze'}-{request.lesson_id}",
            email=user["sub"],
        )
        _log_chat_metrics(analyze_metrics, round(time.time() - start_time, 2),
                          input_tokens, output_tokens)

    return StreamingResponse(stream_reply(), media_type="text/plain")


@app.get("/api/admin/metrics")
def get_admin_metrics(_admin: dict = Depends(require_admin)):
    try:
        with engine.connect() as conn:
            # 1. 📊 Fetch AI Telemetry
            try:
                ai_query = text("SELECT id, session_id, email, prompt_time_sec, input_tokens, output_tokens, created_at FROM ai_metrics_log ORDER BY created_at DESC LIMIT 50")
                ai_results = conn.execute(ai_query).fetchall()
                ai_list = [{"id": r[0], "session_id": r[1], "email": r[2], "prompt_time_sec": r[3], "input_tokens": r[4], "output_tokens": r[5], "created_at": str(r[6])} for r in ai_results]
            except Exception as e:
                _log_exc("ADMIN METRICS: ai_metrics_log read failed (non-fatal)", e)
                ai_list = [] # Failsafe if table doesn't exist yet

            # 2. 📖 Fetch Lesson Progress
            try:
                lesson_query = text("SELECT id, email, lesson_id, completed_at FROM lesson_progress ORDER BY completed_at DESC LIMIT 50")
                lesson_results = conn.execute(lesson_query).fetchall()
                lesson_list = [{"id": r[0], "email": r[1], "lesson_id": r[2], "completed_at": str(r[3])} for r in lesson_results]
            except Exception as e:
                _log_exc("ADMIN METRICS: lesson_progress read failed (non-fatal)", e)
                lesson_list = []

            # 3. 📔 Fetch Daily Check-In Logs (Mood Stats)
            try:
                # We use 'username' here because that's what we named the column in your /api/logs endpoint!
                log_query = text("SELECT username, mood, timestamp FROM logs ORDER BY timestamp DESC LIMIT 50")
                log_results = conn.execute(log_query).fetchall()
                log_list = [{"email": r[0], "mood": r[1], "timestamp": str(r[2])} for r in log_results]
            except Exception as e:
                _log_exc("ADMIN METRICS: logs read failed (non-fatal)", e)
                log_list = []

            # 4. ⚠️ Fetch flagged crisis sessions (most valuable for a clinician)
            try:
                crisis_query = text("SELECT session_id, email, message_excerpt, created_at FROM crisis_events ORDER BY created_at DESC LIMIT 50")
                crisis_results = conn.execute(crisis_query).fetchall()
                crisis_list = [{"session_id": r[0], "email": r[1], "message_excerpt": r[2], "created_at": str(r[3])} for r in crisis_results]
            except Exception as e:
                _log_exc("ADMIN METRICS: crisis_events read failed (non-fatal)", e)
                crisis_list = []

            # 5. 🧰 Fetch coping-tool (SOS/Reset) usage, aggregated per tool with
            # its average distress drop (pre − post SUDS). A NULL delta (rating
            # skipped) is ignored by AVG, so avg_drop reflects only rated sessions.
            try:
                tool_query = text("""
                    SELECT tool_id,
                           COUNT(*) AS uses,
                           ROUND(AVG(pre_distress - post_distress)::numeric, 1) AS avg_drop
                    FROM tool_events
                    GROUP BY tool_id
                    ORDER BY uses DESC
                """)
                tool_results = conn.execute(tool_query).fetchall()
                tools_list = [
                    {"tool_id": r[0], "uses": int(r[1]), "avg_drop": float(r[2]) if r[2] is not None else None}
                    for r in tool_results
                ]
            except Exception as e:
                _log_exc("ADMIN METRICS: tool_events read failed (non-fatal)", e)
                tools_list = []

            # 6. 📈 Aggregate summary cards. Each stat is independently failsafe so a
            # single missing table can't blank the whole strip. mood is stored as
            # TEXT ("1".."5"), so it must be cast before averaging.
            summary = {}
            for key, query in (
                ("active_users", "SELECT COUNT(DISTINCT email) FROM ai_metrics_log"),
                ("total_chats", "SELECT COUNT(*) FROM ai_metrics_log"),
                ("avg_latency_sec", "SELECT ROUND(AVG(prompt_time_sec)::numeric, 2) FROM ai_metrics_log"),
                ("lessons_completed", "SELECT COUNT(*) FROM lesson_progress"),
                ("avg_mood", "SELECT ROUND(AVG(NULLIF(mood, '')::float)::numeric, 2) FROM logs"),
                ("crisis_count", "SELECT COUNT(*) FROM crisis_events"),
                ("tool_sessions", "SELECT COUNT(*) FROM tool_events"),
                ("avg_distress_drop", "SELECT ROUND(AVG(pre_distress - post_distress)::numeric, 1) FROM tool_events"),
            ):
                try:
                    val = conn.execute(text(query)).scalar()
                    summary[key] = float(val) if val is not None else 0
                except Exception as e:
                    _log_exc(f"ADMIN METRICS: summary '{key}' failed (non-fatal)", e)
                    summary[key] = 0

            # Send the ultimate data package back to React
            return {
                "status": "success",
                "data": {
                    "summary": summary,
                    "telemetry": ai_list,
                    "lessons": lesson_list,
                    "logs": log_list,
                    "crisis": crisis_list,
                    "tools": tools_list
                }
            }
            
    except Exception as e:
        _log_exc("ADMIN DASHBOARD ERROR", e)
        raise HTTPException(status_code=500, detail="Failed to fetch metrics.") from e


@app.get("/api/admin/patient")
def get_patient_detail(email: str, _admin: dict = Depends(require_admin)):
    """Per-patient drill-down for the clinician portal.

    Combines one user's mood trajectory, lesson completions, chat volume, and
    crisis flags into a single view. Every sub-query is independently failsafe so
    a missing table degrades that field rather than the whole response. The email
    is always bound as a parameter — never string-formatted into SQL.
    """
    try:
        with engine.connect() as conn:
            # Mood series ordered ASC (oldest → newest) so the frontend sparkline
            # reads left-to-right. mood is TEXT; the frontend coerces to number.
            try:
                mood_rows = conn.execute(
                    text("SELECT mood, timestamp FROM logs WHERE username = :email ORDER BY timestamp ASC"),
                    {"email": email},
                ).fetchall()
                mood_series = [{"mood": r[0], "timestamp": str(r[1])} for r in mood_rows]
            except Exception as e:
                _log_exc("PATIENT DETAIL: mood_series failed (non-fatal)", e)
                mood_series = []

            try:
                lesson_rows = conn.execute(
                    text("SELECT lesson_id, completed_at FROM lesson_progress WHERE email = :email ORDER BY completed_at DESC"),
                    {"email": email},
                ).fetchall()
                lessons = [{"lesson_id": r[0], "completed_at": str(r[1])} for r in lesson_rows]
            except Exception as e:
                _log_exc("PATIENT DETAIL: lessons failed (non-fatal)", e)
                lessons = []

            def _scalar(query, default=0):
                try:
                    val = conn.execute(text(query), {"email": email}).scalar()
                    return val if val is not None else default
                except Exception as e:
                    _log_exc("PATIENT DETAIL: scalar failed (non-fatal)", e)
                    return default

            # Coping-tool usage for this patient, per tool + average distress drop.
            try:
                ptool_rows = conn.execute(
                    text("""
                        SELECT tool_id,
                               COUNT(*) AS uses,
                               ROUND(AVG(pre_distress - post_distress)::numeric, 1) AS avg_drop
                        FROM tool_events
                        WHERE email = :email
                        GROUP BY tool_id
                        ORDER BY uses DESC
                    """),
                    {"email": email},
                ).fetchall()
                tools = [
                    {"tool_id": r[0], "uses": int(r[1]), "avg_drop": float(r[2]) if r[2] is not None else None}
                    for r in ptool_rows
                ]
            except Exception as e:
                _log_exc("PATIENT DETAIL: tools failed (non-fatal)", e)
                tools = []

            chat_count = _scalar("SELECT COUNT(*) FROM ai_metrics_log WHERE email = :email")
            crisis_count = _scalar("SELECT COUNT(*) FROM crisis_events WHERE email = :email")
            tool_count = _scalar("SELECT COUNT(*) FROM tool_events WHERE email = :email")
            last_active = _scalar("SELECT MAX(created_at) FROM ai_metrics_log WHERE email = :email", default=None)

            return {
                "status": "success",
                "data": {
                    "email": email,
                    "mood_series": mood_series,
                    "lessons": lessons,
                    "tools": tools,
                    "chat_count": int(chat_count),
                    "crisis_count": int(crisis_count),
                    "tool_count": int(tool_count),
                    "last_active": str(last_active) if last_active is not None else None,
                },
            }

    except Exception as e:
        _log_exc("PATIENT DETAIL ERROR", e)
        raise HTTPException(status_code=500, detail="Failed to fetch patient detail.") from e