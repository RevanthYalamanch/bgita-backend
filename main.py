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
from safety import detect_crisis, CRISIS_RESPONSE
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
    context: Optional[str] = Field(default=None, max_length=4000)
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

# Cap how many prior turns we replay to the model, to bound prompt size/cost.
MAX_HISTORY_TURNS = int(os.getenv("CHAT_MAX_HISTORY_TURNS", "12"))

# Allow your frontend to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex = r"https://.*\.vercel\.app",
    allow_origins=[
        "http://localhost:3000" 
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
# GEMINI ON VERTEX AI
# ---------------------------------------------------------
# Gemini is served from us-central1 (same region as Cloud SQL here), so no
# special model region is needed. Override the model with the GEMINI_MODEL env
# var (e.g. "gemini-2.5-pro" for more nuance, or pin a dated version).
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "bgita-teacher")
GEMINI_REGION = os.getenv("GEMINI_REGION", "us-central1")
MODEL_ID = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
# Give responses headroom so multi-paragraph CBT replies don't truncate.
MAX_TOKENS = int(os.getenv("GEMINI_MAX_TOKENS", "2048"))

# Log the resolved AI config at startup so a bad model id / region is obvious
# the moment the container boots — no need to wait for a failed chat to find out.
print(
    f"🔧 AI config: project={PROJECT_ID} region={GEMINI_REGION} "
    f"model={MODEL_ID} max_tokens={MAX_TOKENS}"
)

# Vertex-backed google-genai client. Project/location live on the client; the
# system prompt + token cap are attached per-request via GEN_CONFIG (built once
# SYSTEM_PROMPT is defined below). Reads Application Default Credentials.
try:
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=GEMINI_REGION)
    print("✅ Vertex AI (google-genai) client initialized.")
except Exception as e:
    _log_exc("GENAI CLIENT INIT ERROR", e)
    client = None

# System prompt: defines the assistant's role and hard rules. This is passed to
# Gemini as `system_instruction` (not interleaved with the user message), so the
# model treats it as standing instructions rather than user-provided text.
SYSTEM_PROMPT = """You are an empathetic, highly skilled Cognitive Behavioral Therapy (CBT) therapist.
You draw upon the psychological frameworks found in the Bhagavad Gita, but you MUST present them using modern, accessible, secular western terminology.

SAFETY (HIGHEST PRIORITY, overrides all formatting rules below):
- If the user expresses any intent or desire to harm themselves or others, or to end their life, your FIRST priority is their safety, not therapy techniques.
- Respond with warmth and without judgment, gently encourage them to reach out for immediate human help, and surface crisis resources: in the US, call or text 988 (Suicide & Crisis Lifeline) or call 911 in immediate danger; elsewhere, direct them to local emergency services or https://findahelpline.com.
- Never minimize, argue with, or shame these feelings, and never provide instructions that could facilitate self-harm.

Guidelines for your response Content:
1. Translate ancient concepts into universal psychological principles.
2. Avoid using Sanskrit terms, character names, or Indian metaphors unless asked.
3. Validate the user's feelings first using standard CBT empathy.
4. You may be given text retrieved from our clinical database. IF it is relevant, weave it in.
5. IF the database text is not relevant, DO NOT mention the database.
6. Never say "According to the database".
7. Offer the user the ability to ask for the response in a simpler format.
8. Use CBT-adjacent questions to engage users further.
9. When a user asks a question, provide historical background with specific examples.

Guidelines for your response Formatting (CRITICAL):
- NEVER output a single wall of text. Break your responses into short, easily digestible paragraphs (maximum 2-3 sentences per paragraph).
- Use Markdown formatting to make your response visually structured and scannable.
- Use bullet points or numbered lists when explaining multiple concepts, actionable steps, or cognitive reframing exercises.
- Use **bold text** to gently emphasize key psychological terms or core takeaways."""


# Standing generation config now that SYSTEM_PROMPT exists. Gemini treats
# `system_instruction` as standing instructions, separate from user turns; the
# token cap keeps multi-paragraph replies from truncating. Passed on every call.
GEN_CONFIG = types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,
    max_output_tokens=MAX_TOKENS,
)


# Lesson "Practice" reflection (#2) and Commit-step takeaway (#3) both run on the
# SAME standard chatbot system prompt as the main chat (SYSTEM_PROMPT) — per the
# 6/29 request to replace the bespoke lesson prompts with the main one so the
# guide's voice is consistent everywhere. The per-mode steering (reflect vs.
# takeaway) is carried in the user turn assembled in analyze_lesson(), not here.
# Only the token cap differs from the main chat config (shorter lesson replies).
ANALYSIS_GEN_CONFIG = types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,
    max_output_tokens=int(os.getenv("ANALYSIS_MAX_TOKENS", "600")),
)

TAKEAWAY_GEN_CONFIG = types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,
    max_output_tokens=int(os.getenv("ANALYSIS_MAX_TOKENS", "600")),
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


_ensure_metrics_table()
_ensure_lesson_progress_table()


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
    doesn't bloat every historical turn. Gemini uses the roles "user" and
    "model" (its name for the assistant) and requires the conversation to start
    with a user turn, so any leading model turns (e.g. the UI's greeting) are
    dropped.
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

    contents = []
    for turn in (request.history or [])[-MAX_HISTORY_TURNS:]:
        # Map Anthropic-style "assistant" to Gemini's "model" role.
        role = "model" if turn.role == "assistant" else "user"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=turn.content)]))
    # Drop leading model turn(s) so the conversation starts with the user.
    while contents and contents[0].role != "user":
        contents.pop(0)
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_content)]))
    return contents


@app.post("/api/chat")
def chat_with_gita(request: ChatRequest, _rl: None = Depends(rate_limit_chat)):
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

        def crisis_stream():
            yield CRISIS_RESPONSE
            _log_chat_metrics(request, 0.0, 0, 0)

        return StreamingResponse(crisis_stream(), media_type="text/plain")

    if client is None:
        print("🚨 /api/chat called but google-genai client is None (init failed).")
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
            print(f"🤖 Calling Vertex model={MODEL_ID} region={GEMINI_REGION} with {len(messages)} messages…")
            stream = client.models.generate_content_stream(
                model=MODEL_ID, contents=messages, config=GEN_CONFIG
            )
            for chunk in stream:
                # A chunk may carry no text (e.g. a safety-only or usage-only
                # chunk); guard so we don't raise on .text.
                try:
                    if chunk.text:
                        yield chunk.text
                except (ValueError, IndexError):
                    pass
                usage = getattr(chunk, "usage_metadata", None)
                if usage:
                    # Gemini reports cumulative counts; keep the latest non-null
                    # seen (intermediate chunks may report None).
                    if usage.prompt_token_count is not None:
                        input_tokens = usage.prompt_token_count
                    if usage.candidates_token_count is not None:
                        output_tokens = usage.candidates_token_count
            print(f"✅ Vertex reply OK in={input_tokens} out={output_tokens} tokens")
        except Exception as e:
            # Surface type + traceback, and flag the usual suspects so the next
            # failure is diagnosable straight from the log line.
            _log_exc("AI STREAMING ERROR", e)
            status = getattr(e, "status_code", getattr(e, "code", None))
            print(
                f"   ↳ context: status_code={status} model={MODEL_ID} "
                f"region={GEMINI_REGION} project={PROJECT_ID}"
            )
            if status == 404:
                print("   ↳ HINT: 404 usually means the model id isn't valid or enabled in this region. "
                      "Check GEMINI_MODEL / GEMINI_REGION.")
            elif status in (401, 403):
                print("   ↳ HINT: auth/permission — the Cloud Run service account may lack roles/aiplatform.user, "
                      "or the Vertex AI API isn't enabled for this project.")
            elif status == 429:
                print("   ↳ HINT: Vertex quota/rate limit. Check quotas for the region or back off.")
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
            return {"status": "success", "message": f"User registered successfully as {assigned_role}!"}
            
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


@app.post("/api/logs")
async def save_daily_log(log: LogCreate, db: Session = Depends(get_db),
                         user: dict = Depends(require_user)):
    try:
        # Trust the authenticated identity from the token, NOT the request body,
        # so a user can only write journal entries under their own account.
        username = user["sub"]

        # We updated the column names to match your X-ray exactly!
        query = text("""
            INSERT INTO logs (username, timestamp, mood, activity, reflection)
            VALUES (:username, NOW(), :mood, :activity, :reflection)
        """)

        db.execute(query, {
            "username": username,          # Saving their email into the 'username' column
            "mood": str(log.mood_score),   # Converting the 1-5 number into TEXT
            "activity": "Daily Journal",   # Filling the required activity column with a default label
            "reflection": log.diary_text   # Saving the paragraph into the 'reflection' column
        })
        db.commit()
        
        return {"status": "success", "message": "Journal entry saved to database!"}
    except Exception as e:
        db.rollback()
        # Log server-side only; return a true error status with a generic message
        # (a 200 here would make the client falsely believe the entry was saved).
        _log_exc("LOG SAVE ERROR", e)
        raise HTTPException(status_code=500, detail="Could not save your entry. Please try again.") from e


@app.get("/api/logs")
def get_daily_logs(user: dict = Depends(require_user), db: Session = Depends(get_db)):
    """Return the authenticated user's past daily check-ins, most recent first.

    Powers the "view previous diaries" history (6/29 #1). The frontend also uses
    the most-recent timestamp to tell whether the user has already checked in
    today, so the Save Entry button only acts once per day.
    """
    try:
        rows = db.execute(text("""
            SELECT mood, reflection, timestamp
            FROM logs
            WHERE username = :username AND activity = 'Daily Journal'
            ORDER BY timestamp DESC
            LIMIT 90
        """), {"username": user["sub"]}).fetchall()
        return {
            "entries": [
                {"mood": r[0], "reflection": r[1], "timestamp": str(r[2])}
                for r in rows
            ]
        }
    except Exception as e:
        _log_exc("LOG FETCH ERROR", e)
        raise HTTPException(status_code=500, detail="Could not load your entries.") from e


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

    if client is None:
        print("🚨 /api/lesson/analyze called but google-genai client is None (init failed).")
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
        gen_config = TAKEAWAY_GEN_CONFIG
    else:
        user_content = (
            f"{framing}"
            "Here are the answers the person typed for this exercise:\n\n"
            f"{request.answers}\n\n"
            "Write your brief reflection back to them now."
        )
        gen_config = ANALYSIS_GEN_CONFIG

    contents = [types.Content(role="user", parts=[types.Part.from_text(text=user_content)])]

    def stream_reply():
        start_time = time.time()
        input_tokens = output_tokens = 0
        try:
            print(f"🤖 Analyze({'takeaway' if is_takeaway else 'reflect'}): calling model={MODEL_ID} region={GEMINI_REGION}…")
            stream = client.models.generate_content_stream(
                model=MODEL_ID, contents=contents, config=gen_config
            )
            for chunk in stream:
                try:
                    if chunk.text:
                        yield chunk.text
                except (ValueError, IndexError):
                    pass
                usage = getattr(chunk, "usage_metadata", None)
                if usage:
                    if usage.prompt_token_count is not None:
                        input_tokens = usage.prompt_token_count
                    if usage.candidates_token_count is not None:
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

            # Send the ultimate data package back to React
            return {
                "status": "success", 
                "data": {
                    "telemetry": ai_list,
                    "lessons": lesson_list,
                    "logs": log_list
                }
            }
            
    except Exception as e:
        _log_exc("ADMIN DASHBOARD ERROR", e)
        raise HTTPException(status_code=500, detail="Failed to fetch metrics.") from e