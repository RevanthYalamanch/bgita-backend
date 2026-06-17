from fastapi import FastAPI, HTTPException, Request, Depends, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from fastapi.middleware.cors import CORSMiddleware
from anthropic import AnthropicVertex
import os
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
    # Prior turns of the conversation, oldest first. Lets Claude maintain
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

# Cap how many prior turns we replay to Claude, to bound prompt size/cost.
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
# CLAUDE ON VERTEX AI (Model Garden)
# ---------------------------------------------------------
# NOTE: Anthropic models on Vertex are served from specific regions (Sonnet 4.6
# is in us-east5, europe-west1, asia-southeast1), which may differ from the
# region used for Cloud SQL or other Vertex services. Confirm the exact model id
# and region for your project before deploy.
#
# Model: Claude Sonnet 4.6 — a meaningful quality upgrade over Haiku 4.5 for the
# nuance a therapy assistant needs, at a higher per-token cost. Override with the
# CLAUDE_MODEL env var (e.g. a pinned "claude-sonnet-4-6@<date>" if your region
# requires the dated version, or "claude-haiku-4-5@20251001" to revert).
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "bgita-teacher")
CLAUDE_REGION = os.getenv("ANTHROPIC_VERTEX_REGION", "us-east5")
MODEL_ID = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
# Sonnet 4.6 writes richer, multi-paragraph CBT responses; give it more headroom
# than Haiku's 1024 so replies don't truncate mid-thought.
MAX_TOKENS = int(os.getenv("CLAUDE_MAX_TOKENS", "2048"))

try:
    client = AnthropicVertex(project_id=PROJECT_ID, region=CLAUDE_REGION)
except Exception as e:
    print(f"Anthropic Vertex Init Error: {e}")
    client = None

# System prompt: defines the assistant's role and hard rules. With Claude this
# belongs in the `system` parameter (not interleaved with the user message), so
# the model treats it as standing instructions rather than user-provided text.
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


def _keyword_search(user_message: str, limit: int):
    """Lexical full-text search over the corpus (best-effort).

    Complements vector search: catches exact terms/proper nouns the embedding
    might miss ("Arjuna", "dharma", a specific verse). Runs in its own
    connection so a failure here never disturbs the vector query, and returns []
    on any error (e.g. an empty/stopword-only query) so the caller degrades to
    vector-only retrieval.
    """
    if limit <= 0:
        return []
    try:
        with engine.connect() as conn:
            sql = text("""
                SELECT content, embedding
                FROM gita_chunks
                WHERE to_tsvector('english', content) @@ plainto_tsquery('english', :q)
                LIMIT :k
            """)
            return conn.execute(sql, {"q": user_message, "k": limit}).fetchall()
    except Exception as e:
        print(f"Keyword search failed (non-fatal): {e}")
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

        keyword_rows = _keyword_search(user_message, RAG_KEYWORD_CANDIDATES)

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
        print(f"Semantic search failed: {e}")
        return "Warning: Could not retrieve from the clinical database."

# ---------------------------------------------------------
# THE NEW CHAT ENDPOINT
# ---------------------------------------------------------
def _log_chat_metrics(request: ChatRequest, prompt_time: float,
                      input_tokens: int, output_tokens: int):
    """Persist AI telemetry. Best-effort: never raises into the request path."""
    try:
        with engine.begin() as conn:
            # Auto-create the telemetry table
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
        print(f"Failed to log metrics (non-fatal): {db_err}")


def _build_messages(request: ChatRequest) -> list:
    """Assemble the Claude `messages` array: prior turns + the current turn.

    The RAG/lesson context is attached only to the *current* user message so it
    doesn't bloat every historical turn. Anthropic requires the first message to
    be from the user, so any leading assistant turns (e.g. the UI's greeting)
    are dropped.
    """
    clinical_data = get_clinical_context(request.message)

    lesson_instructions = ""
    if request.context:
        lesson_instructions = f"\n[CURRENT LESSON CONTEXT]\n{request.context}\nFocus your entire response on guiding the user through this specific lesson and do not change the subject.\n"

    user_content = f"""{lesson_instructions}
        [DATABASE CONTEXT START]
        {clinical_data}
        [DATABASE CONTEXT END]

        User Message:
        {request.message}"""

    messages = []
    for turn in (request.history or [])[-MAX_HISTORY_TURNS:]:
        messages.append({"role": turn.role, "content": turn.content})
    # Drop leading assistant turn(s) so the conversation starts with the user.
    while messages and messages[0]["role"] != "user":
        messages.pop(0)
    messages.append({"role": "user", "content": user_content})
    return messages


@app.post("/api/chat")
def chat_with_gita(request: ChatRequest, _rl: None = Depends(rate_limit_chat)):
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
        raise HTTPException(status_code=503, detail="AI engine is unavailable.")

    messages = _build_messages(request)

    def stream_reply():
        start_time = time.time()
        input_tokens = output_tokens = 0
        try:
            with client.messages.stream(
                model=MODEL_ID,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=messages,
            ) as stream:
                for delta in stream.text_stream:
                    yield delta
                final = stream.get_final_message()
                input_tokens = final.usage.input_tokens
                output_tokens = final.usage.output_tokens
        except Exception as e:
            print(f"AI streaming error: {e}")
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
            
    except Exception as e:
        print(f"🚨 REAL DATABASE ERROR: {e}") 
        # 🔥 THE FIX: We must raise a true HTTP error so Next.js doesn't fake a successful login!
        raise HTTPException(status_code=400, detail=f"Database Error: {str(e)}")
        
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
        raise HTTPException(status_code=500, detail=str(e))

from sqlalchemy import text

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
        return {"status": "error", "message": str(e)}

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
        print(f"Full lesson insert failed, retrying minimal: {full_err}")

    try:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO lesson_progress (email, lesson_id, completed_at)
                VALUES (:email, :lesson_id, NOW())
            """), {"email": email, "lesson_id": request.lesson_id})
        return {"status": "success", "message": "Lesson progress saved."}
    except Exception as e:
        print(f"Lesson insert failed: {e}")
        raise HTTPException(status_code=500, detail="Could not save lesson progress.")


@app.get("/api/admin/metrics")
def get_admin_metrics(_admin: dict = Depends(require_admin)):
    try:
        with engine.connect() as conn:
            # 1. 📊 Fetch AI Telemetry
            try:
                ai_query = text("SELECT id, session_id, email, prompt_time_sec, input_tokens, output_tokens, created_at FROM ai_metrics_log ORDER BY created_at DESC LIMIT 50")
                ai_results = conn.execute(ai_query).fetchall()
                ai_list = [{"id": r[0], "session_id": r[1], "email": r[2], "prompt_time_sec": r[3], "input_tokens": r[4], "output_tokens": r[5], "created_at": str(r[6])} for r in ai_results]
            except Exception:
                ai_list = [] # Failsafe if table doesn't exist yet

            # 2. 📖 Fetch Lesson Progress
            try:
                lesson_query = text("SELECT id, email, lesson_id, completed_at FROM lesson_progress ORDER BY completed_at DESC LIMIT 50")
                lesson_results = conn.execute(lesson_query).fetchall()
                lesson_list = [{"id": r[0], "email": r[1], "lesson_id": r[2], "completed_at": str(r[3])} for r in lesson_results]
            except Exception:
                lesson_list = []

            # 3. 📔 Fetch Daily Check-In Logs (Mood Stats)
            try:
                # We use 'username' here because that's what we named the column in your /api/logs endpoint!
                log_query = text("SELECT username, mood, timestamp FROM logs ORDER BY timestamp DESC LIMIT 50")
                log_results = conn.execute(log_query).fetchall()
                log_list = [{"email": r[0], "mood": r[1], "timestamp": str(r[2])} for r in log_results]
            except Exception:
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
        print(f"Admin Dashboard Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch metrics.")