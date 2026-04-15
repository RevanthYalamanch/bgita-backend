import re
from fastapi import FastAPI, HTTPException, Request,Depends
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import vertexai
from vertexai.generative_models import GenerativeModel
import os
from sqlalchemy import text
from database import engine, get_db
import hashlib
from sqlalchemy.orm import Session
from typing import Optional
import time

app = FastAPI()

class LogCreate(BaseModel):
    email: str
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

class ChatRequest(BaseModel):
    message: str
    context: Optional[str] = None
    session_id: Optional[str] = "anonymous_session"
    email: Optional[str] = "unknown_user"

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
# Initialize the Gemini 2.5 model via Vertex AI
try:
    vertexai.init(project=os.getenv("GOOGLE_CLOUD_PROJECT", "bgita-teacher"), location="us-central1")
    model = GenerativeModel("gemini-2.5-flash") 
except Exception as e:
    print(f"Vertex AI Init Error: {e}")





# ---------------------------------------------------------
# THE DATABASE SEARCH ENGINE
# ---------------------------------------------------------
def get_clinical_context(user_message: str) -> str:
    """Searches the LangChain vector database for relevant verses."""
    try:
        with engine.connect() as conn:
            # Search the LangChain table's 'document' column
            query = text("""
                SELECT document 
                FROM langchain_pg_embedding 
                WHERE document ILIKE :search_term
                LIMIT 3
            """)
            
            # Look for a verse number (like "2.14") in the instruction prompt
            match = re.search(r'\d+\.\d+', user_message)
            if match:
                search_term = f"%{match.group(0)}%"
            else:
                # Fallback: grab a long word from the user's message to search
                words = [w for w in user_message.split() if len(w) > 4]
                search_term = f"%{words[-1]}%" if words else "%mind%"
            
            results = conn.execute(query, {"search_term": search_term}).fetchall()
            
            if not results:
                return "No specific verses found in the clinical database for this exact topic."
                
            # Bundle the database results into a neat string for the AI
            db_text = "Here is the exact text pulled directly from our clinical_db:\n"
            for row in results:
                db_text += f"- {row[0]}\n"
                
            return db_text
            
    except Exception as e:
        print(f"Database search failed: {e}")
        return "Warning: Could not connect to the clinical database tables."
            
    except Exception as e:
        print(f"Database search failed: {e}")
        return "Warning: Could not connect to the clinical database tables."

# ---------------------------------------------------------
# THE NEW CHAT ENDPOINT
# ---------------------------------------------------------
@app.post("/api/chat")
def chat_with_gita(request: ChatRequest):
    try:
        clinical_data = get_clinical_context(request.message)
        
        # 2. If the frontend sends lesson context, format it for the AI
        lesson_instructions = ""
        if request.context:
            lesson_instructions = f"\n[CURRENT LESSON CONTEXT]\n{request.context}\nFocus your entire response on guiding the user through this specific lesson and do not change the subject.\n"
        
        # 3. Inject it into the Super Prompt
        augmented_prompt = f"""
        You are an empathetic, highly skilled Cognitive Behavioral Therapy (CBT) therapist. 
        You draw upon the psychological frameworks found in the Bhagavad Gita, but you MUST present them using modern, accessible, secular western terminology.
        
        Guidelines for your response Content:
        1. Translate ancient concepts into universal psychological principles.
        2. Avoid using Sanskrit terms, character names, or Indian metaphors unless asked.
        3. Validate the user's feelings first using standard CBT empathy.
        4. Below is some text retrieved from our clinical database. IF it is relevant, weave it in.
        5. IF the database text is not relevant, DO NOT mention the database.
        6. Never say "According to the database".
        7. Offer user ability to ask for response in a simpler format.
        8. Use CBT adjacent questions to engage users further.
        9. When a user asks a question, ask for historical background with specific examples. 
        
        Guidelines for your response Formatting (CRITICAL):
        7. NEVER output a single wall of text. Break your responses into short, easily digestible paragraphs (maximum 2-3 sentences per paragraph).
        8. Use Markdown formatting to make your response visually structured and scannable.
        9. Use bullet points or numbered lists when explaining multiple concepts, actionable steps, or cognitive reframing exercises.
        10. Use **bold text** to gently emphasize key psychological terms or core takeaways.
        
        {lesson_instructions}
        
        [DATABASE CONTEXT START]
        {clinical_data}
        [DATABASE CONTEXT END]
        
        User Message:
        {request.message} 
        """
        # ⏱️ START THE TIMER
        start_time = time.time()
        
        response = model.generate_content(augmented_prompt)
        
        # ⏱️ STOP THE TIMER
        prompt_time = round(time.time() - start_time, 2)
        
        # 📊 EXTRACT TOKENS
        input_tokens = 0
        output_tokens = 0
        if hasattr(response, "usage_metadata"):
            input_tokens = response.usage_metadata.prompt_token_count
            output_tokens = response.usage_metadata.candidates_token_count
            
        # 💾 SAVE METRICS TO POSTGRESQL
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

        return {"reply": response.text}
        
    except Exception as e:
        return {"reply": f"An error occurred in the AI engine: {str(e)}"}



@app.post("/api/register")
def register_user(request: RegisterRequest):
    try:
        # engine.begin() automatically commits the data to the database!
        with engine.begin() as conn:
            # Hash the password so it isn't saved as plain text
            hashed_pw = hashlib.sha256(request.password.encode()).hexdigest()

            SECRET_CODE = "abc123"
            assigned_role = "admin" if request.admin_code == SECRET_CODE else "user"
            
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
            return {"status": "success", "message": "User registered successfully as {assigned_role}!"}
            
    except Exception as e:
        print(f"🚨 REAL DATABASE ERROR: {e}") 
        # 🔥 THE FIX: We must raise a true HTTP error so Next.js doesn't fake a successful login!
        raise HTTPException(status_code=400, detail=f"Database Error: {str(e)}")
        
@app.post("/api/login")
def login_user(request: LoginRequest):
    try:
        with engine.connect() as conn:
            hashed_pw = hashlib.sha256(request.password.encode()).hexdigest()
            
            # Look for a match in the database
            query = text("""
                SELECT name, roles, email 
                FROM users 
                WHERE email = :email AND password = :password
            """)
            
            # Note: In SQLAlchemy 2.0, passing the dict directly into execute works best this way
            result = conn.execute(query, {"email": request.email, "password": hashed_pw}).fetchone()
            
            if result:
                # Success! Send the user's secure data back
                return {
                    "status": "success", 
                    "user": {"name": result[0], "role": result[1], "email": result[2]}
                }
            else:
                # 🔥 FIX: Raise a true HTTP 401 error so Next.js knows it failed
                raise HTTPException(status_code=401, detail="Invalid email or password.")
                
    except Exception as e:
        # 🔥 FIX: Raise a true HTTP 500 error for database crashes
        raise HTTPException(status_code=500, detail=str(e))

from sqlalchemy import text

@app.post("/api/logs")
async def save_daily_log(log: LogCreate, db: Session = Depends(get_db)):
    try:
        # We updated the column names to match your X-ray exactly!
        query = text("""
            INSERT INTO logs (username, timestamp, mood, activity, reflection) 
            VALUES (:username, NOW(), :mood, :activity, :reflection)
        """)
        
        db.execute(query, {
            "username": log.email,         # Saving their email into the 'username' column
            "mood": str(log.mood_score),   # Converting the 1-5 number into TEXT
            "activity": "Daily Journal",   # Filling the required activity column with a default label
            "reflection": log.diary_text   # Saving the paragraph into the 'reflection' column
        })
        db.commit()
        
        return {"status": "success", "message": "Journal entry saved to database!"}
    except Exception as e:
        db.rollback()
        return {"status": "error", "message": str(e)}

@app.get("/api/admin/metrics")
def get_admin_metrics():
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