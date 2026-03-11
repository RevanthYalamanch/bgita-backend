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

app = FastAPI()

class LogCreate(BaseModel):
    email: str
    mood_score: int
    diary_text: str

# Allow your frontend to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://3000-cs-YOUR-UNIQUE-URL.cloudshell.dev"],
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

class ChatRequest(BaseModel):
    message: str
    context: Optional[str] = None



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
        You are an empathetic, highly skilled Cognitive Behavioral Therapy (CBT) guide. 
        You draw upon the psychological frameworks found in the Bhagavad Gita, but you MUST present them using modern, accessible, secular western terminology.
        
        Guidelines for your response:
        1. Translate ancient concepts into universal psychological principles.
        2. Avoid using Sanskrit terms, character names, or Indian metaphors unless asked.
        3. Validate the user's feelings first using standard CBT empathy.
        4. Below is some text retrieved from our clinical database. IF it is relevant, weave it in.
        5. IF the database text is not relevant, DO NOT mention the database.
        6. Never say "According to the database".
        
        {lesson_instructions}
        
        [DATABASE CONTEXT START]
        {clinical_data}
        [DATABASE CONTEXT END]
        
        User Message:
        {request.message} 
        """
        
        response = model.generate_content(augmented_prompt)
        return {"reply": response.text}
        
    except Exception as e:
        return {"reply": f"An error occurred in the AI engine: {str(e)}"}

class AuthRequest(BaseModel):
    email: str
    password: str
    name: str = ""
    role: str = "user"

@app.post("/api/register")
def register_user(request: AuthRequest):
    try:
        # engine.begin() automatically commits the data to the database!
        with engine.begin() as conn:
            # Hash the password so it isn't saved as plain text
            hashed_pw = hashlib.sha256(request.password.encode()).hexdigest()
            
            # Notice the ARRAY[:role] down below! This fixes the Postgres error.
            query = text("""
                INSERT INTO users (username, email, password, name, roles) 
                VALUES (:email, :email, :password, :name, ARRAY[:role])
            """)
            
            conn.execute(query, {
                "email": request.email, 
                "password": hashed_pw, 
                "name": request.name, 
                "role": request.role
            })
            return {"status": "success", "message": "User registered successfully!"}
            
    except Exception as e:
        print(f"🚨 REAL DATABASE ERROR: {e}") 
        return {"status": "error", "message": "Registration failed. Email may already exist."}
        
@app.post("/api/login")
def login_user(request: AuthRequest):
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

from pydantic import BaseModel
from ai_engine import GitaAIEngine

# Initialize your AI Engine
ai_engine = GitaAIEngine()

# Define the structure of an incoming chat message
class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
async def therapy_chat(request: ChatRequest):
    try:
        # Pass the user's message to Vertex AI
        ai_response = ai_engine.generate_response(request.message)
        
        # In case the response is an object, extract the text
        reply_text = ai_response.text if hasattr(ai_response, 'text') else str(ai_response)
        
        return {"reply": reply_text}
    except Exception as e:
        return {"reply": f"I'm having trouble connecting to my thoughts right now. (Error: {str(e)})"}