# backend/database.py
import os
from google.cloud.sql.connector import Connector
import sqlalchemy
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()
connector = Connector()

def getconn():
    """Establishes the secure connection to your old Cloud SQL instance."""
    return connector.connect(
        os.getenv("INSTANCE_CONNECTION_NAME"),
        "pg8000",
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASS"),
        db=os.getenv("DB_NAME")
    )

# Create the engine
engine = sqlalchemy.create_engine("postgresql+pg8000://", creator=getconn)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ---------------------------------------------------------
# DATABASE REFLECTION: Load all existing tables automatically
# ---------------------------------------------------------
Base = automap_base()
Base.prepare(autoload_with=engine)

# Create a session factory to use in your API routes
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Example: If your old database had a table named "users", 
# Python will automatically map it to a class for you:
# User = Base.classes.users

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()