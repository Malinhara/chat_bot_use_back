import os
from pathlib import Path
from typing import Dict, Optional,List
import openai
from langchain_openai import OpenAIEmbeddings
from fastapi import Depends, FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel
from dotenv import load_dotenv
from passlib.context import CryptContext
import psycopg2
from psycopg2 import OperationalError
import bcrypt

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

app = FastAPI()

class UserLogin(BaseModel):
    email: str
    password: str

class UserRegister(BaseModel):
    email: str
    password: str
    
class User(Base):
    __tablename__ = 'user'

    email = Column(String(255), primary_key=True)  # Ensure a sufficient length for email
    password = Column(String(255))  # Ensure a sufficient length for password

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found")



openai.api_key = OPENAI_API_KEY

UPLOAD_DIR = Path("uploads")
VECTOR_DIR = Path("vector_stores")
ALLOWED_EXTENSIONS = {".pdf", ".txt"}

UPLOAD_DIR.mkdir(exist_ok=True)
VECTOR_DIR.mkdir(exist_ok=True)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatQuery(BaseModel):
    query: str
    doc_name: str


class delete(BaseModel):
    filename: str

vector_store_cache: Dict[str, FAISS] = {}

def process_document(file_path: Path) -> FAISS:
    loader = PyPDFLoader(str(file_path)) if file_path.suffix == ".pdf" else TextLoader(str(file_path))
    documents = loader.load()
    
    texts = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    ).split_documents(documents)
    
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(texts, embeddings)

def get_vectorstore(file_path: Path) -> FAISS:
    if file_path.stem in vector_store_cache:
        return vector_store_cache[file_path.stem]
    
    vectorstore = process_document(file_path)
    vector_store_cache[file_path.stem] = vectorstore
    return vectorstore




def verify_db_connection():
    try:
        # Try to connect to the database
        with psycopg2.connect(DATABASE_URL) as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1;")
                result = cursor.fetchone()
                if result:
                    print("Database connection verified successfully!")
                else:
                    print("Failed to verify the database connection.")
    except OperationalError as e:
        print(f"Database connection failed: {e}")

# Verify the connection when the app starts
verify_db_connection()



@app.post("/register")
async def register(user: UserRegister, db: Session = Depends(get_db)):
    # Query the database using the User model
    db_user = db.query(User).filter(User.email == user.email).first()
    
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Hash the password using CryptContext
    hashed_password = pwd_context.hash(user.password)
    
    # Create a new User instance (use the actual User model here)
    new_user = User(email=user.email, password=hashed_password)
    
    # Add the new user to the session, commit, and refresh to get the updated user
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return {"message": "User registered successfully"}


# Login route
@app.post("/login")
async def login(user: UserLogin, db: Session = Depends(get_db)):
    # Query the database using the User model and the email
    db_user = db.query(User).filter(User.email == user.email).first()
    
    # Check if the user exists and if the password is correct
    if not db_user or not pwd_context.verify(user.password, db_user.password):
        raise HTTPException(status_code=400, detail="Invalid credentials")
    
    return {"message": "Login successful"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if Path(file.filename).suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    
    file_path = UPLOAD_DIR / file.filename
    try:
        content = await file.read()
        file_path.write_bytes(content)
        get_vectorstore(file_path)
        return {"message": "Success", "doc_name": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
@app.get("/uploaded_files", response_model=List[str])
async def get_uploaded_files():
    try:
        # Retrieve list of files in the UPLOAD_DIR
        files = [f.name for f in UPLOAD_DIR.iterdir() if f.is_file()]
        return files
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/delete_file")
async def delete_file(deletefile: delete):
    file_path = UPLOAD_DIR / deletefile.filename
    vectorstore_path = VECTOR_DIR / f"{file_path.stem}.faiss"
    
    # Check if the file exists and delete it
    if file_path.exists():
        try:
            os.remove(file_path)
            print(f"File {deletefile.filename} deleted successfully.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")
    
    # Check if the vector store exists and delete it
    if vectorstore_path.exists():
        try:
            os.remove(vectorstore_path)
            print(f"Vector store for {deletefile.filename} deleted successfully.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error deleting vector store: {str(e)}")
    
    return {"message": "File and related data deleted successfully."}

@app.post("/chat")
async def chat_with_document(chat_query: ChatQuery):

    file_path = UPLOAD_DIR / chat_query.doc_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        vectorstore = get_vectorstore(file_path)
        qa_chain = RetrievalQA.from_chain_type(
            OpenAI(temperature=0),
            retriever=vectorstore.as_retriever()
        )
    
        return {"response": qa_chain.run(chat_query.query)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

