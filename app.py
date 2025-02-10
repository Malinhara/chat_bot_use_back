# import os
# import openai  # or use Bedrock if required
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.document_loaders import PyPDFLoader, TextLoader
# from langchain.llms import OpenAI  # Can be replaced with Bedrock LLM
# from langchain.chains import RetrievalQA
# from fastapi import FastAPI, File, UploadFile, Form
# from dotenv import load_dotenv
# load_dotenv()
# app = FastAPI()

# openai_api_key = os.getenv("OPENAI_API_KEY")  
# openai.api_key = openai_api_key

# def process_document(file_path):
#     if file_path.endswith(".pdf"):
#         loader = PyPDFLoader(file_path)
#     else:
#         loader = TextLoader(file_path)

#     documents = loader.load()
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     texts = text_splitter.split_documents(documents)

#     embeddings = OpenAIEmbeddings()
#     vectorstore = FAISS.from_documents(texts, embeddings)
#     return vectorstore

# @app.post("/upload/")
# async def upload_document(file: UploadFile = File(...)):
#     file_path = f"uploads/{file.filename}"
#     os.makedirs("uploads", exist_ok=True)

#     with open(file_path, "wb") as buffer:
#         buffer.write(await file.read())

#     vectorstore = process_document(file_path)
#     return {"message": "File uploaded and processed successfully", "doc_name": file.filename}

# @app.post("/chat")
# async def chat_with_document(query: str = Form(...), doc_name: str = Form(...)):
#     file_path = f"uploads/{doc_name}"
#     vectorstore = process_document(file_path)

#     retriever = vectorstore.as_retriever()
#     llm = OpenAI()
#     qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

#     response = qa_chain.run(query)
#     return {"response": response}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8100)



import os
from pathlib import Path
from typing import Dict, Optional, List
import openai
from fastapi import Depends, FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, EmailStr
from dotenv import load_dotenv
from passlib.context import CryptContext
# import asyncio

# Load environment variables
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")  # MongoDB connection string
DATABASE_NAME = "chatbot_db"

# Initialize MongoDB connection
client = AsyncIOMotorClient(MONGO_URI)
db = client[DATABASE_NAME]
users_collection = db["users"]

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

app = FastAPI()

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserRegister(BaseModel):
    email: EmailStr
    password: str
    
class delete(BaseModel):
    filename: str

UPLOAD_DIR = Path("uploads")
VECTOR_DIR = Path("vector_stores")
ALLOWED_EXTENSIONS = {".pdf", ".txt"}

UPLOAD_DIR.mkdir(exist_ok=True)
VECTOR_DIR.mkdir(exist_ok=True)

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

class DeleteFileRequest(BaseModel):
    filename: str

vector_store_cache: Dict[str, FAISS] = {}


# async def verify_connection():
#     try:
#         client = AsyncIOMotorClient(MONGO_URI)
#         db = client[DATABASE_NAME]
        
#         # Run a simple command to check connection
#         server_info = await db.command("ping")
#         print("✅ MongoDB Connection Successful!", server_info)
        
#     except Exception as e:
#         print(f"❌ MongoDB Connection Failed: {e}")


# asyncio.run(verify_connection())

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


@app.post("/register")
async def register(user: UserRegister):
    existing_user = await users_collection.find_one({"email": user.email})
    
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_password = pwd_context.hash(user.password)
    
    new_user = {"email": user.email, "password": hashed_password}
    await users_collection.insert_one(new_user)
    
    return {"message": "User registered successfully"}


@app.post("/login")
async def login(user: UserLogin):
    existing_user = await users_collection.find_one({"email": user.email})
    
    if not existing_user or not pwd_context.verify(user.password, existing_user["password"]):
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8100)
