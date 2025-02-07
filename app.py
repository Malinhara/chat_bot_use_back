

import os
from pathlib import Path
from typing import Dict, Optional,List

import openai
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
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
