import os
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.processing import process_document
import backend.vectorstore as vectorstore
from backend.rag import generate_rag_response

app = FastAPI(title="DevSynt RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path(__file__).resolve().parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


class QueryRequest(BaseModel):
    question: str


@app.get("/stats")
def get_dashboard_stats():
    total_chunks = vectorstore.count_chunks()
    files = list(UPLOAD_DIR.glob("*")) if UPLOAD_DIR.exists() else []
    return {
        "total_documents": len(files),
        "total_chunks": total_chunks,
        "documents": [f.name for f in files]
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        chunks = process_document(file_path)
        indexed_count = vectorstore.add_chunks(chunks)
        return {
            "status": "success",
            "filename": file.filename,
            "chunks_added": indexed_count
        }
    except Exception as e:
        if file_path.exists():
            os.remove(file_path)
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/chat")
def chat_endpoint(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    return generate_rag_response(request.question)