"""Vector database layer (ChromaDB): store chunks with metadata and search them."""
import os
from pathlib import Path
import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from dotenv import load_dotenv
from google import genai

load_dotenv(override=True)

BASE_DIR = Path(__file__).resolve().parent.parent
_db_path = Path(os.getenv("DATABASE_URL", "./chroma_db"))
DB_PATH = str(_db_path if _db_path.is_absolute() else BASE_DIR / _db_path)
COLLECTION_NAME = "documents"


class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self, api_key: str, model_name: str = "gemini-embedding-001"):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def name(self) -> str:
        """Required by ChromaDB to identify the custom embedding function."""
        return f"gemini_embedding_{self.model_name}"

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=[text]
            )
            embeddings.append(response.embeddings[0].values)
        return embeddings


_client = None
_collection = None


def get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=DB_PATH)
        api_key = os.getenv("API_KEY")
        
        embedding_fn = GeminiEmbeddingFunction(api_key=api_key)
        
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            configuration={"hnsw": {"space": "cosine"}},
            embedding_function=embedding_fn
        )
    return _collection


def add_chunks(chunks):
    """Embed and store chunks."""
    if not chunks:
        return 0
    get_collection().add(
        ids=[c["chunk_id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[{
            "doc_id": c["doc_id"],
            "doc_name": c["doc_name"],
            "page": c["page"] or 0,
            "chunk_index": c["chunk_index"],
        } for c in chunks],
    )
    return len(chunks)


def search(query, n_results=5, doc_ids=None):
    """Semantic search."""
    col = get_collection()
    if col.count() == 0:
        return []
    where = {"doc_id": {"$in": doc_ids}} if doc_ids else None
    res = col.query(query_texts=[query], n_results=min(n_results, col.count()), where=where)
    hits = []
    for i in range(len(res["ids"][0])):
        meta = res["metadatas"][0][i]
        hits.append({
            "chunk_id": res["ids"][0][i],
            "text": res["documents"][0][i],
            "doc_id": meta["doc_id"],
            "doc_name": meta["doc_name"],
            "page": meta["page"] or None,
            "distance": res["distances"][0][i],
        })
    return hits


def delete_document(doc_id):
    get_collection().delete(where={"doc_id": doc_id})


def count_chunks():
    return get_collection().count()


def reset_collection():
    """Delete everything from ChromaDB."""
    global _collection
    get_collection()
    _client.delete_collection(COLLECTION_NAME)
    _collection = None