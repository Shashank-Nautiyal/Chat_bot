from langchain_community.vectorstores import FAISS
from app.llm import embedding_model
from pathlib import Path
from app.rag.loader import load_file
from app.rag.splitter import text_splitter

VECTORDB_DIR = Path("data/vectordb")




def _index_path(thread_id: str) -> Path:
    return VECTORDB_DIR / str(thread_id)


def _load_index(thread_id: str):
    path = _index_path(thread_id)
    if not path.exists():
        return None
    return FAISS.load_local(str(path), embedding_model, allow_dangerous_deserialization=True)


import os, tempfile

def ingest(file_path: str, thread_id: str, filename: str | None = None) -> dict:
    docs = load_file(file_path)
    name = filename or Path(file_path).name
    for d in docs:
        d.metadata["source"] = name          
    chunks = text_splitter(docs)
    if not chunks:
        raise ValueError("No text found in file (scanned PDF?)")

    store = _load_index(thread_id)
    if store is None:
        store = FAISS.from_documents(chunks, embedding_model)
    else:
        store.add_documents(chunks)
    _index_path(thread_id).mkdir(parents=True, exist_ok=True)
    store.save_local(str(_index_path(thread_id)))
    return {"filename": name, "pages": len(docs), "chunks": len(chunks)}


def ingest_bytes(file_bytes: bytes, thread_id: str, filename: str) -> dict:
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as f:
        f.write(file_bytes)
        tmp = f.name
    try:
        return ingest(tmp, thread_id, filename)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass

def query(question: str, thread_id: str, k: int = 4):
    store = _load_index(thread_id)
    if store is None:
        return []
    return store.similarity_search(question, k=k)

