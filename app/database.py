from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from app.llm import get_memory_embeddings
import os

pool = ConnectionPool(
    conninfo=os.getenv("DATABASE_URL"),
    max_size=10,
    kwargs={"autocommit": True, "prepare_threshold": 0},
)
checkpointer = PostgresSaver(pool)
checkpointer.setup()   



_embeddings = get_memory_embeddings()
EMBED_DIMS = len(_embeddings.embed_query("dimension probe"))

def retrieve_all_thread(user_id=None):
    seen = []
    flt = {"user_id": user_id} if user_id else None
    for cp in checkpointer.list(None):         
        tid = cp.config["configurable"]["thread_id"]
        if tid not in seen:
            seen.append(tid)
    return seen[::-1]                         

DB_URI = os.getenv("DATABASE_URL")

store = PostgresStore(
    pool,
    index={"embed": _embeddings, "dims": EMBED_DIMS, "fields": ["data"]},
)
   
store.setup()
