from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from app.rag.vector_store import query as rag_query

@tool
def rag_tool(query: str, config: RunnableConfig) -> dict:
    """Search documents uploaded in this chat. Use for any question about uploaded files."""
    thread_id = config["configurable"]["thread_id"]
    docs = rag_query(query, thread_id=thread_id, k=4)
    if not docs:
        return {"error": "No document indexed for this chat. Upload a file first.", "query": query}
    return {
        "query": query,
        "context": [d.page_content for d in docs],
        "metadata": [d.metadata for d in docs],
        "source_files": sorted({d.metadata.get("source") for d in docs}),
    }