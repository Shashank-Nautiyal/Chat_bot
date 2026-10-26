from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
import os
from llama_cpp import Llama

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7,
)

import llama_cpp
from langchain_core.embeddings import Embeddings

class GGUFEmbeddings(Embeddings):
    def __init__(self, model_path: str, n_ctx: int = 2048):
        self.llm = llama_cpp.Llama(
            model_path=model_path,
            embedding=True,
            n_ctx=n_ctx,
            pooling_type=llama_cpp.LLAMA_POOLING_TYPE_LAST,  # Qwen3-Embedding uses last-token pooling
            verbose=False,
        )

    def _embed(self, text: str):
        vec = self.llm.embed(text)
        if vec and isinstance(vec[0], list):   # token-level output -> take last token
            vec = vec[-1]
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        return [x / norm for x in vec]         # normalize for similarity search

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


embedding_model = GGUFEmbeddings(r"C:\Users\shash\.cache\huggingface\hub\models--Qwen--Qwen3-Embedding-4B-GGUF\snapshots\f4602530db1d980e16da9d7d3a70294cf5c190be\Qwen3-Embedding-4B-Q4_K_M.gguf")





