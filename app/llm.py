import os
from functools import lru_cache
from dotenv import load_dotenv
load_dotenv()
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.embeddings import Embeddings

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.7)
extractor_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "minilm").lower()


class GGUFEmbeddings(Embeddings):
    def __init__(self, model_path: str, n_ctx: int = 2048):
        import llama_cpp                      # lazy: only needed for this option
        self.llm = llama_cpp.Llama(
            model_path=model_path, embedding=True, n_ctx=n_ctx,
            pooling_type=llama_cpp.LLAMA_POOLING_TYPE_LAST, verbose=False,
        )

    def _embed(self, text: str):
        vec = self.llm.embed(text)
        if vec and isinstance(vec[0], list):
            vec = vec[-1]
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        return [x / norm for x in vec]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    if EMBEDDING_PROVIDER == "llamacpp":
        path = os.getenv("GGUF_EMBED_PATH")
        if not path:
            raise RuntimeError("Set GGUF_EMBED_PATH in .env to use llama.cpp embeddings")
        return GGUFEmbeddings(path)
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

@lru_cache(maxsize=1)
def get_memory_embeddings() -> Embeddings:
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )