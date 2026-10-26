from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, Docx2txtLoader,
)

ENCODINGS = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]


def _text_loader(path):
    class _Loader:
        def load(self):
            last_err = None
            for enc in ENCODINGS:
                try:
                    return TextLoader(path, encoding=enc).load()
                except Exception as e:
                    last_err = e
            raise last_err
    return _Loader()


LOADERS = {
    ".pdf": PyPDFLoader,
    ".txt": _text_loader,
    ".md": _text_loader,
    ".docx": Docx2txtLoader,
}


def load_file(path: str):
    ext = Path(path).suffix.lower()
    if ext not in LOADERS:
        raise ValueError(f"Unsupported file type: {ext}")
    docs = LOADERS[ext](path).load()
    for d in docs:
        d.metadata["source"] = Path(path).name
    return docs
