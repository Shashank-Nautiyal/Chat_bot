from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.rag.loader import load_file

splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
            )

def text_splitter(docs):
   
    return splitter.split_documents(docs)
    
   





