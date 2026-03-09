from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = ROOT / "data" / "chroma"


def build_retriever(k: int = 3):
    """Create a retriever backed by the script_kb Chroma collection."""
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name="script_kb",
    )
    return db.as_retriever(search_kwargs={"k": k})


def retrieve_context(query: str, k: int = 3, fallback_context: str = ""):
    """Retrieve relevant docs and merge them into a context string.

    If retrieval returns no docs, fallback_context is returned so the caller can
    still keep continuity with previously useful context.
    """
    retriever = build_retriever(k=k)
    docs = retriever.invoke(query)

    if not docs:
        return docs, fallback_context

    context = "\n\n".join(d.page_content for d in docs)
    return docs, context
