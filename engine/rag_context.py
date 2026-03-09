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


def retrieve_context(query: str, k: int = 3, fallback_context: str = "", threshold: float = 0.5):
    retriever = build_retriever(k=k)

    docs_and_scores = retriever.vectorstore.similarity_search_with_score(query, k=k)

    filtered_docs = []
    for doc, score in docs_and_scores:
        if score < threshold:
            filtered_docs.append(doc)

    if not filtered_docs:
        return [], fallback_context

    context = "\n\n".join(d.page_content for d in filtered_docs)
    return filtered_docs, context