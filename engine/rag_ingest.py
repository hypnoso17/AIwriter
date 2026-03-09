from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIR = ROOT / "knowledge"
CHROMA_DIR = ROOT / "data" / "chroma"

def load_docs():
    docs = []

    for path in KNOWLEDGE_DIR.rglob("*"):
        if path.suffix.lower() in [".md", ".txt"] and "story" in path.name.lower():
            loader = TextLoader(str(path), encoding="utf-8")
            file_docs = loader.load()

            for d in file_docs:
                d.metadata["source"] = path.name
                d.metadata["type"] = "story"
            
            docs.extend(file_docs)

    return docs

def main():
    docs = load_docs()
    print(f"loaded docs = {len(docs)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    chunks = splitter.split_documents(docs)
    print(f"chunks = {len(chunks)}")

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name="script_kb",
    )

    db.persist()
    print(f"done. vectordb saved to: {CHROMA_DIR}")

if __name__ == "__main__":
    main()