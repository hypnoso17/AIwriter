from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = ROOT / "data" / "chroma"

embeddings = OllamaEmbeddings(model="nomic-embed-text")

db = Chroma(
    persist_directory=str(CHROMA_DIR),
    embedding_function=embeddings,
    collection_name="script_kb",
)

query = "主角最不能接受什么行为？"
docs = db.similarity_search(query, k=2)

for i, d in enumerate(docs, 1):
    print(f"\n===== result {i} =====")
    print("source:", d.metadata)
    print(d.page_content)