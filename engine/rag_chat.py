from pathlib import Path
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma

ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = ROOT / "data" / "chroma"

embeddings = OllamaEmbeddings(model="nomic-embed-text")

db = Chroma(
    persist_directory=str(CHROMA_DIR),
    embedding_function=embeddings,
    collection_name="script_kb",
)

retriever = db.as_retriever(search_kwargs={"k": 3})

llm = ChatOllama(model="qwen2.5:7b-instruct", temperature=0.3)

query = "写一个第一集开场，突出主角讨厌撒谎的性格"

docs = retriever.invoke(query)

context = "\n\n".join([d.page_content for d in docs])

prompt = f"""
你是专业编剧。

以下是世界观和人物设定，请严格遵守，不要违背设定：

{context}

任务：
根据上述设定，写一个电视剧第一集的开场场景。
要求：
1. 突出主角李远“讨厌撒谎和伪造证据”的性格
2. 风格偏现实主义悬疑
3. 不要出现超自然元素
4. 输出用中文
"""

response = llm.invoke(prompt)

print("\n===== 检索结果 =====\n")
for i, d in enumerate(docs, 1):
    print(f"[{i}] source={d.metadata}")
    print(d.page_content)
    print()

print("\n===== AI生成 =====\n")
print(response.content)