from langchain_ollama import ChatOllama, OllamaEmbeddings

# 1) 测试 embedding
emb = OllamaEmbeddings(model="nomic-embed-text")
vec = emb.embed_query("测试一下本地 embedding 是否正常")
print("embedding dim =", len(vec))

# 2) 测试生成模型
llm = ChatOllama(model="qwen2.5:7b-instruct", temperature=0.2)
resp = llm.invoke("请用一句话介绍你自己。")
print("llm reply =", resp.content)