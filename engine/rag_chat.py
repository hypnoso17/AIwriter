from collections import deque

from langchain_ollama import ChatOllama

from rag_context import retrieve_context

llm = ChatOllama(model="qwen2.5:7b-instruct", temperature=0.3)

SYSTEM_PROMPT = """你是专业编剧助手。
请严格遵守给定的世界观和人物设定，不要违背设定。
输出使用中文。"""

RECENT_TURNS = 6
PINNED_NOTES_LIMIT = 20


def format_recent_dialogue(dialogue_history, max_turns: int = RECENT_TURNS):
    """Format recent conversation turns so they can be injected into the prompt."""
    if not dialogue_history:
        return "（暂无历史对话）"

    recent = list(dialogue_history)[-max_turns:]
    return "\n".join(f"{role}: {content}" for role, content in recent)


def format_pinned_notes(pinned_notes):
    """Format manually pinned notes discovered during conversation."""
    if not pinned_notes:
        return "（暂无会话固化依据）"
    return "\n".join(f"- {note}" for note in pinned_notes)


def build_prompt(
    user_query: str,
    retrieved_context: str,
    recent_dialogue: str,
    pinned_notes: str,
):
    return f"""
{SYSTEM_PROMPT}

【知识库检索上下文】
{retrieved_context}

【会话中固化的依据】
{pinned_notes}

【最近对话上下文】
{recent_dialogue}

【当前用户需求】
{user_query}

请结合知识库设定、会话固化依据与最近对话，给出连贯且符合设定的回复。
如果三者冲突：优先以知识库设定为准，其次是会话固化依据，再次是最近对话。
"""


def main():
    print("进入 RAG 对话模式，输入 exit 或 quit 退出。")
    print("可用命令：")
    print("  /pin <内容>   将当前会话里新发现的重要依据固化到记忆")
    print("  /pins         查看已固化依据")
    print("  /clearpins    清空已固化依据\n")

    dialogue_history = deque(maxlen=50)
    pinned_notes = deque(maxlen=PINNED_NOTES_LIMIT)
    last_good_context = ""

    while True:
        query = input("你：").strip()
        if query.lower() in {"exit", "quit"}:
            print("已退出。")
            break
        if not query:
            continue

        if query.startswith("/pin "):
            note = query[5:].strip()
            if note:
                pinned_notes.append(note)
                print(f"已固化依据：{note}\n")
            else:
                print("用法：/pin <内容>\n")
            continue

        if query == "/pins":
            print("\n===== 已固化依据 =====")
            for idx, note in enumerate(pinned_notes, 1):
                print(f"{idx}. {note}")
            if not pinned_notes:
                print("（空）")
            print()
            continue

        if query == "/clearpins":
            pinned_notes.clear()
            print("已清空固化依据。\n")
            continue

        docs, context = retrieve_context(query, k=3, fallback_context=last_good_context)
        if docs:
            last_good_context = context

        recent_dialogue = format_recent_dialogue(dialogue_history)
        pinned_notes_text = format_pinned_notes(pinned_notes)
        prompt = build_prompt(query, context, recent_dialogue, pinned_notes_text)

        response = llm.invoke(prompt)
        answer = response.content

        dialogue_history.append(("用户", query))
        dialogue_history.append(("助手", answer))

        print("\n===== 检索结果 =====\n")
        if docs:
            for i, d in enumerate(docs, 1):
                print(f"[{i}] source={d.metadata}")
                print(d.page_content)
                print()
        else:
            print("本轮未检索到新片段，已回退使用上一次有效检索上下文。\n")

        print("===== AI回复 =====\n")
        print(answer)
        print()


if __name__ == "__main__":
    main()
