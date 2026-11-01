from app.llm import extractor_llm

MAX_UNSUMMARIZED = 10   # summarize when more than this many messages are unsummarized
KEEP_LAST = 5           # recent messages kept verbatim after summarizing
MAX_SEND = 15           # hard cap on messages sent to the model

SUMMARY_PROMPT = """Update the running summary of a conversation.

Current summary:
{summary}

New messages:
{transcript}

Write the updated summary in under 150 words. Keep names, facts the user stated,
decisions, uploaded file names and open tasks. Drop greetings and filler.
Return only the summary."""


def needs_summary(state) -> bool:
    return len(state["messages"]) - state.get("summarized_upto", 0) > MAX_UNSUMMARIZED


def find_cut(messages, start, keep_last=KEEP_LAST):
    cut = len(messages) - keep_last
    while cut > start and messages[cut].type != "human":
        cut -= 1
    return max(cut, start)


def summarize_node(state):
    msgs = state["messages"]
    start = state.get("summarized_upto", 0)
    cut = find_cut(msgs, start)
    if cut <= start:
        return {}
    lines = [
        f"{m.type}: {m.content}"
        for m in msgs[start:cut]
        if m.type in ("human", "ai") and isinstance(m.content, str) and m.content.strip()
    ]
    try:
        out = extractor_llm.invoke(SUMMARY_PROMPT.format(
            summary=state.get("summary") or "(none)",
            transcript="\n".join(lines),
        ))
    except Exception as e:
        print(f"[summary] failed: {e}")
        return {}                      # keep history as is, retry next turn
    return {"summary": out.content.strip(), "summarized_upto": cut}