import uuid
from typing import List
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage ,HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
import re

from app.llm import extractor_llm

#******state********
class MemoryItem(BaseModel):
    text: str = Field(description="Atomic user memory")
    is_new: bool = Field(description="True if new, false if duplicate")

class MemoryDecision(BaseModel):
    should_write: bool
    memories: List[MemoryItem] = Field(default_factory=list)

#*********** prompt *******************
SYSTEM_PROMPT_TEMPLATE = """You are a helpful assistant with memory capabilities.
If user-specific memory is available, use it to personalize 
your responses based on what you know about the user.

Your goal is to provide relevant, friendly, and tailored 
assistance that reflects the user’s preferences, context, and past interactions.

If the user’s name or relevant personal context is available, always personalize your responses by:
    – Always Address the user by name (e.g., "Sure, shashank...") when appropriate
    – Referencing known projects, tools, or preferences (e.g., "your MCP server python based project")
    – Adjusting the tone to feel friendly, natural, and directly aimed at the user

Avoid generic phrasing when personalization is possible.

Use personalization especially in:
    – Greetings and transitions
    – Help or guidance tailored to tools and frameworks the user uses
    – Follow-up messages that continue from past context

Always ensure that personalization is based only on known user details and not assumed.

In the end suggest 3 relevant further questions based on the current response and user profile

The user’s memory (which may be empty) is provided as: {user_details_content}
"""


MEMORY_PROMPT = """You manage long-term memory about the user.
Existing similar memories:
{user_details_content}

From the user's latest message, extract only durable facts: name, preferences,
projects, tools, goals. Ignore questions, small talk and one-off requests.
Write each memory as one short standalone sentence. Set is_new=false if it is
already covered by existing memories. If nothing is worth saving, set should_write=false."""

memory_extractor = extractor_llm.with_structured_output(MemoryDecision)


def memory_ns(config:RunnableConfig):
    raw=config.get("configurable",{}).get("user_id") or "guest"
    uid= re.sub(r"[^a-zA-Z0-9-]","_",str(raw))[:64]
    return ("user", uid, "details")

def human_texts(messages, n=2):
    texts = [m.content for m in messages if m.type == "human" and isinstance(m.content, str)]
    return texts[-n:]


HINTS = re.compile(
    r"\b(my name|call me|i am|i'm|i live|i work|i study|i like|i love|i prefer|"
    r"i use|i have|i want|my favou?rite|my project|my goal|remember|mera naam|meri|mujhe)\b",
    re.I,
)

def looks_memorable(text: str) -> bool:
    t = (text or "").strip()
    return len(t.split()) >= 4 and bool(HINTS.search(t))

def recall_facts(store: BaseStore, config: RunnableConfig, messages, k=4, min_score=0.25):
    if not config.get("configurable", {}).get("use_memory", True):
        return []
    query = " ".join(human_texts(messages, 2))
    if not query.strip():
        return []
    try:
        hits = store.search(memory_ns(config), query=query, limit=k)
    except Exception as e:
        print(f"[memory] recall failed: {e}")
        return []
    return [h.value["data"] for h in hits if h.score is None or h.score >= min_score]



def remember_node(state, config: RunnableConfig, *, store: BaseStore):
    if not config.get("configurable", {}).get("use_memory", True):
        return {}
    text = (human_texts(state["messages"], 1) or [""])[0]
    if not text.strip():
        return {}
    ns = memory_ns(config)
    try:
        similar = store.search(ns, query=text, limit=5)
        existing = "\n".join(f"- {s.value['data']}" for s in similar) or "(empty)"
        decision = memory_extractor.invoke([
            SystemMessage(content=MEMORY_PROMPT.format(user_details_content=existing)),
            HumanMessage(content=text),
        ])
        if decision.should_write:
            for mem in decision.memories:
                if mem.is_new and mem.text.strip():
                    store.put(ns, str(uuid.uuid4()), {"data": mem.text.strip()})
    except Exception as e:
        print(f"[memory] save failed: {e}")
    return {}





    