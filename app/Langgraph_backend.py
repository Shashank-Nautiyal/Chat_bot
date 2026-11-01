from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.store.base import BaseStore
from langchain_core.messages import trim_messages
from app.summary import needs_summary, summarize_node, MAX_SEND
from app.llm import llm
from app.tools import tools
from app.database import checkpointer, store
from app.memory import (
    SYSTEM_PROMPT_TEMPLATE, recall_facts, remember_node, human_texts, looks_memorable,
)

class chatbot_state(TypedDict):
    messages : Annotated[list[BaseMessage],add_messages]
    summary: str
    summarized_upto: int

# ******* tools ******

llm_with_tools=llm.bind_tools(tools)


# ***** nodes **********

def chat_node(state: chatbot_state, config: RunnableConfig, *, store: BaseStore):
    facts = recall_facts(store, config, state["messages"])
    content = SYSTEM_PROMPT_TEMPLATE.format(user_details_content="\n".join(facts) or "(empty)")
    if state.get("summary"):
        content += f"\n\nSummary of the earlier conversation:\n{state['summary']}"

    recent = state["messages"][state.get("summarized_upto", 0):]
    recent = trim_messages(
        recent, strategy="last", token_counter=len,
        max_tokens=MAX_SEND, start_on="human", include_system=False,
    )
    response = llm_with_tools.invoke([SystemMessage(content=content)] + recent)
    return {"messages": [response]}

tool_node = ToolNode(tools)

def after_final(state: chatbot_state):
    text = (human_texts(state["messages"], 1) or [""])[0]
    return "remember_node" if looks_memorable(text) else END

def route_after_chat(state: chatbot_state):
    if getattr(state["messages"][-1], "tool_calls", None):
        return "tools"
    if needs_summary(state):
        return "summarize_node"
    return after_final(state)

# *********** graph *************

graph = StateGraph(chatbot_state)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
graph.add_node("remember_node", remember_node)
graph.add_node("summarize_node", summarize_node)

graph.add_edge(START, "chat_node")
graph.add_edge("tools", "chat_node")
graph.add_conditional_edges("chat_node", route_after_chat,
                            ["tools", "summarize_node", "remember_node", END])
graph.add_conditional_edges("summarize_node", after_final, ["remember_node", END])
graph.add_edge("remember_node", END)

chat_bot = graph.compile(checkpointer=checkpointer, store=store)

