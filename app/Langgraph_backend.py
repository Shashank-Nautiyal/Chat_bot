from langgraph.graph import StateGraph , START ,END
from langgraph.graph.message import add_messages
from typing import TypedDict , Annotated 
from langchain_core.messages import BaseMessage , HumanMessage

from app.llm import llm

from app.tools import  tools
from langgraph.prebuilt import ToolNode , tools_condition

from app.database import checkpointer

class chatbot_state(TypedDict):
    messages : Annotated[list[BaseMessage],add_messages]


# ******* tools ******

llm_with_tools=llm.bind_tools(tools)


# ***** nodes **********

def chat_node(state:chatbot_state):
    message=state["messages"]

    response=llm_with_tools.invoke(message)

    return {"messages":[response]}

tools = ToolNode(tools)


# *********** graph *************

graph=StateGraph(chatbot_state)

graph.add_node("chat_node",chat_node)
graph.add_node("tools",tools)

graph.add_edge(START,"chat_node")
graph.add_conditional_edges("chat_node",tools_condition)
graph.add_edge("tools", "chat_node")

chat_bot=graph.compile(checkpointer=checkpointer)

