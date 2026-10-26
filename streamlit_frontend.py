import streamlit as st
from app.Langgraph_backend import chat_bot
from app.database import retrieve_all_thread
from langchain_core.messages import HumanMessage , AIMessage ,ToolMessage
import uuid
from app.rag.vector_store import ingest_bytes

# ********* adding css to frontend **** **** ****
st.set_page_config(page_title="LangGraph Chatbot", page_icon="Chat_bot", layout="wide")

st.markdown("""
<style>
/* primary buttons (New Chat + active chat) */
section[data-testid="stSidebar"] button[kind="primary"] {
    background-color:#58A6A7;
    border-color: #0C6769;
    color: white;
}
section[data-testid="stSidebar"] button[kind="primary"]:hover {
    background-color: #9CEDB6;
    border-color: #0C6769;
}
/* left-align the label */
section[data-testid="stSidebar"] .stButton > button div,
section[data-testid="stSidebar"] .stButton > button p {
    justify-content: flex-start;
    text-align: left;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

st.header("Chat bot")


# *******  helper function  ******

def new_thread_id():
    return str(uuid.uuid4())

def start_new_chat():
    tid=new_thread_id()
    st.session_state["thread_id"]=tid
    st.session_state["chat_threads"].append(tid)
    st.session_state["thread_titles"][tid]="New Chat"
    st.session_state["message_history"]=[]

def load_conversation(tid):  
    state = chat_bot.get_state(config={"configurable": {"thread_id": tid}})
    msgs=state.values.get("messages",[])
    return [
        {"role": "user" if m.type == "human" else "assistant", "content": m.content}
        for m in msgs
    ]

def make_title(text, limit=30):
    text = " ".join(text.split()) 
    return text if len(text) <= limit else text[:limit].rstrip() + "…"

def get_title(tid):
    titles = st.session_state["thread_titles"]
    if tid not in titles:
        state = chat_bot.get_state(config={"configurable": {"thread_id": tid}})
        msgs = state.values.get("messages", [])
        first = next((m.content for m in msgs if m.type == "human"), None)
        titles[tid] = make_title(first) if first else "New Chat"
    return titles[tid]

#  ****** session state initialization ********

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_thread()
    st.session_state["thread_titles"] = {}
if "thread_id" not in st.session_state:
    start_new_chat()



def switch_thread(tid):
    st.session_state["thread_id"] = tid
    st.session_state["message_history"] = load_conversation(tid)



# *********** input tab**************

prompt = st.chat_input(
    "Type here or attach a file",
    accept_file="multiple",
    file_type=["pdf", "txt", "md", "docx"],
)

user_input = prompt.text if prompt else None
uploaded_files = prompt.files if prompt else []

if prompt:
    tid = st.session_state["thread_id"]
    if st.session_state["thread_titles"].get(tid) == "New Chat":
        first = user_input or (uploaded_files[0].name if uploaded_files else "")
        if first:
            st.session_state["thread_titles"][tid] = make_title(first)


#  *******   sidebar ********

st.sidebar.title("💬 LangGraph Chatbot")
st.sidebar.button("New Chat", icon="➕", on_click=start_new_chat,
                  use_container_width=True, type="primary")

st.sidebar.divider()
st.sidebar.caption("RECENT CHATS")

for tid in reversed(st.session_state["chat_threads"]):
    is_active = tid == st.session_state["thread_id"]
    st.sidebar.button(
        get_title(tid),
        key=f"thread_{tid}",
        icon="⚪" if is_active else "💬",
        type="primary" if is_active else "secondary",
        use_container_width=True,
        on_click=switch_thread,
        args=(tid,),
    )

# ****** render_message_history ********

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ***** calling langGraph backend to get llm response *****
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage  # top of file

# ... at the bottom, replacing the old `if user_input:` block ...

if prompt:
    thread_id = st.session_state["thread_id"]

    #  index attached files into this chat's vector store

    for f in uploaded_files:
        with st.spinner(f"Indexing {f.name}..."):
            try:
                info = ingest_bytes(f.getvalue(), thread_id, f.name)
                note = f"📎 {info['filename']} indexed ({info['chunks']} chunks)"
            except Exception as e:
                note = f"⚠️ Could not index {f.name}: {e}"
        st.session_state["message_history"].append({"role": "user", "content": note})
        with st.chat_message("user"):
            st.markdown(note)

    #  send the text to the LLM, if there is any
    
    if user_input:
        st.session_state["message_history"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        config = {
            "configurable": {"thread_id": thread_id},
            "metadata": {"thread_id": thread_id},
            "run_name": "chat_turn",
        }

        with st.chat_message("assistant"):
            status_holder = {"box": None}

            def ai_only_stream():
                for message_chunk, metadata in chat_bot.stream(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=config,
                    stream_mode="messages",
                ):
                    # show a status box while a tool runs
                    if isinstance(message_chunk, ToolMessage):
                        tool_name = getattr(message_chunk, "name", "tool")
                        label = f"🔧 Using `{tool_name}` …"
                        if status_holder["box"] is None:
                            status_holder["box"] = st.status(label, expanded=True)
                        else:
                            status_holder["box"].update(label=label, state="running", expanded=True)

                    # stream only assistant text (skip empty tool-call chunks)
                    if isinstance(message_chunk, AIMessage) and message_chunk.content:
                        yield message_chunk.content

            ai_message = st.write_stream(ai_only_stream())

            if status_holder["box"] is not None:
                status_holder["box"].update(label="✅ Tool finished", state="complete", expanded=False)

        st.session_state["message_history"].append({"role": "assistant", "content": ai_message})
    




