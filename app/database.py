from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3


conn= sqlite3.connect("chatbot.db",check_same_thread=False)
checkpointer=SqliteSaver(conn=conn)


def retrieve_all_thread():
    all_thread=set()
    for checkpoint in (checkpointer.list(None)):
        all_thread.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_thread)
