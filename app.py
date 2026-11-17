import os
import streamlit as st
from typing import Annotated, TypedDict

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# ================= CONFIG =================
def _secret(key: str) -> str:
    """Streamlit Cloud -> st.secrets, local -> .env"""
    try:
        return st.secrets[key].strip()
    except Exception:
        from dotenv import load_dotenv
        load_dotenv()
        return os.getenv(key, "").strip()


GROQ_KEY       = _secret("groq_api_key")
ASTRA_TOKEN    = _secret("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_ENDPOINT = _secret("ASTRA_DB_API_ENDPOINT")


# ================= VECTOR STORE (cached) =================
@st.cache_resource
def get_retriever(collection_name: str = "langgraph", k: int = 4):
    from langchain_astradb import AstraDBVectorStore
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    store = AstraDBVectorStore(
        collection_name=collection_name,
        embedding=embeddings,
        api_endpoint=ASTRA_ENDPOINT,
        token=ASTRA_TOKEN,
    )
    return store.as_retriever(search_kwargs={"k": k})


# ================= GRAPH =================
class State(TypedDict):
    messages: Annotated[list, add_messages]
    context: str


@st.cache_resource
def build_graph(model_name: str, use_rag: bool):
    llm = ChatGroq(groq_api_key=GROQ_KEY, model_name=model_name)
    retriever = get_retriever() if use_rag else None

    def retrieve(state: State):
        query = state["messages"][-1].content
        docs = retriever.invoke(query)
        context = "\n\n---\n\n".join(d.page_content for d in docs)
        return {"context": context}

    def generate(state: State):
        msgs = list(state["messages"])
        if state.get("context"):
            msgs = [
                SystemMessage(content=(
                    "Answer the user's question using ONLY the context below. "
                    "If the answer is not in the context, say you don't know.\n\n"
                    f"Context:\n{state['context']}"
                ))
            ] + msgs
        return {"messages": llm.invoke(msgs)}

    builder = StateGraph(State)
    builder.add_node("generate", generate)

    if use_rag:
        builder.add_node("retrieve", retrieve)
        builder.add_edge(START, "retrieve")
        builder.add_edge("retrieve", "generate")
    else:
        builder.add_edge(START, "generate")

    builder.add_edge("generate", END)
    return builder.compile()


# ================= UI =================
st.set_page_config(page_title="LangGraph RAG Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 LangGraph RAG Chatbot")
st.caption("LangGraph + Groq + AstraDB")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Settings")

    model_choice = st.selectbox(
        "Model",
        [
            "qwen/qwen3.8-27b",              # best quality
            "openai/gpt-oss-safeguard-20b",  # cheapest + fastest
            "minimaxai/minimax-m2.7",        # contact sales
            "meta-models/muse-spark-open",   # contact sales
        ],
        index=0,
        help="qwen = best quality | gpt-oss-20b = cheapest & fastest",
    )

    use_rag = st.toggle(
        "Enable RAG (AstraDB)",
        value=True,
        help="Turn off to use as a plain chatbot without retrieval",
    )

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption(f"Messages: {len(st.session_state.get('messages', []))}")

# --- Session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Render history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat input ---
if prompt := st.chat_input("Type your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build message history for the graph
    history = []
    for m in st.session_state.messages:
        if m["role"] == "user":
            history.append(HumanMessage(content=m["content"]))
        else:
            history.append(AIMessage(content=m["content"]))

    # Call the graph
    graph = build_graph(model_choice, use_rag)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..." if not use_rag else "Searching context..."):
            try:
                result = graph.invoke({"messages": history})
                answer = result["messages"][-1].content
            except Exception as e:
                answer = f"❌ Error: `{e}`"
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})