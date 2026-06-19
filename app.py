import os
import sys
import streamlit as st
from faster_whisper import WhisperModel
import tempfile

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from streamlit_mic_recorder import mic_recorder

# 1. Page Configuration
st.set_page_config(page_title="Agentic RAG Assistant", page_icon="🤖", layout="centered")
st.title("🤖 Agentic RAG Assistant")
st.caption("Ask questions about your ingested local documents.")

# 2. Load Resources & Cache them to prevent reloading on every click
@st.cache_resource
def initialize_rag_agent():
    load_dotenv()  
    
    try:
        embeddings = OllamaEmbeddings(
            model=os.getenv("EMBEDDING_MODEL"),
        )

        vector_store = Chroma(
            collection_name=os.getenv("COLLECTION_NAME"),
            embedding_function=embeddings,
            persist_directory=os.getenv("DATABASE_LOCATION"), 
        )

        llm = init_chat_model(
            os.getenv("CHAT_MODEL"),
            model_provider=os.getenv("MODEL_PROVIDER"),
            temperature=0
        )
        
        @tool
        def retrieve_knowledge(query: str) -> str:
            """Search and retrieve information from the local knowledge base regarding the query."""
            retrieved_docs = vector_store.similarity_search(query, k=5)

            serialized = ""
            for doc in retrieved_docs:
                title = doc.metadata.get("title", "Unknown Source")
                filename = doc.metadata.get("source", "Unknown File")
                serialized += f"Document Title: {title} (File: {filename})\nContent: {doc.page_content}\n\n"

            return serialized

        tools = [retrieve_knowledge]

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a personal document assistant. Use the 'retrieve_knowledge' tool to look up facts "
                "from the vector store before answering questions. Always synthesize your answer "
                "concisely based ONLY on the retrieved documents. Do not use emojis in your response.\n\n"
                "For every piece of information you provide, state its source title and filename.\n"
                "If the retrieved information doesn't contain the answer, say 'I don't know'."
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        
        return agent_executor

    except Exception as e:
        st.error(f"Error initializing resources: {e}")
        st.info("Tip: Verify that your local Ollama instance is running and your .env variables are correct.")
        sys.exit(1)

@st.cache_resource
def load_whisper_model():
    """
    Loads Faster-Whisper model (CPU optimized for Mac Intel).
    """
    model = WhisperModel(
        "tiny",
        device="cpu",
        compute_type="int8"
    )
    return model

# Initialize the agent
agent_executor = initialize_rag_agent()

whisper_model = load_whisper_model()

# 3. Handle Session State for Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Internal history for the LLM
if "ui_messages" not in st.session_state:
    st.session_state.ui_messages = []   # History formatted for Streamlit UI
if "last_audio_hash" not in st.session_state:
    st.session_state.last_audio_hash = None
if "pending_voice_text" not in st.session_state:
    st.session_state.pending_voice_text = None

# Sidebar Options
with st.sidebar:
    st.header("Settings & Tools")
    if st.button("Clear Chat History", type="primary"):
        st.session_state.chat_history = []
        st.session_state.ui_messages = []
        st.session_state.pending_voice_text = None
        st.rerun()
    
    st.divider()
    st.markdown(f"**Chat Model:** `{os.getenv('CHAT_MODEL')}`")
    st.markdown(f"**Embedding:** `{os.getenv('EMBEDDING_MODEL')}`")

# 4. Display Existing Messages
for message in st.session_state.ui_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

col_text, col_mic, col_send = st.columns([0.7, 0.15, 0.15])

if "pending_transcribed_text" in st.session_state:
    st.session_state.text_input_value = st.session_state.pending_transcribed_text
    del st.session_state.pending_transcribed_text

with col_text:
    if "text_input_value" not in st.session_state:
        st.session_state.text_input_value = ""

    text_input = st.text_input(
        "Query input",
        placeholder="Ask something...",
        label_visibility="collapsed",
        key="text_input_value"
    )

with col_mic:
    audio_record = mic_recorder(
        start_prompt="🎤",
        stop_prompt="⏹",
        key="mic",
        just_once=True
    )

with col_send:
    send_button = st.button("Send", use_container_width=True, type="primary")


# 5. TRANSCRIBE AUDIO

voice_text = None

if audio_record and "bytes" in audio_record:
    audio_bytes = audio_record["bytes"]
    audio_hash = hash(audio_bytes)

    if audio_hash != st.session_state.last_audio_hash:
        st.session_state.last_audio_hash = audio_hash

        with st.spinner("Transcribing..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                    f.write(audio_bytes)
                    tmp_path = f.name

                segments, _ = whisper_model.transcribe(tmp_path)
                voice_text = " ".join(s.text for s in segments).strip()

                st.session_state["pending_transcribed_text"] = voice_text
                st.success("Transcribed (editable)")
                st.rerun()

                os.remove(tmp_path)

            except Exception as e:
                st.error(f"Transcription error: {e}")
                st.session_state.pending_voice_text = None


# 6. FINAL QUERY RESOLUTION
user_question = None

if send_button:
    if text_input and text_input.strip():
        user_question = text_input.strip()

# 7. RUN AGENT
if user_question:

    st.chat_message("user").markdown(user_question)

    st.session_state.ui_messages.append(
        {"role": "user", "content": user_question}
    )

    with st.chat_message("assistant"):
        with st.status("Thinking...", expanded=False):
            try:
                result = agent_executor.invoke({
                    "input": user_question,
                    "chat_history": st.session_state.chat_history
                })

                response = result["output"]

            except Exception as e:
                response = f"Error: {e}"

        st.markdown(response)

    st.session_state.ui_messages.append(
        {"role": "assistant", "content": response}
    )

    st.session_state.chat_history.append(HumanMessage(content=user_question))
    st.session_state.chat_history.append(AIMessage(content=response))