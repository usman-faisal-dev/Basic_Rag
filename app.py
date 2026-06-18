import os
import sys
import streamlit as streamlit
import streamlit as st
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent

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
            retrieved_docs = vector_store.similarity_search(query, k=10)

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

# Initialize the agent
agent_executor = initialize_rag_agent()

# 3. Handle Session State for Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Internal history for the LLM
if "ui_messages" not in st.session_state:
    st.session_state.ui_messages = []   # History formatted for Streamlit UI

# Sidebar Options
with st.sidebar:
    st.header("Settings & Tools")
    if st.button("Clear Chat History", type="primary"):
        st.session_state.chat_history = []
        st.session_state.ui_messages = []
        st.rerun()
    
    st.divider()
    st.markdown(f"**Chat Model:** `{os.getenv('CHAT_MODEL')}`")
    st.markdown(f"**Embedding:** `{os.getenv('EMBEDDING_MODEL')}`")

# 4. Display Existing Messages
for message in st.session_state.ui_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. Handle User Input
if user_question := st.chat_input("What would you like to know?"):
    
    # Display user question
    st.chat_message("user").markdown(user_question)
    st.session_state.ui_messages.append({"role": "user", "content": user_question})

    # Generate assistant response
    with st.chat_message("assistant"):
        # Streamlit status spinner handles the "Thinking & Searching" stage beautifully
        with st.status("Agent searching knowledge base...", expanded=False) as status:
            try:
                result = agent_executor.invoke({
                    "input": user_question, 
                    "chat_history": st.session_state.chat_history
                })
                ai_message = result["output"]
                status.update(label="Search complete!", state="complete", expanded=False)
                
            except Exception as e:
                status.update(label="Error processing request", state="error")
                ai_message = f"An error occurred: {e}\n\nPlease check your Ollama backend tool support configuration."

        # Display the response text
        st.markdown(ai_message)
        
    # Append to state tracking loops
    st.session_state.ui_messages.append({"role": "assistant", "content": ai_message})
    st.session_state.chat_history.append(HumanMessage(content=user_question))
    st.session_state.chat_history.append(AIMessage(content=ai_message))