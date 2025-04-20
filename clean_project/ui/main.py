import os
import streamlit as st
from streamlit_option_menu import option_menu
from dotenv import load_dotenv, set_key, find_dotenv
import json
import logging
from ui.rag_app import RAGApp
from ui.config import SIDEBAR_LOGO_PATH, MAIN_LOGO_PATH
from ui.document_processor import DocumentProcessor
from ui.notion_processor import NotionProcessor
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize session state
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.messages = []
    st.session_state.temperature = float(os.getenv("TEMPERATURE", "0.7"))
    st.session_state.notion_api_key = os.getenv("NOTION_API_KEY", "")
    st.session_state.notion_page_ids = os.getenv("NOTION_PAGE_IDS", "")
    st.session_state.integrations = None

# Initialize the RAG app
rag_app = RAGApp()
document_processor = DocumentProcessor()
notion_processor = NotionProcessor(document_processor=document_processor)

# Check if Notion integrations need to be loaded
if st.session_state.notion_api_key and st.session_state.integrations is None:
    try:
        integrations_result = notion_processor.list_integrations()
        if integrations_result.get("status") == "success":
            st.session_state.integrations = integrations_result.get("integrations", [])
    except Exception as e:
        logger.error(f"Failed to load Notion integrations: {str(e)}")

# Page title and configuration
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

# Function to update environment variables in .env file
def update_env_variable(key, value):
    """Update environment variable in .env file"""
    dotenv_path = find_dotenv()
    set_key(dotenv_path, key, value)
    os.environ[key] = value
    logger.info(f"Updated environment variable {key}")

# Custom CSS
st.markdown("""
<style>
    .stTextInput>div>div>input {
        border-radius: 10px;
    }
    .stButton>button {
        border-radius: 10px;
    }
    div.stButton > button:first-child {
        background-color: #4CAF50;
        color: white;
    }
    .main-header {
        font-size: 2.5rem !important;
        font-weight: bold;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #E3F2FD;
        border-left: 5px solid #2196F3;
    }
    .chat-message.assistant {
        background-color: #F1F8E9;
        border-left: 5px solid #8BC34A;
    }
    .chat-message .message-content {
        display: flex;
        flex-direction: row;
        align-items: flex-start;
    }
    .chat-message .avatar {
        min-width: 40px;
        margin-right: 1rem;
    }
    .chat-message .message {
        flex-grow: 1;
    }
    .sources-dropdown {
        margin-top: 0.5rem;
        border-top: 1px solid #ccc;
        padding-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    # Logo in sidebar
    if os.path.exists(SIDEBAR_LOGO_PATH):
        st.image(SIDEBAR_LOGO_PATH, width=100)
    
    # Navigation menu
    selected = option_menu(
        "Main Menu",
        ["Chat", "Settings", "About"],
        icons=["chat-dots-fill", "gear-fill", "info-circle-fill"],
        menu_icon="list",
        default_index=0,
    )

# Main content area
if selected == "Chat":
    # Logo and title
    if os.path.exists(MAIN_LOGO_PATH):
        col1, col2 = st.columns([1, 5])
        with col1:
            st.image(MAIN_LOGO_PATH, width=80)
        with col2:
            st.markdown("<h1 class='main-header'>RAG Chatbot</h1>", unsafe_allow_html=True)
            st.markdown("Ask me anything about the documents in my knowledge base.")
    else:
        st.markdown("<h1 class='main-header'>RAG Chatbot</h1>", unsafe_allow_html=True)
        st.markdown("Ask me anything about the documents in my knowledge base.")
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display chat messages
        for message in st.session_state.messages:
            role = message["role"]
            content = message["content"]
            with st.chat_message(role):
                st.write(content)
                
                # If this is an assistant message with sources, display them
                if role == "assistant" and "source_documents" in message:
                    source_docs = message.get("source_documents", [])
                    if source_docs:
                        with st.expander("View Sources"):
                            for i, doc in enumerate(source_docs):
                                st.markdown(f"**Source {i+1}:**")
                                st.markdown(doc["content"])
                                if "metadata" in doc and doc["metadata"]:
                                    st.markdown("**Metadata:**")
                                    st.json(doc["metadata"])
                                st.markdown("---")
        
        # Chat input
        if prompt := st.chat_input("Enter your question..."):
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
            
            # Get response from RAG app
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = rag_app.get_response(prompt)
                    
                    # Display assistant message
                    st.write(response["response"])
                    
                    # Add sources as expandable section if available
                    if "source_documents" in response and response["source_documents"]:
                        with st.expander("View Sources"):
                            for i, doc in enumerate(response["source_documents"]):
                                st.markdown(f"**Source {i+1}:**")
                                st.markdown(doc["content"])
                                if "metadata" in doc and doc["metadata"]:
                                    st.markdown("**Metadata:**")
                                    st.json(doc["metadata"])
                                st.markdown("---")
            
            # Add assistant message to chat
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response["response"],
                "source_documents": response.get("source_documents", [])
            })
    
    # Clear chat button
    col1, col2 = st.columns([5, 1])
    with col2:
        if st.button("Clear Chat"):
            st.session_state.messages = []
            rag_app.clear_chat_history()
            st.experimental_rerun()

elif selected == "Settings":
    st.markdown("<h1 class='main-header'>Settings</h1>", unsafe_allow_html=True)
    st.markdown("Configure your RAG application settings")
    
    # Temperature setting
    st.subheader("Model Settings")
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.temperature,
        step=0.1,
        help="Controls the randomness of the AI's responses. Lower values make responses more focused and deterministic."
    )
    
    if st.button("Save Temperature"):
        result = rag_app.set_temperature(temperature)
        if result["status"] == "success":
            st.session_state.temperature = temperature
            update_env_variable("TEMPERATURE", str(temperature))
            st.success(f"Temperature set to {temperature}")
        else:
            st.error(result["message"])
    
    # Notion integration settings
    st.subheader("Notion Integration")
    
    # Display existing integrations if API key is available
    if st.session_state.notion_api_key:
        with st.spinner("Checking Notion integrations..."):
            integrations = rag_app.list_notion_integrations()
            
            if integrations.get("status") == "success":
                st.success("Notion API key is valid")
                if "integrations" in integrations and integrations["integrations"]:
                    st.markdown("**Your Notion Integrations:**")
                    for integration in integrations["integrations"]:
                        st.markdown(f"- {integration.get('name', 'Unnamed Integration')}")
                else:
                    st.info("No integrations found with current API key")
            else:
                st.error(f"Notion API key error: {integrations.get('message', 'Unknown error')}")
                if "help" in integrations:
                    st.markdown(integrations["help"])
    
    # Notion API key input
    notion_api_key = st.text_input(
        "Notion API Key",
        type="password",
        value=st.session_state.notion_api_key,
        help="Your Notion integration secret key. Create one at https://www.notion.so/my-integrations"
    )
    
    # Notion page IDs input
    notion_page_ids = st.text_area(
        "Notion Page IDs",
        value=st.session_state.notion_page_ids,
        help="Comma-separated list of Notion page IDs or URLs to load into the knowledge base"
    )
    
    # Save Notion settings button
    if st.button("Save Notion Settings"):
        if notion_api_key != st.session_state.notion_api_key:
            update_env_variable("NOTION_API_KEY", notion_api_key)
            st.session_state.notion_api_key = notion_api_key
        
        if notion_page_ids != st.session_state.notion_page_ids:
            update_env_variable("NOTION_PAGE_IDS", notion_page_ids)
            st.session_state.notion_page_ids = notion_page_ids
        
        st.success("Notion settings saved")
        st.info("Reloading application with new settings...")
        st.experimental_rerun()
    
    # Load Notion pages section
    if st.session_state.notion_api_key:
        st.subheader("Load Notion Page")
        page_id = st.text_input(
            "Notion Page ID or URL",
            help="Enter the full URL or ID of a Notion page to load it into the knowledge base"
        )
        
        if st.button("Load Page"):
            with st.spinner("Loading Notion page..."):
                result = rag_app.load_notion_page(page_id)
                
                if result["status"] == "success":
                    st.success(result["message"])
                    if "chunks_added" in result:
                        st.info(f"Added {result['chunks_added']} chunks to the knowledge base")
                else:
                    st.error(f"Error: {result.get('message', 'Unknown error')}")
                    if "help" in result:
                        st.markdown(result["help"])

elif selected == "About":
    st.markdown("<h1 class='main-header'>About</h1>", unsafe_allow_html=True)
    st.markdown("""
    ## RAG Chatbot
    
    This application uses Retrieval Augmented Generation (RAG) to provide accurate answers to your questions
    based on a knowledge base of documents.
    
    ### Features:
    
    - Ask questions about documents in the knowledge base
    - Integrates with Notion to load and process pages
    - Provides source references for generated answers
    - Customizable model settings
    
    ### How to use:
    
    1. Go to the Chat tab to ask questions
    2. Configure settings in the Settings tab
    3. To add Notion pages, enter your API key and page IDs in Settings
    
    ### Technologies used:
    
    - Streamlit for the user interface
    - LangChain for document processing
    - Google Gemini for text generation
    - ChromaDB for vector storage
    
    Built with 💻 using Python
    """)
