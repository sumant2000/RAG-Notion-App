import sys
import os
import yaml
import streamlit as st
from streamlit_option_menu import option_menu
import json
from dotenv import load_dotenv

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embedding import Embedding, JSONHandler
from llm import RAGApp

# Load environment variables
load_dotenv()

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User Interface
st.set_page_config(page_title="RAG Desktop App", layout="wide")

# Custom CSS
st.markdown(
    """
    <style>
    .stMarkdown {color: #000000 !important;}
    .stTextArea textarea {color: #000000 !important;}
    </style>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css" rel="stylesheet">
    """,
    unsafe_allow_html=True,
)

# Load config
with open('../config.yaml') as file:
    config = yaml.safe_load(file)

# Initialize RAG app
rag_app = RAGApp(
    embedding_model=config["embedding"]["model"],
    llm_model=config["llm"]["model"],
    max_token=config["llm"]["max_token"],
    temperature=config["llm"]["temperature"]
)

# Sidebar
with st.sidebar:
    selected = option_menu(
        menu_title="Menu",
        options=["Chat", "Settings"],
        icons=["chat", "gear"],  # Font Awesome
        menu_icon="menu-down",
        default_index=0,
    )

# Initialize session state for temperature
if "temperature" not in st.session_state:
    st.session_state.temperature = config["llm"]["temperature"]  # default temperature

# Chat Layout
if selected == "Chat":

    st.title("Chat with RAG Bot")

    # First loading message
    if not st.session_state.chat_history:
        st.session_state.chat_history.append({"user": None, "bot": "How can I help you?"})

    # Display conversation history
    chat_container = st.container()
    with chat_container:
        for chat in st.session_state.chat_history:
            if chat["user"]:
                st.markdown(f'<div style="color: #000000;"><strong>You:</strong> {chat["user"]}</div>', unsafe_allow_html=True)
            if chat["bot"]:
                st.markdown(
                    f"""
                    <div style='display: flex; align-items: flex-start; margin-bottom: 10px;'>
                        <div style='font-size: 24px; margin-right: 10px;'>
                            <i class="fa fa-robot" style="color: #555;"></i>  <!-- Robot-icon -->
                        </div>
                        <div style='background-color: #f0f0f0; padding: 10px; border-radius: 10px; color: #000000;'>
                            {chat["bot"]}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # User Input
    user_input = st.text_area(
        label="Placeholder Label", 
        placeholder="Type your question here...", 
        height=100, 
        label_visibility="collapsed"
    )

    send_button = st.button("Send", key="send_button")

    if send_button and user_input.strip():
        # Spinner
        with st.spinner("Generating response..."):
            
            # DEBUG:
            print(f"##### Passing temp to RAG: {st.session_state.temperature} #####")

            # RAGApp
            response = rag_app.get_response(user_input, search_type=config["llm"]["search_type"], 
                                            k=config["llm"]["k"], fetch_k=config["llm"]["fetch_k"], eval_mode=False)

        # Update chat history
        st.session_state.chat_history.append({"user": user_input, "bot": response})
        
        # Display the updated convo
        with chat_container:
            st.markdown(f'<div style="color: #000000;"><strong>You:</strong> {user_input}</div>', unsafe_allow_html=True)
            st.markdown(
                f"""
                <div style='display: flex; align-items: flex-start; margin-bottom: 10px;'>
                    <div style='font-size: 24px; margin-right: 10px;'>
                            <i class="fa fa-robot" style="color: #555;"></i>  <!-- Robot-icon -->
                    </div>
                    <div style='background-color: #f0f0f0; padding: 10px; border-radius: 10px; color: #000000;'>
                        {response}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

# Setting layout
elif selected == "Settings":
    st.title("Settings")
    st.markdown('<div style="color: #000000;">Here you can configure the app settings.</div>', unsafe_allow_html=True)
    st.markdown('<div style="color: #000000;">#### Adjust the creativity of responses:</div>', unsafe_allow_html=True)
    # Temperature slider
    st.session_state.temperature = st.slider(
        "Temperature (0.1-1.0):",   # label
        min_value=0.1,              # min
        max_value=1.0,              # max
        value=st.session_state.temperature,  # default temperature
        step=0.1                    # step
    )
