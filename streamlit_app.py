"""
Kavak Travel Assistant — Streamlit Frontend
============================================
Run with: python -m streamlit run streamlit_app.py
"""
import warnings
import os

# Suppress transformers warnings about missing optional dependencies
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*torchvision.*')

import streamlit as st
import uuid
from datetime import datetime

from main import chat

st.set_page_config(
    page_title="Kavak Travel Assistant",
    page_icon="✈️",
    layout="centered",
)

st.markdown(
    """
    <style>
    .stChatMessage { border-radius: 12px; }
    .main { background-color: #f0f4f8; }
    h1 { color: #0d47a1; }
    .subtitle { color: #1565c0; font-size: 0.9rem; margin-top: -10px; font-weight: 500; }
    .conversation-item {
        padding: 8px;
        border-radius: 8px;
        margin: 4px 0;
        cursor: pointer;
        border: 1px solid #90caf9;
    }
    .conversation-item:hover {
        background-color: #e3f2fd;
        border-color: #1976d2;
    }
    .conversation-item.selected {
        background-color: #bbdefb;
        border-color: #1565c0;
    }
    button[kind="primary"] {
        background-color: #1565c0 !important;
        color: white !important;
    }
    button[kind="primary"]:hover {
        background-color: #0d47a1 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state
if "conversations" not in st.session_state:
    st.session_state.conversations = {}
if "current_conversation_id" not in st.session_state:
    st.session_state.current_conversation_id = None
if "pending_input" not in st.session_state:
    st.session_state.pending_input = None

def create_new_conversation():
    """Create a new conversation and return its ID."""
    conversation_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    st.session_state.conversations[conversation_id] = {
        "title": f"Chat {len(st.session_state.conversations) + 1}",
        "messages": [],
        "history": [],
        "created_at": timestamp,
        "last_updated": timestamp
    }
    return conversation_id

def get_current_conversation():
    """Get the current conversation data."""
    if st.session_state.current_conversation_id and st.session_state.current_conversation_id in st.session_state.conversations:
        return st.session_state.conversations[st.session_state.current_conversation_id]
    return None

def save_current_conversation():
    """Save the current conversation state."""
    if st.session_state.current_conversation_id and st.session_state.current_conversation_id in st.session_state.conversations:
        conv = st.session_state.conversations[st.session_state.current_conversation_id]
        conv["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")

def switch_conversation(conversation_id):
    """Switch to a different conversation."""
    save_current_conversation()
    st.session_state.current_conversation_id = conversation_id
    st.rerun()

def generate_conversation_title(messages):
    """Generate a title for the conversation based on first user message."""
    for msg in messages:
        if msg["role"] == "user":
            # Take first 50 characters of the first user message
            title = msg["content"][:50]
            if len(msg["content"]) > 50:
                title += "..."
            return title
    return "New Chat"

# Create initial conversation if none exists
if not st.session_state.conversations:
    initial_id = create_new_conversation()
    st.session_state.current_conversation_id = initial_id

# Ensure we have a current conversation
if not st.session_state.current_conversation_id or st.session_state.current_conversation_id not in st.session_state.conversations:
    if st.session_state.conversations:
        st.session_state.current_conversation_id = list(st.session_state.conversations.keys())[0]
    else:
        initial_id = create_new_conversation()
        st.session_state.current_conversation_id = initial_id

current_conv = get_current_conversation()

st.title("✈️ Kavak Travel Assistant")
st.markdown(
    '<p class="subtitle">Ask about flights, visas, and travel policies</p>',
    unsafe_allow_html=True,
)
st.divider()

# Sidebar for conversation history
with st.sidebar:
    st.markdown("## 💬 Conversations")
    
    # New Chat button
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        new_id = create_new_conversation()
        switch_conversation(new_id)
    
    st.divider()
    
    # List all conversations
    for conv_id, conv_data in st.session_state.conversations.items():
        is_selected = conv_id == st.session_state.current_conversation_id
        title = conv_data.get("title", "Untitled")
        
        # Create clickable conversation item
        if st.button(
            f"{'📌 ' if is_selected else ''}{title}",
            key=f"conv_{conv_id}",
            use_container_width=True,
            help=f"Created: {conv_data['created_at']}\nUpdated: {conv_data['last_updated']}"
        ):
            switch_conversation(conv_id)
    
    st.divider()
    st.markdown(
        '<p style="font-size:0.75rem; color:#aaa;">Built for Kavak AI Engineer Case Study</p>',
        unsafe_allow_html=True,
    )

# Main chat interface
if current_conv:
    # Show suggestions only for empty conversations
    if not current_conv["messages"]:
        st.markdown("**Try asking:**")
        cols = st.columns(2)
        suggestions = [
            "Find me a round-trip to Tokyo in August with Star Alliance only, no overnight layovers.",
            "Can UAE passport holders visit Japan without a visa?",
            "What's the refund policy for non-refundable tickets?",
            "Show me the cheapest flights from Dubai to Tokyo.",
        ]
        for i, suggestion in enumerate(suggestions):
            col = cols[i % 2]
            if col.button(suggestion, key=f"sug_{i}", use_container_width=True):
                st.session_state.pending_input = suggestion
                st.rerun()

    # Display messages
    for msg in current_conv["messages"]:
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "✈️"):
            st.markdown(msg["content"])

    # Handle pending input
    if st.session_state.pending_input:
        user_input = st.session_state.pending_input
        st.session_state.pending_input = None
    else:
        user_input = st.chat_input("Ask about flights, visas, or travel policies…")

    if user_input:
        # Add user message
        current_conv["messages"].append({"role": "user", "content": user_input})
        current_conv["history"].append({"role": "user", "content": user_input})
        
        # Keep history manageable (last 20 messages)
        if len(current_conv["history"]) > 20:
            current_conv["history"] = current_conv["history"][-20:]
        
        # Update conversation title if it's the first message
        if len([m for m in current_conv["messages"] if m["role"] == "user"]) == 1:
            current_conv["title"] = generate_conversation_title(current_conv["messages"])
        
        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_input)

        # Get assistant response
        with st.chat_message("assistant", avatar="✈️"):
            with st.spinner("Aria is thinking…"):
                response = chat(user_input, current_conv["history"])
            st.markdown(response)

        # Add assistant message
        current_conv["messages"].append({"role": "assistant", "content": response})
        current_conv["history"].append({"role": "assistant", "content": response})
        
        # Save conversation
        save_current_conversation()
        
        st.rerun()