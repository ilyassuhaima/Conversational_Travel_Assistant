"""
Kavak Travel Assistant — Streamlit Frontend
============================================
Run with: streamlit run streamlit_app.py
"""

import streamlit as st

from main import chat

# ── Page config ──────────────────────────────
st.set_page_config(
    page_title="Kavak Travel Assistant",
    page_icon="✈️",
    layout="centered",
)

# ── Custom CSS ───────────────────────────────
st.markdown(
    """
    <style>
    .stChatMessage { border-radius: 12px; }
    .main { background-color: #f8f9fa; }
    h1 { color: #1a1a2e; }
    .subtitle { color: #6c757d; font-size: 0.9rem; margin-top: -10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ───────────────────────────────────
st.title("✈️ Kavak Travel Assistant")
st.markdown(
    '<p class="subtitle">Powered by Claude + LangGraph · Ask about flights, visas, and travel policies</p>',
    unsafe_allow_html=True,
)
st.divider()

# ── Session state ────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.history = []

# ── Suggested prompts (shown only at start) ──
if not st.session_state.messages:
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

# ── Render chat history ───────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "✈️"):
        st.markdown(msg["content"])

# ── Handle pending suggestion click ──────────
if "pending_input" in st.session_state:
    user_input = st.session_state.pop("pending_input")
else:
    user_input = st.chat_input("Ask about flights, visas, or travel policies…")

# ── Process input ────────────────────────────
if user_input:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    # Get assistant response
    with st.chat_message("assistant", avatar="✈️"):
        with st.spinner("Aria is thinking…"):
            response = chat(user_input, st.session_state.history)
        st.markdown(response)

    # Update history
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.history.append({"role": "user", "content": user_input})
    st.session_state.history.append({"role": "assistant", "content": response})

# ── Sidebar info ─────────────────────────────
with st.sidebar:
    st.markdown("## 🛠️ About Aria")
    st.markdown(
        """
        **Aria** is a conversational travel assistant built with:
        - 🤖 **Groq** (llama-3.3-70b) as the reasoning LLM
        - 🔗 **LangGraph** for stateful agent orchestration
        - 🔍 **FAISS** vector store for RAG retrieval
        - 📐 **HuggingFace Embeddings** (`all-MiniLM-L6-v2`) — fully local

        ---
        **Aria can help you:**
        - 🔎 Search flights by route, alliance, price, cabin
        - 🛂 Check visa requirements by passport
        - 💳 Understand refund & cancellation policies
        - 🧳 Learn about baggage allowances
        - ⭐ Compare airline alliances

        ---
        """
    )
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history = []
        st.rerun()

    st.markdown(
        '<p style="font-size:0.75rem; color:#aaa;">Built for Kavak AI Engineer Case Study</p>',
        unsafe_allow_html=True,
    )