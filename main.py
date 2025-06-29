# myntra_fashion_assistant.py
"""
Streamlit Chatbot UI for Myntra Fashion Assistant.
• LLM interprets user intent, then embedding model retrieves top-K items.
• Chat history and query clarity preserved.
"""

import os, json, pathlib
from dotenv import load_dotenv

import faiss
from sentence_transformers import SentenceTransformer

try:
    import streamlit as st
except ModuleNotFoundError:
    st = None

INDEX_FILE = pathlib.Path("myntra_bge.index")
META_FILE  = pathlib.Path("myntra_meta.json")
MODEL_NAME = "BAAI/bge-large-en-v1.5"
TOP_K      = 5

valid_img = lambda u: u and u.strip() != "-" and u.strip().startswith("http") and u.strip().split(";")[0] or None

import streamlit as st
import openai

OPENAI_KEY = st.secrets["OPENAI_API_KEY"]

if st is not None and st.runtime.exists():

    @st.cache_resource(show_spinner=" Loading model…")
    def _model():
        return SentenceTransformer(MODEL_NAME)

    @st.cache_resource(show_spinner="Loading index…")
    def _index_meta():
        if not INDEX_FILE.exists() or not META_FILE.exists():
            st.error("Missing index or metadata file.")
            st.stop()
        return faiss.read_index(str(INDEX_FILE)), json.loads(META_FILE.read_text())

    model  = _model()
    index, meta = _index_meta()

    def search(query: str, k: int = TOP_K):
        q_vec = model.encode([query], normalize_embeddings=True).astype("float32")
        D, I  = index.search(q_vec, k)
        return [
            {
                "title": p.get("name"),
                "brand": p.get("brand"),
                "price": p.get("price"),
                "url":   p.get("purl"),
                "img":   valid_img(p.get("img"))
            }
            for rank, idx in enumerate(I[0])
            if (p := meta[idx])
        ]

    st.set_page_config("Fashion Assistant", layout="centered")
    st.title("Fashion Chat")
    st.caption("Ask your outfit question. I’ll understand and find what fits best.")

    if "history" not in st.session_state:
        st.session_state.history = []

    user_q = st.chat_input("What are you looking for?")
    if user_q:
        st.session_state.history.append({"role": "user", "content": user_q})

        with st.spinner("Thinking..."):
            search_query = user_q
            if OPENAI_KEY:
                try:
                    import openai
                    client = openai.OpenAI(api_key=OPENAI_KEY)
                    rewrite_prompt = (
                        "You are a shopping assistant that helps interpret user needs for product search.\n"
                        f"User asked: {user_q}\n"
                        "Rewrite this as a precise fashion product search query (e.g., 'blue cotton shirts for men', 'gold jewelry', etc)."
                    )
                    rewrite = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": rewrite_prompt}],
                        max_tokens=50,
                    )
                    search_query = rewrite.choices[0].message.content.strip()
                except Exception as e:
                    st.warning("Failed to refine query, using original.")

            hits = search(search_query)

        if not hits:
            st.chat_message("assistant").write("I couldn’t find anything for that.")
        else:
            # Short stylist response
            if OPENAI_KEY:
                try:
                    items = "\n".join(f"{i+1}. {h['title']} by {h['brand']} (₹{h['price']})" for i, h in enumerate(hits))
                    summary_prompt = (
                        f"User asked: {user_q}\n\n"
                        f"Matching products:\n{items}\n\n"
                        "Mention 2-3 great picks for the user and why. Be concise and helpful.")
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": summary_prompt}],
                        max_tokens=200,
                    )
                    st.session_state.history.append({"role": "assistant", "content": response.choices[0].message.content.strip()})
                except Exception:
                    st.session_state.history.append({"role": "assistant", "content": "Here are some suggestions you might like:"})
            else:
                st.session_state.history.append({"role": "assistant", "content": "Here are some suggestions you might like:"})

            for h in hits[:3]:
                st.session_state.history.append({"role": "assistant", "content": f"**{h['title']}**\n_{h['brand']}_ · ₹{h['price']}\n[View ▶]({h['url']})", "img": h["img"]})

    # Render chat history
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            if "img" in msg and msg["img"]:
                st.image(msg["img"], width=140)
            st.markdown(msg["content"])
