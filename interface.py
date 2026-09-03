"""
Streamlit front end for the News Article Summarizer.

Run with:
    streamlit run interface.py
"""

import os

import streamlit as st

from news_summarizer import NewsArticleSummarizer

st.set_page_config(page_title="News Article Summarizer", layout="centered")

st.title("News Article Summarizer")
st.write("Paste a news article URL, pick a model, and get an AI-generated summary.")

# --- Sidebar: model configuration ------------------------------------------
st.sidebar.header("Model")

model_type = st.sidebar.radio(
    "Backend",
    options=["ollama", "gemini"],
    format_func=lambda x: "Local (Ollama)" if x == "ollama" else "Gemini API",
)

if model_type == "ollama":
    model_name = st.sidebar.text_input("Ollama model name", value="llama3.2")
    api_key = None
else:
    model_name = st.sidebar.selectbox(
        "Gemini model",
        options=["gemini-3.6-flash", "gemini-3.8-flash"],
    )
    api_key = st.sidebar.text_input(
        "Gemini API key",
        value=os.getenv("GEMINI_API_KEY", ""),
        type="password",
        help="Loaded from the GEMINI_API_KEY environment variable if set.",
    )

chain_type = st.sidebar.selectbox(
    "Strategy",
    options=["stuff", "map_reduce"],
    format_func=lambda x: (
        "Single pass (fast, 1 request)"
        if x == "stuff"
        else "Map-reduce (more requests, better for long articles)"
    ),
)

summary_type = st.sidebar.radio("Summary style", options=["detailed", "concise"])

# --- Main area: input and results -------------------------------------------
url = st.text_input("Article URL", placeholder="https://example.com/some-article")
run = st.button("Generate summary", type="primary")

if run:
    if not url:
        st.warning("Please enter a URL first.")
        st.stop()
    if model_type == "gemini" and not api_key:
        st.warning("Please enter a Gemini API key, or switch to the local Ollama model.")
        st.stop()

    with st.spinner("Fetching article and generating summary..."):
        try:
            summarizer = NewsArticleSummarizer(
                model_type=model_type,
                model_name=model_name,
                api_key=api_key,
            )
            result = summarizer.summarize(
                url, summary_type=summary_type, chain_type=chain_type
            )
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

    if "error" in result:
        st.error(result["error"])
        st.stop()

    st.subheader(result["title"] or "Untitled")

    meta_parts = []
    if result["authors"]:
        meta_parts.append(", ".join(result["authors"]))
    if result["publish_date"]:
        meta_parts.append(str(result["publish_date"]))
    if meta_parts:
        st.caption(" · ".join(meta_parts))

    st.markdown(result["summary"])

    with st.expander(f"Show {len(result['chunks'])} source chunk(s)"):
        for i, chunk in enumerate(result["chunks"], start=1):
            st.markdown(f"**Chunk {i}**")
            st.write(chunk.page_content)