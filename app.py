import os
import io
import time
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain

# ============== Setup ==============
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

st.set_page_config(page_title="PDF Summarizer + Q&A", page_icon="📚", layout="wide")

# Global CSS (colors, animations, layout polish)
st.markdown("""
<style>
/* Page background & base typography */
html, body, .stApp { background: linear-gradient(135deg, #eef6ff 0%, #f8fbff 60%, #ffffff 100%) !important; }
* { font-family: 'Segoe UI', system-ui, -apple-system, Roboto, 'Helvetica Neue', Arial, 'Noto Sans', 'Liberation Sans', sans-serif; }

/* Header with animated gradient text */
.hero {
  text-align: center;
  margin: 8px 0 18px;
}
.hero h1 {
  font-size: 2.1rem;
  line-height: 1.2;
  font-weight: 800;
  background: linear-gradient(90deg, #0ea5e9, #6366f1, #22c55e);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  animation: hueShift 6s linear infinite;
}
@keyframes hueShift { 0% { filter: hue-rotate(0deg);} 100% { filter: hue-rotate(360deg);} }

/* Cards */
.card {
  background: #ffffffcc;
  backdrop-filter: blur(6px);
  border-radius: 16px;
  padding: 18px 18px;
  border: 1px solid rgba(0,0,0,0.06);
  box-shadow: 0 8px 24px rgba(0,0,0,0.06);
  transition: transform .2s ease, box-shadow .2s ease;
}
.card:hover { transform: translateY(-2px); box-shadow: 0 12px 32px rgba(0,0,0,0.09); }

/* Buttons */
.stButton>button {
  background: linear-gradient(135deg, #2563eb, #0ea5e9);
  color: #fff;
  border: 0;
  padding: 0.65rem 1.1rem;
  border-radius: 12px;
  font-weight: 700;
  letter-spacing: .2px;
  transition: transform .12s ease, box-shadow .2s ease, opacity .2s ease;
  box-shadow: 0 6px 14px rgba(37,99,235,.25);
}
.stButton>button:hover { transform: translateY(-1px); box-shadow: 0 10px 20px rgba(14,165,233,.25); }
.stButton>button:active { transform: translateY(0); opacity: .9; }

/* Inputs */
input[type="text"] {
  border-radius: 12px !important;
  border: 1px solid #e6eef8 !important;
  box-shadow: 0 4px 10px rgba(14,165,233,.08) inset !important;
}

/* Summary & Answer highlights */
.summary {
  background: #fff7e6;
  border-left: 6px solid #ffb020;
  padding: 12px 14px;
  border-radius: 12px;
  animation: fadeIn .4s ease;
}
.answer {
  background: #eaf6ff;
  border-left: 6px solid #2ea6ff;
  padding: 12px 14px;
  border-radius: 12px;
  animation: fadeIn .4s ease;
}

/* Shimmer skeleton for loading blocks */
.skeleton {
  border-radius: 12px;
  height: 64px;
  background: linear-gradient(90deg, #eef4ff 25%, #f6f9ff 37%, #eef4ff 63%);
  background-size: 400% 100%;
  animation: shimmer 1.4s ease infinite;
}
@keyframes shimmer { 0% { background-position: 200% 0; } 100% { background-position: -200% 0; } }

/* Fade-in */
@keyframes fadeIn { from {opacity: 0; transform: translateY(2px);} to {opacity: 1; transform: translateY(0);} }
.small { font-size: .92rem; color: #566; }
.badge {
  display: inline-block;
  font-size: .72rem;
  background: #eef6ff;
  color: #2563eb;
  padding: 4px 8px;
  border-radius: 999px;
  border: 1px solid #d9e9ff;
  margin-left: 6px;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<div class='hero'><h1>📚 PRAVACHAK- PDF Summarizer + Q&A </h1><div class='small'>Upload PDFs• Ask questions about it<span class='badge'>pravachak bot</span></div></div>", unsafe_allow_html=True)

# Guard
if not OPENROUTER_API_KEY:
    st.error("Add your OpenRouter API key to `.env` as `OPENROUTER_API_KEY` and restart the app.")
    st.stop()

# Sidebar (options)
with st.sidebar:
    st.markdown("### ⚙️ Options")
    chunk_size = st.slider("Chunk size", 500, 2000, 1000, 100)
    overlap = st.slider("Chunk overlap", 50, 400, 150, 50)
    k = st.slider("Retriever: top-k passages", 2, 10, 4, 1)
    temp = st.slider("LLM temperature", 0.0, 1.0, 0.0, 0.1)
    st.markdown("---")
    auto_summary = st.toggle("Auto-generate summary after upload", value=True)
    st.caption("Tip: lower chunk size for tighter retrieval; increase k for broader answers.")

# File upload
uploaded = st.file_uploader("📂 Upload a PDF", type=["pdf"])

# Keep state across interactions
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "docs" not in st.session_state:
    st.session_state.docs = None
if "summary" not in st.session_state:
    st.session_state.summary = None

def build_pipeline(temp_path: str):
    # Load PDF
    loader = PyPDFLoader(temp_path)
    documents = loader.load()

    # Split text
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    split_docs = splitter.split_documents(documents)

    # Embeddings (HuggingFace – local)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Vector store
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return split_docs, vectorstore

def get_llm():
    # DeepSeek via OpenRouter (OpenAI-compatible client)
    return ChatOpenAI(
        model="deepseek/deepseek-chat",
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        temperature=temp,
    )

def animated_placeholder(duration=1.6):
    ph = st.empty()
    for _ in range(int(duration*4)):
        ph.markdown("<div class='skeleton'></div>", unsafe_allow_html=True)
        time.sleep(0.25)
    ph.empty()

# Main layout: two columns
col_left, col_right = st.columns([1.1, 1])

if uploaded:
    # Save temp file
    temp_path = f"temp_{uploaded.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded.getbuffer())

    with st.spinner("🔧 Processing PDF..."):
        # Animated shimmer
        animated_placeholder(1.2)
        split_docs, vectorstore = build_pipeline(temp_path)
        st.session_state.vectorstore = vectorstore
        st.session_state.docs = split_docs

    llm = get_llm()

    with col_left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### 📝 Summary")
        if auto_summary and st.session_state.summary is None:
            with st.spinner("✨ Summarizing..."):
                chain = load_summarize_chain(llm, chain_type="map_reduce")
                # new API: invoke
                out = chain.invoke({"input_documents": st.session_state.docs})
                st.session_state.summary = out["output_text"]

        if st.button("Regenerate Summary"):
            with st.spinner("✨ Summarizing..."):
                chain = load_summarize_chain(llm, chain_type="map_reduce")
                out = chain.invoke({"input_documents": st.session_state.docs})
                st.session_state.summary = out["output_text"]

        if st.session_state.summary:
            st.markdown(f"<div class='summary'>{st.session_state.summary}</div>", unsafe_allow_html=True)
            # Download
            content = st.session_state.summary
            buf = io.BytesIO(content.encode("utf-8"))
            st.download_button("⬇️ Download Summary (TXT)", data=buf, file_name="summary.txt", mime="text/plain")
        else:
            st.info("Click **Regenerate Summary** to create a summary, or enable **Auto-generate** in the sidebar.")

        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### 💬 Ask about the PDF")

        if st.session_state.vectorstore is None:
            st.warning("Upload a PDF first to enable Q&A.")
        else:
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": k})
            qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

            q = st.text_input("Your question")
            go = st.button("🔎 Get Answer")

            if go and q.strip():
                with st.spinner("🤔 Thinking..."):
                    animated_placeholder(0.8)
                    result = qa_chain.invoke({"query": q})
                st.markdown(f"<div class='answer'>{result['result']}</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # Fun animation
    st.balloons()

else:
    with col_left:
        st.markdown("<div class='card'><p class='small'>Upload a PDF to begin. You’ll get a concise summary and can ask focused questions with retrieval-augmented answers.</p></div>", unsafe_allow_html=True)
    with col_right:
        st.markdown("<div class='card'><p class='small'>No file uploaded yet. The Q&A panel will activate after we build the vector index using local HuggingFace embeddings.</p></div>", unsafe_allow_html=True)
