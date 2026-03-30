import streamlit as st
import os
from agent import run_agent
from utils import extract_text_from_file

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedScan AI · Report Analyser",
    page_icon="🩺",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0f1117;
    color: #e8eaf0;
}

.block-container { max-width: 780px; padding-top: 2.5rem; }

h1, h2, h3 { font-family: 'DM Serif Display', serif; }

.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    line-height: 1.2;
    background: linear-gradient(135deg, #a78bfa, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.hero-sub {
    color: #7c8399;
    font-size: 1rem;
    font-weight: 300;
    margin-bottom: 2rem;
}

.upload-box {
    border: 2px dashed #2e3248;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    background: #161926;
    margin-bottom: 1.5rem;
}

.result-card {
    background: linear-gradient(145deg, #161926, #1a1f2e);
    border: 1px solid #2a2f45;
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin-top: 1.5rem;
    line-height: 1.8;
    font-size: 0.97rem;
    color: #d1d5e8;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}

.result-card h4 {
    font-family: 'DM Serif Display', serif;
    color: #a78bfa;
    font-size: 1.2rem;
    margin-bottom: 0.8rem;
}

.step-pill {
    display: inline-block;
    background: #1e2235;
    border: 1px solid #2e3248;
    border-radius: 999px;
    padding: 0.25rem 0.85rem;
    font-size: 0.78rem;
    color: #7c8399;
    margin-bottom: 1rem;
}

.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #0ea5e9);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 2rem;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.95rem;
    font-weight: 500;
    cursor: pointer;
    width: 100%;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

div[data-testid="stFileUploader"] {
    background: #161926;
    border: 2px dashed #2e3248;
    border-radius: 14px;
    padding: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">MedScan AI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Upload a USG report or prescription — the agent will analyse and summarise patient details.</div>', unsafe_allow_html=True)

# ── API key ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # st.markdown("### ⚙️ Configuration")
    # api_key = st.text_input("Tavily API Key", type="password", placeholder="tvly-...")
    # st.markdown("---")
    st.markdown("**How it works**")
    st.markdown("""
1. Upload a USG report or prescription (PDF / image / text)  
2. The agent extracts patient details  
3. It searches the web for context  
4. Returns a concise patient summary
    """)
    st.markdown("---")
    st.caption("Powered by LangGraph + Tavily + GPT-4o")

# ── File upload ───────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Drop your report here",
    type=["pdf", "png", "jpg", "jpeg", "txt"],
    label_visibility="collapsed",
)

if uploaded_file:
    st.success(f"✅ **{uploaded_file.name}** uploaded ({uploaded_file.size // 1024} KB)")

    if st.button("🔍 Analyse Report"):
        if not uploaded_file:
            st.error("Please upload a file first.")
        else:
            

            with st.spinner("Extracting text from document…"):
                raw_text = extract_text_from_file(uploaded_file)

            if not raw_text.strip():
                st.error("Could not extract text from the file. Try a clearer scan or a text-based PDF.")
            else:
                with st.expander("📄 Extracted raw text (preview)", expanded=False):
                    st.text(raw_text[:1500] + ("…" if len(raw_text) > 1500 else ""))

                with st.spinner("Agent is thinking…"):
                    summary, warnings = run_agent(raw_text)
                    summary.replace("\n", "<br>")

                st.markdown('<div class="result-card"><h4>🩺 Patient Summary</h4>' +
                            summary + "</div>",
                            unsafe_allow_html=True)

                st.download_button(
                    "⬇️ Download Summary",
                    data=summary,
                    file_name="patient_summary.txt",
                    mime="text/plain",
                )