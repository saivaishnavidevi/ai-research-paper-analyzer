import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import re

# --- 1. PAGE CONFIGURATION & CUSTOM STYLING ---
st.set_page_config(
    page_title="AI Research Paper Analyzer & Knowledge Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject modern CSS styling for cards, headers, and spacing
st.markdown("""
    <style>
    .main-title { font-size: 28px; font-weight: 700; color: #1E3A8A; margin-bottom: 5px; }
    .sub-title { font-size: 15px; color: #4B5563; margin-bottom: 25px; }
    .section-card { padding: 20px; border-radius: 10px; background-color: #F8FAFC; border: 1px solid #E2E8F0; margin-bottom: 15px; }
    .metric-container { background-color: #EFF6FF; padding: 15px; border-radius: 8px; border-left: 5px solid #3B82F6; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

# Main App Header
st.markdown('<p class="main-title">📄 AI Research Paper Analyzer & Knowledge Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Advanced document intelligence platform for automated scholarly paper review and semantic vector search.</p>', unsafe_allow_html=True)

# Check if a file has already been uploaded in this session
if "uploaded_file_obj" not in st.session_state:
    st.session_state["uploaded_file_obj"] = None

# --- 2. DYNAMIC LAYOUT: CENTERED WELCOME WHEN EMPTY ---
if st.session_state["uploaded_file_obj"] is None:
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
            <div style="padding: 30px; border-radius: 12px; background-color: #F8FAFC; border: 2px dashed #CBD5E1; text-align: center;">
                <h3>📤 Upload Your Research Paper</h3>
                <p style="color: #64748B; font-size: 14px;">Select an academic PDF to generate an automated structured summary.</p>
            </div>
        """, unsafe_allow_html=True)
        st.write("")
        uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"], label_visibility="collapsed")
        if uploaded_file is not None:
            st.session_state["uploaded_file_obj"] = uploaded_file
            st.rerun()
    st.markdown("<br>", unsafe_allow_html=True)
else:
    # Sidebar control center once a file is active
    with st.sidebar:
        st.markdown("### 🎛️ Control Center")
        st.markdown("---")
        st.markdown("#### 📁 Document Management")
        new_upload = st.file_uploader("Upload a different PDF", type=["pdf"])
        if new_upload is not None:
            st.session_state["uploaded_file_obj"] = new_upload
            st.rerun()
        st.success("Active PDF Loaded Successfully!")
        st.markdown("---")
        st.caption("System v2.1 | AI & Data Science")
        
    uploaded_file = st.session_state["uploaded_file_obj"]

# --- 3. PROCESS AND DISPLAY PDF ANALYSIS IF UPLOADED ---
if st.session_state["uploaded_file_obj"] is not None:
    uploaded_file = st.session_state["uploaded_file_obj"]
    
    # Extract text from PDF
    reader = PdfReader(uploaded_file)
    extracted_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            extracted_text += text + "\n"
            
    # Normalize text (fix hyphenated line breaks)
    extracted_text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', extracted_text)
    extracted_text = re.sub(r'\s+', ' ', extracted_text)

    st.markdown("---")
    st.markdown("### 📑 Uploaded Paper Analysis Dashboard")
    
    # Metric Overview Row
    m1, m2 = st.columns(2)
    with m1:
        st.metric(label="Total Pages Extracted", value=len(reader.pages))
    with m2:
        st.metric(label="Processing Status", value="Parsed & Normalized", delta="Success")
    
    with st.expander("🔍 View Full Extracted Text Document"):
        st.text_area("Full Text Content", extracted_text, height=200, label_visibility="collapsed")
        
    # --- SMART SECTION EXTRACTOR ---
    def extract_section(text, keywords, fallback_sentences):
        for kw in keywords:
            pattern = rf'(?:{kw})\s*[:\-\u2013]?\s*(.*?)(?=(?:Abstract|Introduction|Methodology|Results|Conclusion|References|$))'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                res = match.group(1).strip()
                if len(res) > 80:
                    sentences = [s.strip() for s in re.split(r'(?<=[.?!])\s+', res) if len(s.strip()) > 20]
                    return " ".join(sentences[:3]) if sentences else res[:400]
        return fallback_sentences

    intro_summary = extract_section(extracted_text, ["Abstract", "Introduction", "Context"], "Introduction details could not be automatically isolated.")
    method_summary = extract_section(extracted_text, ["Methodology", "Approach", "Materials and Methods", "Model"], "Methodology details could not be automatically isolated.")
    finding_summary = extract_section(extracted_text, ["Results", "Findings", "Analysis", "Evaluation"], "Findings and results could not be automatically isolated.")
    conclusion_summary = extract_section(extracted_text, ["Conclusion", "Discussion", "Summary"], "Conclusion notes could not be automatically isolated.")

    st.markdown("### 🔍 Automated Paper Summary")
    st.markdown("Structured insights extracted automatically from document sections:")
    
    # Structured Layout Cards for Summaries
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown(f"""
            <div class="section-card">
                <h4 style="margin-top:0; color:#1E3A8A;">📌 1. Context / Introduction</h4>
                <p style="color:#334155; font-size:14px; line-height:1.5;">{intro_summary}</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class="section-card">
                <h4 style="margin-top:0; color:#1E3A8A;">📊 2. Key Insights / Findings</h4>
                <p style="color:#334155; font-size:14px; line-height:1.5;">{finding_summary}</p>
            </div>
        """, unsafe_allow_html=True)
        
    with col_b:
        st.markdown(f"""
            <div class="section-card">
                <h4 style="margin-top:0; color:#1E3A8A;">⚙️ 3. Core Approach / Methodology</h4>
                <p style="color:#334155; font-size:14px; line-height:1.5;">{method_summary}</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class="section-card">
                <h4 style="margin-top:0; color:#1E3A8A;">⚠️ 4. Conclusion / Summary Note</h4>
                <p style="color:#334155; font-size:14px; line-height:1.5;">{conclusion_summary}</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

# --- 4. MAIN PAGE: PRE-INDEXED SEARCH WORKFLOW ---
st.markdown("---")
st.markdown("### 🔍 Search Pre-Indexed Research Database")
st.markdown("Search through curated academic papers instantly using semantic vector similarity.")

@st.cache_resource
def load_resources():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    df = pd.read_csv("processed_papers.csv")
    embeddings = model.encode(df['combined_text'].fillna('').tolist(), show_progress_bar=False)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return model, df, embeddings

with st.spinner("Loading search vector index..."):
    model, df, document_embeddings = load_resources()

# Search Input Box with clean layout
query = st.text_input("Enter research topic, keyword, or methodology query:", placeholder="e.g., Transformer architecture, RAG optimization...")

if query:
    query_vector = model.encode([query])
    query_vector = query_vector / np.linalg.norm(query_vector, axis=1, keepdims=True)
    
    similarities = np.dot(document_embeddings, query_vector.T).flatten()
    top_indices = np.argsort(similarities)[::-1][:5]
    
    st.markdown(f"#### Top 5 Matching Papers for: *'{query}'*")
    st.write("")
    
    for i, idx in enumerate(top_indices):
        row = df.iloc[idx]
        score = similarities[idx]
        
        # Styled Expanders for Search Results
        with st.expander(f"📌 [{i+1}] {row.get('title', 'Untitled Paper')}  —  (Similarity Score: {score:.2f})"):
            st.markdown(f"**Description / Abstract:**")
            st.write(row.get('description', 'No description available.'))
            if 'tag' in df.columns:
                st.markdown(f"<span style='background-color: #EFF6FF; color: #1D4ED8; padding: 3px 8px; border-radius: 4px; font-size: 12px; font-weight: 600;'>Category: {row.get('tag', 'General')}</span>", unsafe_allow_html=True)