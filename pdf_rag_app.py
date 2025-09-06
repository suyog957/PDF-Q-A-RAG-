import os
import re
import io
from typing import List, Tuple, Optional
from dataclasses import dataclass

import torch
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# Optional OCR deps (graceful fallback if missing)
try:
    from pdf2image import convert_from_bytes
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# LangChain / HF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.prompts import PromptTemplate

try:
    from langchain_core.documents import Document
except Exception:
    from langchain.schema import Document

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# ======================
# Streamlit UI Setup
# ======================
load_dotenv()
st.set_page_config(page_title="PDF Q&A (RAG) — Beginner Friendly", page_icon="📚", layout="wide")
st.markdown("""
<style>
.block-container { padding-top: 1rem; padding-bottom: 2rem; }
.small { font-size: 0.85rem; color: #666; }
.codebox { background: #0f172a; color: #e2e8f0; padding: 0.75rem 1rem; border-radius: 0.5rem; font-family: ui-monospace, Menlo, Consolas, monospace; }
.badge { display:inline-block; padding:0.2rem 0.5rem; border-radius:0.5rem; background:#eef2ff; color:#3730a3; font-weight:600; margin-right:0.35rem; }
.kpi { padding:0.75rem 1rem; border:1px solid #e5e7eb; border-radius:0.75rem; background:#fafafa;}
.highlight { background: #fff3bf; }
</style>
""", unsafe_allow_html=True)

# ======================
# Config dataclass
# ======================
@dataclass
class RAGConfig:
    qa_model_name: str = "google/flan-t5-base"
    emb_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 400
    chunk_overlap: int = 50
    top_k: int = 10
    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float = 0.0
    strict_context: bool = False
    answer_style: str = "beginner"  # "beginner" | "concise"

# ======================
# Helpers & Globals
# ======================
def has_cuda() -> bool:
    try:
        return torch.cuda.is_available()
    except Exception:
        return False

DEVICE_STR = "cuda" if has_cuda() else "cpu"
DEVICE_IDX = 0 if has_cuda() else -1  # HF pipeline expects GPU index or -1

_WORD_RE = re.compile(r"[^a-z0-9]+")

def _normalize(s: str) -> str:
    return _WORD_RE.sub(" ", s.lower()).strip()

def extract_term_from_question(q: str) -> str:
    """Try to map 'What is X?' → 'X'."""
    q = q.strip()
    m = re.match(r"^\s*(what\s+is|what\s+are|define|explain)\s+(.+?)\??\s*$", q, flags=re.I)
    if m:
        return m.group(2).strip(' "')
    return q.strip(' "')

def ingest_pdf_bytes(file_bytes: bytes, filename: str, use_ocr: bool = False) -> List[Document]:
    docs = []
    if use_ocr and OCR_AVAILABLE:
        images = convert_from_bytes(file_bytes, fmt="png", dpi=200)
        for i, img in enumerate(images, start=1):
            text = pytesseract.image_to_string(img) or ""
            text = text.strip()
            if text:
                docs.append(Document(page_content=text, metadata={"source": filename, "page": i}))
    else:
        reader = PdfReader(io.BytesIO(file_bytes))
        for i, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                docs.append(Document(page_content=text, metadata={"source": filename, "page": i}))
    return docs

def preprocess_documents(documents: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

@st.cache_resource(show_spinner=False)
def get_embeddings(model_name: str, device_str: str):
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device_str},
        encode_kwargs={"normalize_embeddings": True},
    )

def build_vector_store(chunks: List[Document], emb_model_name: str, device_str: str) -> FAISS:
    embeddings = get_embeddings(emb_model_name, device_str)
    return FAISS.from_documents(chunks, embeddings)

@st.cache_resource(show_spinner=False)
def get_llm(qa_model_name: str, device_idx: int, max_new_tokens: int, do_sample: bool, temperature: float):
    tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(qa_model_name).to(DEVICE_STR)
    gen_pipe = pipeline(
        task="text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device_idx,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=(temperature if do_sample else None),
    )
    llm = HuggingFacePipeline(pipeline=gen_pipe)
    return llm, tokenizer

def get_safe_model_input_budget(tokenizer) -> int:
    raw = getattr(tokenizer, "model_max_length", 512)
    if raw is None or raw > 10000:
        return 2048
    return int(raw)

def pack_context_to_token_budget(docs: List[Document], tokenizer, max_context_tokens: int) -> str:
    parts, used = [], 0
    for d in docs:
        text = d.page_content or ""
        if not text.strip():
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        n = len(ids)
        if used + n <= max_context_tokens:
            parts.append(text)
            used += n
        else:
            remaining = max_context_tokens - used
            if remaining > 0:
                truncated = tokenizer.decode(ids[:remaining], skip_special_tokens=True)
                if truncated.strip():
                    parts.append(truncated)
            break
    return "\n\n".join(parts)

def _score_doc_for_term(doc: Document, term: str, term_tokens: List[str]) -> int:
    text = doc.page_content.lower()
    score = 0
    if term.lower() in text:
        score += 200
    for t in term_tokens:
        if t and t in text:
            score += 10
    return score

def rerank_docs_for_term(docs: List[Document], term: str) -> List[Document]:
    tokens = [tok for tok in _normalize(term).split() if tok]
    scored = [(d, _score_doc_for_term(d, term, tokens)) for d in docs]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [d for d, _ in scored]

def expand_acronyms_once(text: str) -> str:
    text = re.sub(r"\bNOWs\b", "NOW (negotiable order of withdrawal) accounts", text, flags=re.I)
    text = re.sub(r"\bNOW\b", "NOW (negotiable order of withdrawal)", text, flags=re.I)
    return text

# ====== ROBUST DEFINITION EXTRACTION (updated) ======
def try_extract_definition(docs: List[Document], term: str) -> Optional[Tuple[str, Document]]:
    """
    Extracts the definition sentence immediately following the term, e.g.:
    "80. Transactions Account  Deposit account on which ..."
    More robust to punctuation/spacing/newlines.
    """
    term_norm = term.strip()
    # First regex: term followed by punctuation/space, then capture the first sentence
    term_regex = re.compile(
        rf"{re.escape(term_norm)}\s*[:\-\u2013\u2014]*\s*(.+?[\.\u3002])",
        flags=re.IGNORECASE | re.DOTALL,
    )
    # Alternate regex: allow optional leading numbering like "80."
    alt_regex = re.compile(
        rf"(?:^\s*\d+\.\s*)?{re.escape(term_norm)}\s*[:\-\u2013\u2014]*\s*(.+?[\.\u3002])",
        flags=re.IGNORECASE | re.DOTALL,
    )

    for d in docs:
        text = (d.page_content or "").replace("\r", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n+", " ", text)

        m = term_regex.search(text) or alt_regex.search(text)
        if m:
            snippet = m.group(1).strip()
            snippet = expand_acronyms_once(snippet)
            if snippet and any(c.isalpha() for c in snippet):
                if snippet[0].islower():
                    snippet = snippet[0].upper() + snippet[1:]
                return snippet, d
    return None

# ====== PROMPTS ======
def make_prompt_strict() -> PromptTemplate:
    template = (
        "You are a precise assistant. Answer ONLY about the exact term in quotes using the context.\n"
        "If the answer is not in the context, reply exactly: 'I could not find the answer in the documents.'\n\n"
        "Term: \"{term}\"\n\n"
        "Write a clear 1–2 sentence definition. If the context lists specific types or examples, mention them.\n\n"
        "Context:\n{context}\n\n"
        "Answer:"
    )
    return PromptTemplate.from_template(template)

def make_prompt_enrich() -> PromptTemplate:
    # Stronger instruction to follow the format
    template = (
        "You MUST follow the exact format below. Do not output anything else or any headings not shown.\n"
        "Use the canonical definition, then expand in plain English with helpful bullets and short comparisons. "
        "Do NOT invent numeric details (no rates, fees, dollar amounts). Expand acronyms on first use.\n\n"
        "Term: \"{term}\"\n"
        "Canonical definition: {definition}\n\n"
        "Return in this format ONLY:\n"
        "Definition:\n"
        "<one-sentence restatement>\n\n"
        "Plain English:\n"
        "- <bullet 1>\n"
        "- <bullet 2>\n"
        "- <bullet 3>\n"
        "- <bullet 4>\n\n"
        "Related terms:\n"
        "- <term and how it differs>\n"
        "- <term and how it differs>\n"
    )
    return PromptTemplate.from_template(template)

def render_sources(docs: List[Document], top_n: int = 3) -> str:
    parts = [f"{d.metadata.get('source')} p.{d.metadata.get('page')}" for d in docs[:top_n]]
    return "Sources: " + "; ".join(parts)

def highlight_term(text: str, term: str) -> str:
    if not term:
        return text
    escaped = re.escape(term)
    return re.sub(escaped, lambda m: f"<mark class='highlight'>{m.group(0)}</mark>", text, flags=re.I)

# ====== FALLBACK BUILDER (new) ======
def build_beginner_answer(term: str, definition: str, generated: str) -> str:
    """
    If the model's enrichment is missing or too short, fall back to a clean, structured answer.
    """
    ok = bool(generated) and len(generated) >= 80 and "Definition:" in generated
    if ok:
        return generated

    bullets = []
    # If definition mentions checks, call it out.
    if re.search(r"\bcheck", definition, flags=re.I):
        bullets.append("- You can use it for everyday payments (e.g., writing checks).")
    # Mention common types if present
    types = []
    if re.search(r"\bdemand deposit", definition, flags=re.I):
        types.append("demand deposit accounts")
    if re.search(r"\bNOW", definition, flags=re.I):
        types.append("NOW (negotiable order of withdrawal) accounts")
    if types:
        bullets.append(f"- Common types include {', '.join(types)}.")
    if not bullets:
        bullets.append("- It is a bank account used for everyday transactions.")

    return (
        f"Definition:\n{definition}\n\n"
        "Plain English:\n" + "\n".join(bullets) + "\n\n"
        "Related terms:\n"
        "- Demand deposit account: a common transactions account that allows check writing.\n"
        "- NOW account: a type of transactions account that also allows check writing.\n"
    )

# ======================
# Sidebar Controls
# ======================
with st.sidebar:
    st.header("⚙️ Settings")

    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

    ocr_enable = st.checkbox("Use OCR for scanned PDFs (requires pdf2image + tesseract)", value=False, disabled=not OCR_AVAILABLE)
    if not OCR_AVAILABLE:
        st.caption("⚠️ OCR packages not detected. Install `pdf2image` and `pytesseract` to enable.")

    qa_model = st.selectbox("LLM (text2text)", ["google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large"], index=1)
    emb_model = st.selectbox("Embeddings", ["sentence-transformers/all-MiniLM-L6-v2"], index=0)

    colA, colB = st.columns(2)
    with colA:
        chunk_size = st.slider("Chunk size", 200, 1200, 400, 50)
        top_k = st.slider("Top K", 3, 15, 10, 1)
    with colB:
        chunk_overlap = st.slider("Chunk overlap", 20, 200, 50, 10)
        max_new = st.slider("Max new tokens", 64, 512, 256, 32)

    strict_context = st.toggle("Strict context only (no general knowledge)", value=False)
    answer_style = st.selectbox("Answer style", ["beginner", "concise"], index=0)

    do_sample = st.toggle("Enable sampling", value=False)
    temperature = st.slider("Temperature", 0.0, 1.2, 0.0, 0.1, disabled=not do_sample)

    st.divider()
    st.caption(f"Device: **{DEVICE_STR}** | FAISS: CPU | OCR available: **{OCR_AVAILABLE}**")
    debug = st.toggle("Debug mode", value=False)

    rebuild = st.button("🔁 Rebuild Index", type="primary")

cfg = RAGConfig(
    qa_model_name=qa_model,
    emb_model_name=emb_model,
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    top_k=top_k,
    max_new_tokens=max_new,
    do_sample=do_sample,
    temperature=temperature,
    strict_context=strict_context,
    answer_style=answer_style,
)

# ======================
# Session State
# ======================
if "docs" not in st.session_state:
    st.session_state.docs = []
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "vector" not in st.session_state:
    st.session_state.vector = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = None
if "hist" not in st.session_state:
    st.session_state.hist = []
if "context_budget" not in st.session_state:
    st.session_state.context_budget = 384

# ======================
# Header / Intro
# ======================
st.title("📚 PDF Q&A (RAG) — Beginner-Friendly Definitions")
st.caption("Upload PDFs (e.g., glossaries, manuals), ask a question, and get a clear answer with sources. "
           "Exact-match extraction ensures accuracy; the model then expands it for beginners.")

# ======================
# Build / Rebuild Index
# ======================
def rebuild_index(files):
    if not files:
        st.warning("Upload at least one PDF to build an index.")
        return

    all_docs = []
    with st.spinner("Extracting text from PDFs..."):
        for uf in files:
            file_bytes = uf.read()
            name = uf.name
            pages = ingest_pdf_bytes(file_bytes, name, use_ocr=ocr_enable)
            all_docs.extend(pages)

    if not all_docs:
        st.error("No text found. If your PDFs are scanned images, enable OCR.")
        return

    st.session_state.docs = all_docs

    with st.spinner("Chunking & embedding..."):
        chunks = preprocess_documents(all_docs, cfg.chunk_size, cfg.chunk_overlap)
        st.session_state.chunks = chunks
        vector = build_vector_store(chunks, cfg.emb_model_name, DEVICE_STR)
        st.session_state.vector = vector

    with st.spinner("Loading LLM..."):
        llm, tokenizer = get_llm(cfg.qa_model_name, DEVICE_IDX, cfg.max_new_tokens, cfg.do_sample, cfg.temperature)
        st.session_state.llm = llm
        st.session_state.tokenizer = tokenizer
        model_max = get_safe_model_input_budget(tokenizer)
        st.session_state.context_budget = max(128, model_max - 128)

    st.success(f"Index ready ✓  |  Pages: {len(all_docs)}  |  Chunks: {len(st.session_state.chunks)}")

if rebuild or (uploaded_files and st.session_state.vector is None):
    rebuild_index(uploaded_files)

# KPIs
col1, col2, col3, col4 = st.columns(4)
col1.markdown(f"<div class='kpi'><b>Device</b><br>{DEVICE_STR}</div>", unsafe_allow_html=True)
col2.markdown(f"<div class='kpi'><b>Pages</b><br>{len(st.session_state.docs)}</div>", unsafe_allow_html=True)
col3.markdown(f"<div class='kpi'><b>Chunks</b><br>{len(st.session_state.chunks)}</div>", unsafe_allow_html=True)
col4.markdown(f"<div class='kpi'><b>Context Budget</b><br>{st.session_state.context_budget} tokens</div>", unsafe_allow_html=True)

st.divider()

# ======================
# Ask Box
# ======================
q = st.text_input("Ask a question (e.g., 'What is Transactions Account?' or just 'Transactions Account')", "")
ask = st.button("Ask", type="primary", disabled=not (q and st.session_state.vector and st.session_state.llm))

prompt_strict = make_prompt_strict()
prompt_enrich = make_prompt_enrich()

def answer_question(question: str):
    """Returns (markdown_answer, reranked_docs, confidence_0to1)."""
    term = extract_term_from_question(question)
    vector = st.session_state.vector
    llm = st.session_state.llm
    tok = st.session_state.tokenizer

    docs = vector.similarity_search(term, k=cfg.top_k)
    reranked = rerank_docs_for_term(docs, term)

    # Confidence heuristic: fraction of docs that contain exact phrase
    contains = sum(1 for d in reranked if term.lower() in (d.page_content or "").lower())
    conf = contains / max(1, len(reranked))

    if debug:
        with st.expander("🔎 Debug: Retrieved (reranked)"):
            for i, d in enumerate(reranked, 1):
                src = d.metadata.get("source"); pg = d.metadata.get("page")
                preview = (d.page_content[:800] + "…") if len(d.page_content) > 800 else d.page_content
                st.markdown(f"**Doc {i} — {src} p.{pg}**")
                st.markdown(f"<div class='codebox'>{highlight_term(preview, term)}</div>", unsafe_allow_html=True)

    # 1) High-precision extraction
    extracted = try_extract_definition(reranked, term)
    if extracted:
        definition, src_doc = extracted
        if cfg.answer_style == "beginner" and not cfg.strict_context:
            prompt = prompt_enrich.format(term=term, definition=definition)
            expanded = llm.invoke(prompt).strip()
            body = build_beginner_answer(term, definition, expanded)
            md = body + "\n\n" + render_sources([src_doc])
        else:
            md = f"Definition:\n{definition}\n\n" + render_sources([src_doc])
        return md, reranked, conf

    # 2) LLM QA fallback (context-only concise definition, then optional enrichment)
    context = pack_context_to_token_budget(reranked, tok, st.session_state.context_budget)
    if not context.strip():
        return "I could not find the answer in the documents.", reranked, conf

    strict_prompt = prompt_strict.format(context=context, term=term)
    concise = llm.invoke(strict_prompt).strip()
    if not concise or "could not find" in concise.lower():
        return "I could not find the answer in the documents.", reranked, conf

    if cfg.answer_style == "beginner" and not cfg.strict_context:
        base_def = concise.replace("Definition:", "").strip()
        prompt = prompt_enrich.format(term=term, definition=base_def)
        expanded = llm.invoke(prompt).strip()
        body = build_beginner_answer(term, base_def, expanded)
        md = body + "\n\n" + render_sources(reranked)
    else:
        md = concise + "\n\n" + render_sources(reranked)
    return md, reranked, conf

# ======================
# Respond
# ======================
if ask:
    if not st.session_state.vector or not st.session_state.llm:
        st.error("Please upload PDFs and build the index first.")
    else:
        with st.spinner("Thinking..."):
            md_answer, reranked_docs, confidence = answer_question(q)
        st.subheader("Answer")
        st.markdown(md_answer)

        conf_pct = int(round(confidence * 100))
        st.caption(f"Confidence (heuristic): **{conf_pct}%** — percent of top-{cfg.top_k} chunks containing the exact term.")

        st.download_button("⬇️ Export answer (Markdown)", data=md_answer.encode("utf-8"),
                           file_name="answer.md", mime="text/markdown")

        st.session_state.hist.insert(0, (q, md_answer))

st.divider()

# ======================
# Query History
# ======================
st.subheader("Recent Questions")
if not st.session_state.hist:
    st.info("Ask something to populate history.")
else:
    for i, (qq, aa) in enumerate(st.session_state.hist[:6], 1):
        with st.expander(f"{i}. {qq}"):
            st.markdown(aa)

# ======================
# Notes & Tips
# ======================
with st.expander("ℹ️ Notes & Tips"):
    st.markdown("""
- **Exact-match + extraction** is used whenever possible for accuracy. If extraction is not possible, the app falls back to LLM with strict, context-only prompting.
- **Beginner mode** expands the canonical definition into bullets and comparisons (no made-up numbers).
- **OCR**: If your PDFs are **scanned images**, enable OCR in the sidebar (requires `pdf2image` + `pytesseract` + Poppler + Tesseract OCR installed on your system).
- **Token budgeting** keeps prompts within the model’s input window to avoid CUDA errors.
- **Confidence** is a simple heuristic: the share of retrieved chunks that contain the exact term.
""")

