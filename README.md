# PDF RAG Streamlit App — Concepts & End‑to‑End Architecture

This document explains **every major concept** used in the PDF Q&A Streamlit app, plus the **complete end‑to‑end (E2E) architecture**. It’s meant as a companion to the codebase so new contributors (or future you) can quickly reason about how the system works and why it was designed this way.

---

## Table of Contents

1. [High‑Level Overview](#high-level-overview)
2. [End‑to‑End Dataflow](#end-to-end-dataflow)
3. [Core Concepts](#core-concepts)
   - [Retrieval‑Augmented Generation (RAG)](#retrieval-augmented-generation-rag)
   - [PDF Ingestion (PyPDF2) & OCR (pdf2image + pytesseract)](#pdf-ingestion-pypdf2--ocr-pdf2image--pytesseract)
   - [Chunking (RecursiveCharacterTextSplitter)](#chunking-recursivecharactertextsplitter)
   - [Embeddings (sentence-transformers)](#embeddings-sentence-transformers)
   - [Vector Store (FAISS)](#vector-store-faiss)
   - [Retrieval](#retrieval)
   - [Reranking (Lexical Heuristic)](#reranking-lexical-heuristic)
   - [Definition Extraction (Regex)](#definition-extraction-regex)
   - [Token Budgeting & Truncation](#token-budgeting--truncation)
   - [Language Model (Transformers + FLAN‑T5)](#language-model-transformers--flan-t5)
   - [LangChain Wrappers](#langchain-wrappers)
   - [Prompting Strategy](#prompting-strategy)
   - [Beginner‑Friendly Fallback Builder](#beginner-friendly-fallback-builder)
   - [Confidence Heuristic](#confidence-heuristic)
   - [Session State & Caching](#session-state--caching)
   - [GPU/CPU Detection](#gpucpu-detection)
4. [Streamlit UI Architecture](#streamlit-ui-architecture)
5. [Configuration & Tuning](#configuration--tuning)
6. [Persistence (Optional)](#persistence-optional)
7. [Security & Privacy](#security--privacy)
8. [Performance Notes](#performance-notes)
9. [Troubleshooting](#troubleshooting)
10. [Extensibility Roadmap](#extensibility-roadmap)
11. [Glossary](#glossary)

---

## High‑Level Overview

This app answers questions **from your own PDFs** using a **Retrieval‑Augmented Generation (RAG)** pipeline. We index PDFs into a vector database (FAISS), retrieve the most relevant chunks for a query, and then use a small instruction‑tuned LLM (FLAN‑T5) to produce a clear answer. To maximize factuality:

- We **rerank** candidates to favor **exact phrase matches**.
- We attempt **exact definition extraction** from “glossary‑style” text (high precision).
- If needed, we fall back to **context‑only prompting** and finally a **beginner‑friendly formatter** so the answer is never just a one‑word echo.

### Key Design Goals
- **Accuracy first** (extraction > generation)
- **Explainability** (citations: filename + page)
- **Robustness** (token budgeting to avoid CUDA errors)
- **Usability** (Streamlit UI, markdown export, history, debug views)

---

## End‑to‑End Dataflow

```text
        ┌───────────────────────────────┐
        │           User/Browser        │
        └───────────────┬──────────────┘
                        │  (Streamlit UI)
                        ▼
               ┌──────────────────┐
               │ PDF Upload (UI)  │
               └─────┬────────────┘
                     │  bytes
      ┌──────────────┴────────────────────────────┐
      │  Ingestion                                │
      │  • PyPDF2 text extraction                 │
      │  • (Optional) OCR: pdf2image + pytesseract│
      └──────────────┬────────────────────────────┘
                     │  per-page text → Documents{page_content, metadata}
                     ▼
            ┌─────────────────────────┐
            │ Chunking (Recursive…TS) │
            └──────────┬──────────────┘
                       │  chunked Documents (content + source,page)
                       ▼
              ┌───────────────────────┐
              │ Embeddings (SBERT)    │
              └──────────┬────────────┘
                         │ vectors
                         ▼
                ┌─────────────────────┐
                │ FAISS Vector Store  │
                └──────────┬──────────┘
                           │
            Query          │       Build
         ┌─────────────────┘
         │
         ▼
┌──────────────────────────────────┐
│ Retrieval (top‑K) + Reranking    │
└──────────┬───────────────────────┘
           │ docs
           ▼
  ┌──────────────────────────────┐
  │ Exact Definition Extraction  │───► If success → Enrich (beginner) → Answer
  └──────────────────────────────┘
           │ else
           ▼
  ┌──────────────────────────────────────────┐
  │ Token‑Budgeted Context + Strict Prompt   │───► LLM (FLAN‑T5) → (optional Enrich) → Answer
  └──────────────────────────────────────────┘

Answer + Citations + Confidence + Export
```

---

## Core Concepts

### Retrieval‑Augmented Generation (RAG)
RAG combines **information retrieval** with **text generation**. The model doesn’t “remember” your PDFs; instead it gets a **fresh context window** containing the most relevant snippets, then **generates** an answer grounded in that context. This avoids training/fine‑tuning and helps reduce hallucinations.

### PDF Ingestion (PyPDF2) & OCR (pdf2image + pytesseract)
- **PyPDF2** extracts text from text‑based PDFs (fast and reliable when source is digital).  
- **OCR path** (optional): For scanned PDFs, pages are rasterized with **pdf2image** and text is recognized with **pytesseract**. OCR is slower and noisier but necessary when the PDF contains only images.  
- Each page becomes a **LangChain `Document`** with `page_content` and `metadata` (`source` filename, `page` number).

### Chunking (RecursiveCharacterTextSplitter)
We split long documents into overlapping **chunks** (e.g., 400 chars with 50 overlap). Chunking:
- Provides smaller semantic units for retrieval.
- Overlap preserves context at boundaries.
- Uses **RecursiveCharacterTextSplitter** to avoid splitting inside words or key structures when possible.

### Embeddings (sentence-transformers)
- We use **`sentence-transformers/all-MiniLM-L6-v2`** to encode chunks to dense vectors.
- Embeddings capture semantic similarity (not exact string match), enabling robust retrieval even when the query phrasing differs from the source text.
- Normalize embeddings for stable cosine similarity.

### Vector Store (FAISS)
- **FAISS** is a fast vector search library (CPU here).  
- It indexes embedding vectors and supports top‑K similarity search (k‑nearest neighbors).
- We store the original chunk (`Document`) alongside its vector for later context assembly and citation.

### Retrieval
- For a user query, we compute its embedding and retrieve **Top‑K** candidate chunks (e.g., 10).  
- Higher **K** improves recall but can increase latency and context length.

### Reranking (Lexical Heuristic)
To fight “nearest‑neighbor but not exact” issues in definitions, we apply a **lexical reranker**:
- Heavy bonus for **exact phrase presence** (term string found in the text).  
- Additional bonus for **token coverage** (query tokens found in text).  
- This surfaces **dictionary‑style entries** first.

> Why this matters: If the document is a glossary, the *exact* definition is usually present as a short sentence following the term. Reranking moves it to the top.

### Definition Extraction (Regex)
We parse “term + definition” patterns using regex that tolerates:
- Optional numbering (`80.`), punctuation (`: – —`), whitespace/newlines.  
- We grab the **first full sentence** after the term.  
- Example match:  
  `“Transactions Account: Deposit account on which a customer can write checks.”`

If extraction succeeds, we treat that as the **canonical definition** (high precision).

### Token Budgeting & Truncation
LLMs have a **max input length**. To avoid CUDA device‑side asserts and truncation bugs:
- We read the tokenizer’s `model_max_length`, reserve some tokens for instructions, and allocate the rest for **context**.
- We **pack retrieved chunks** until the budget is filled, truncating the last one if needed.
- This keeps the prompt within safe limits.

### Language Model (Transformers + FLAN‑T5)
- We use a **seq2seq instruction‑tuned model** (e.g., `google/flan-t5-base`).  
- FLAN‑T5 performs well for QA and follows instructions better than GPT‑2‑style decoders in short‑context RAG.  
- Loaded via **Hugging Face Transformers** with a **text2text‑generation pipeline**.

### LangChain Wrappers
- **`HuggingFaceEmbeddings`**: standard embeddings interface (model name + device + kwargs).  
- **`FAISS`**: vector store integration (`from_documents`, `similarity_search`).  
- **`PromptTemplate`**: parameterized prompts for strict QA and enrichment.  
- **`HuggingFacePipeline`**: wraps a Transformers pipeline as a LangChain LLM.

### Prompting Strategy
We use a **two‑stage** approach:

1) **Strict, context‑only definition**  
   - “Answer only using the context; if not present, say: *I could not find the answer…*”  
   - Constrains the model to avoid hallucinations.

2) **Beginner‑friendly enrichment** (optional)  
   - Given the canonical definition, ask the LLM to **restate** and **explain in bullets and comparisons**.  
   - Forbid numeric invention (rates, fees) to keep it safe.

### Beginner‑Friendly Fallback Builder
Even if the LLM ignores formatting, we guarantee a usable answer:
- Check if generated text matches the required format and length.  
- If not, **synthesize a structured answer** from the canonical definition only (no fabricated numbers).  
- This ensures the user never sees a bare echo like “Transactions Account”.

### Confidence Heuristic
A lightweight confidence signal: **fraction of top‑K chunks containing the exact term**.  
- Not a calibrated probability, but useful for UX (“search found multiple direct matches” vs “only semantic matches”).

### Session State & Caching
- **`st.session_state`** holds the vector store, tokenizer, histories.  
- **`@st.cache_resource`** caches embeddings model and LLM across reruns to avoid reload costs.  
- Rebuild index only when inputs change (new PDFs or settings).

### GPU/CPU Detection
- The app auto‑detects CUDA via `torch.cuda.is_available()` and configures device indices accordingly.  
- FAISS runs CPU by default; the GPU log warning is benign unless you explicitly require FAISS‑GPU.

---

## Streamlit UI Architecture

**Sidebar:**  
- File uploader (multi‑PDF), **OCR toggle**.  
- Model pickers (FLAN‑T5 size), embedding model.  
- Tuning sliders: chunk size/overlap, Top‑K, max new tokens.  
- **Strict context** toggle, **Answer style** (beginner/concise).  
- Sampling controls (optional).  
- **Debug mode**, **Rebuild Index** button.

**Main area:**  
- KPIs (device, pages, chunks, context budget).  
- **Ask box** and **Answer** panel.  
- **Citations** (filename + page).  
- **Confidence** heuristic.  
- **Export** answer to Markdown.  
- **Query history** expandable list.  
- **Notes & Tips** expander.

UX touches:
- Term **highlighting** in debug snippets.  
- ASCII‑style codeboxes for previews.  
- Clear distinctions between extraction, strict QA, and enriched outputs.

---

## Configuration & Tuning

- **Chunk Size / Overlap**: Smaller chunks give precise retrieval for glossary‑style definitions (e.g., 300–600 chars).  
- **Top‑K**: 8–12 is a good range; larger K improves recall but risks context bloat.  
- **Model Size**: `flan‑t5‑base` balances speed/quality; `large` improves quality on GPUs.  
- **Max New Tokens**: 128–256 is enough for definitions + bullets.  
- **Strict Context**: Enable for compliance; disable to allow enriched explanations.  
- **Sampling/Temperature**: Keep off for deterministic answers; enable only if you want stylistic variety.

---

## Persistence (Optional)

Speed up restarts by **saving/loading** the FAISS index:

```python
# Save (after building)
vector.save_local("rag_index")

# Load (at startup)
from langchain_community.vectorstores import FAISS
emb = get_embeddings(cfg.emb_model_name, DEVICE_STR)
vector = FAISS.load_local("rag_index", emb, allow_dangerous_deserialization=True)
```

> Use the **same embedding model** between save/load.

---

## Security & Privacy

- PDFs are processed **locally**; no external calls at runtime beyond initial model downloads.  
- Avoid uploading sensitive documents to shared machines.  
- Clear HF cache (`~/.cache/huggingface`) if needed.  
- If enabling OCR, note that OCR output might contain transcription errors—validate before sharing.

---

## Performance Notes

- Prefer **text‑based PDFs** (OCR only when necessary).  
- Keep chunk size reasonable; very large chunks reduce retriever precision.  
- Use a GPU for faster generation; CPU works but is slower.  
- Reduce **Top‑K** or **Max New Tokens** if latency is high.  
- Consider a **cross‑encoder reranker** for higher‑quality ranking (at higher compute cost).

---

## Troubleshooting

- **“Failed to load GPU Faiss”** → Harmless on CPU builds.  
- **CUDA device‑side assert** → Token budget exceeded; this app packs context to avoid it. If you change models, keep K/size modest.  
- **OCR not working** → Ensure `pdf2image`, `pytesseract`, **Poppler**, and **Tesseract OCR** are installed and on `PATH`. Restart shell.  
- **Empty answers** → Use Debug mode; verify the PDFs actually contain the term; adjust chunking/Top‑K.  
- **Wrong page in citation** → Try smaller chunks so glossary entries stay intact; increase Top‑K to capture the right page.

---

## Extensibility Roadmap

- **Clickable citations** that open PDFs at page offsets (JS viewer integration).  
- **Cross‑encoder reranking** (e.g., `ms-marco` cross‑encoders) for sharper top‑K ordering.  
- **UI persistence** and **index save/load** controls.  
- **Batch Q&A** with CSV/Markdown exports.  
- Support DOCX/HTML ingestion; add page‑image thumbnails in UI.  
- **Docker** packaging (Poppler/Tesseract included).

---

## Glossary

- **RAG**: Retrieval‑Augmented Generation—look up relevant text, then generate an answer with it.  
- **Embedding**: Numeric vector representation of text capturing semantic meaning.  
- **FAISS**: Facebook AI Similarity Search, fast vector database for nearest‑neighbor lookups.  
- **Chunking**: Splitting long documents into smaller overlapping pieces for retrieval.  
- **Reranking**: Re‑ordering retrieved chunks using heuristics or models to surface better candidates.  
- **Regex**: Regular expressions; here used to parse “Term: Definition.” patterns robustly.  
- **Token Budgeting**: Ensuring the prompt + context stays under the model’s max length.  
- **FLAN‑T5**: Instruction‑tuned seq2seq LLM; good at following prompts for QA tasks.  
- **Strict Context**: Constrains model to only use provided context; no outside knowledge.  
- **Beginner Mode**: Restates the canonical definition, adds plain‑English bullets and comparisons.  
- **OCR**: Optical Character Recognition—extracts text from images of document pages.  
- **Streamlit**: Python UI framework for data apps with reactive components.  
- **LangChain**: Library providing abstractions for LLM apps (LLMs, prompts, vector stores, chains).

---

## Appendix: Key Functions (Mapping to Code)

- `ingest_pdf_bytes(...)`: Load PDF bytes → per‑page `Document`s (OCR or PyPDF2).  
- `preprocess_documents(...)`: Chunk pages preserving metadata.  
- `get_embeddings(...)`: Load Sentence‑Transformers model (cached).  
- `build_vector_store(...)`: Build FAISS index from chunked `Document`s.  
- `get_llm(...)`: Load FLAN‑T5 via Transformers pipeline (cached).  
- `pack_context_to_token_budget(...)`: Assemble safe‑length context.  
- `rerank_docs_for_term(...)`: Lexical reranker for exact match prioritization.  
- `try_extract_definition(...)`: Regex‑based definition extraction.  
- `make_prompt_strict(...)`, `make_prompt_enrich(...)`: Prompt templates.  
- `build_beginner_answer(...)`: Guaranteed structured fallback.  
- `answer_question(...)`: Orchestrates retrieval → extraction or strict QA → enrichment → answer.  

---
