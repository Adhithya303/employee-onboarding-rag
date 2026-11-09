# Tulane University Faculty Onboarding Assistant (FastAPI + React + Conversational RAG)

## Important Dataset & Scope Note
Although the source PDF file is stored at `data/raw/tulane_staff_handbook.pdf`, inspection of the extracted text confirmed that the document is actually the **Tulane University Faculty Handbook** (Updated August 7, 2026; PDF pages 1–191) together with an appended **Tulane University Policies and Procedures for Responding to Allegations of Research Misconduct** (approved March 7, 2022; PDF pages 192–238).

- **Faculty Scope Only:** This system is a **Faculty Onboarding Assistant** scoped strictly to Tulane University faculty policies and research misconduct procedures in the supplied PDF. Faculty policies must **never** be presented as general staff policies.
- **Document Version Currency:** Answers reflect only the supplied document version (`Updated August 7, 2026` / `RMP March 7, 2022`), whose current live university policy status is unverified.
- **Local Single-User Demo Scope:** This web application is designed as a local single-user demo. Conversation IDs stored in local SQLite (`data/conversations/chat_history.db`) are convenience identifiers for switching and resuming chats, not an authentication or access-control mechanism.
- **Preserved Artifacts:** The original source PDF (`data/raw/tulane_staff_handbook.pdf`), raw page extractions (`data/processed/pages.json`, `data/processed/handbook.txt`), chunks (`data/processed/chunks.json`), and FAISS index (`data/index/`) are preserved intact.

---

## Architecture (CPU Local Retrieval + Hosted Gemini Generation + FastAPI + React)

1. **Stage 1 — PDF Extraction (`backend/extract_pdf.py`):**
   - Extracts text page by page (1-based page numbers) from `data/raw/tulane_staff_handbook.pdf` into `data/processed/pages.json` and `data/processed/handbook.txt`.
2. **Stage 2 — Preprocessing, Paragraph-Aware Chunking & CPU FAISS Indexing (`backend/preprocess_chunk.py`, `backend/build_index.py`):**
   - Strips recognizable standalone running headers/footers (`Tulane University Faculty Handbook | Page X` and `RMP Page X of 47`).
   - Excludes verified Table of Contents pages (`2, 3, 193`) and blank/header-only pages (`49, 50, 140–146`), recording all `12` skipped pages and reasons in `data/processed/ingestion_report.json`.
   - Keeps `Faculty Handbook (Main)` (pages `1–191`) and `Appended Research Misconduct Policy` (pages `192–238`) strictly separated.
   - Tokenizes with `sentence-transformers/all-MiniLM-L6-v2` (`max_seq_length = 256`, `device="cpu"`), generating `749` chunks (`<= 195` content tokens, max `242` total embedding input tokens) and building a normalized CPU `faiss.IndexFlatIP` cosine similarity index under `data/index/`.
3. **Stage 3 — Conversational RAG Engine & Terminal Apps (`backend/retriever.py`, `backend/conversation_store.py`, `backend/llm_client.py`, `backend/rag_engine.py`, `backend/search_demo.py`, `backend/chat_demo.py`):**
   - **Session-Scoped CPU Retrieval:** Loads the CPU `SentenceTransformer` model and saved FAISS index once per session (with compatibility and staleness fingerprint verification) and retrieves up to `8` deduplicated candidate chunks within a bounded token budget (`1,500` content tokens).
   - **SQLite Conversation Persistence:** Stores conversations, messages, answer statuses, and cited sources in `data/conversations/chat_history.db`.
   - **Bounded-History Follow-up Rewriting & Clarification:** Uses bounded recent conversation history to rewrite follow-up questions into standalone retrieval queries, preserves explicit topic changes, never invents missing details, and asks a clarification question when a reference is genuinely ambiguous.
   - **Grounded Answer Generation & Citation Validation:** Uses the official Google Gen AI SDK (`google-genai`) with explicit HTTP timeouts (`HttpOptions`), bounded transient retries, and structured JSON schemas. Answers strictly from the supplied `[S1]..[Sn]` handbook passages for the current turn, or states `"I couldn’t find that in the provided handbook."` when unsupported.
4. **Stage 4 — Local Web Application (`backend/main.py` + `frontend/`):**
   - **FastAPI Backend (`backend/main.py`):**
     - `GET /api/health`
     - `GET /api/ready`
     - `POST /api/conversations`
     - `GET /api/conversations`
     - `GET /api/conversations/{id}/messages` (accepts `source_mode: "handbook" | "uploaded_pdf"`)
     - `POST /api/conversations/{id}/upload` (chat-scoped multipart PDF upload with faculty relevance validation)
     - `GET /api/conversations/{id}/document` (retrieves status of uploaded PDF for this chat)
     - `DELETE /api/conversations/{id}/document` (removes uploaded PDF and its vector index)
     - Loads the CPU embedding model and saved FAISS index once during FastAPI `lifespan`, runs synchronous retrieval/LLM/SQLite work in worker threads (`anyio.to_thread.run_sync`) so the async event loop is never blocked, and serializes turns per conversation (`409 Conflict` on overlapping duplicate sends to the same conversation while allowing concurrent turns across different conversations).
   - **React + TypeScript + Vite Frontend (`frontend/`):**
     - Sidebar with `+ New Chat` and saved conversations, welcome screen with starter handbook questions, safe text rendering, `Enter` to send / `Shift+Enter` for newline, draft preservation on error, expandable citation cards (PDF pages, section, chunk ID, excerpt), `localStorage` active conversation restoration, and race-condition guards when switching chats mid-request.
     - **Chat-Scoped PDF Upload & Mode Selector:** Upload control for 1 searchable faculty PDF per chat (max 10 MiB, 50 pages), live processing/relevance feedback, removal confirmation, and a toggle between `Tulane Faculty Handbook` and `Uploaded PDF only` search modes.

5. **Stage 5 — Chat-Scoped Custom PDF Upload & Grounded QA (`backend/upload_processing.py`, `backend/upload_relevance.py`, `backend/upload_manager.py`):**
   - **File Validation:** Enforces strict 10 MiB streaming byte limit, `%PDF-` signature check, pypdf extraction, page count limit (50 pages), text length limits, and scanned/image-only rejection (no OCR).
   - **Faculty Relevance Gate:** Sends bounded text sample (beginning, middle, end, headings) to Gemini for structured `accept`, `reject`, or `needs_clarification` classification. Ignores untrusted instructions embedded in the PDF.
   - **Chunking & Indexing:** Chunks with `all-MiniLM-L6-v2` tokenizer without Tulane-specific exclusions, computes normalized float32 CPU embeddings, and publishes a chat-scoped FAISS `IndexFlatIP` index under `data/uploads/<document_id>/`.
   - **Strict Chat Isolation:** In `uploaded_pdf` mode, queries retrieve exclusively from the current chat's verified upload. If the answer is absent, returns `"I couldn't find that in the uploaded PDF."` Other chats cannot access or retrieve from that document.
   - **Document Removal:** Deleting the document purges disk artifacts and halts future retrieval while preserving existing transcript messages and citation cards.

---

## Important Note on Citations & Heading Hints
- **Citation Label Validation vs. Semantic Support:** `backend/rag_engine.py` programmatically verifies that every cited label (`[S1]`, `[S2]`, ...) belongs to the evidence chunks supplied in the current turn (and retries once if the model outputs an unknown label or invalid structure). However, **valid citation labels alone do not prove semantic support**—always verify policy claims against the displayed PDF page numbers and supporting excerpts.
- **Heading Hints:** Detected section headings in chunk metadata are extraction hints and may occasionally appear out of order due to floating frames in the source PDF stream; the passage text itself is always the primary evidence.

---

## Environment Configuration (`.env` and `frontend/.env.example`)

1. **Backend `.env` (Project Root):**
   ```powershell
   Copy-Item .env.example .env
   ```
   Edit `.env` and set your Google AI Studio key and model:
   ```ini
   GEMINI_API_KEY=your_actual_api_key
   GEMINI_MODEL=gemini-3.5-flash-lite
   GEMINI_TIMEOUT_SECONDS=30
   GEMINI_MAX_RETRIES=2
   ```
   - Never put `GEMINI_API_KEY` in `frontend/.env`.
2. **Frontend `frontend/.env` (Optional):**
   Defaults to `VITE_API_BASE_URL=http://127.0.0.1:8000`. If needed, copy `frontend/.env.example` to `frontend/.env`:
   ```powershell
   Copy-Item frontend\.env.example frontend\.env
   ```

---

## Exact Windows PowerShell Commands

Run all commands starting from the project root (`c:\Users\adhit\Downloads\employee-onboarding-rag`):

### 1. Install Backend Dependencies
```powershell
.\.venv\Scripts\pip install --extra-index-url https://download.pytorch.org/whl/cpu -r backend\requirements.txt
```

### 2. Start the FastAPI Backend (Terminal 1 — from Project Root)
```powershell
.\.venv\Scripts\python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
- Backend API & health check: `http://127.0.0.1:8000/api/health`
- Interactive OpenAPI docs: `http://127.0.0.1:8000/docs`

### 3. Install Frontend Dependencies & Start Vite Dev Server (Terminal 2)
```powershell
Set-Location frontend
npm install
npm run dev
```

### 4. Open the Local Web Application
Open your browser to:
```text
http://127.0.0.1:5173
```

### 5. Run Tests & Frontend Build Verification (from Project Root)
```powershell
# Run backend unit & FastAPI integration tests (mocked LLM + real CPU FAISS retrieval & SQLite)
.\.venv\Scripts\python -m pytest -v

# Run frontend TypeScript typecheck and production build
Set-Location frontend
npm run typecheck
npm run build
Set-Location ..
```

### 6. Existing Terminal Commands (Preserved)
```powershell
# Retrieval-only terminal search (no API key required)
.\.venv\Scripts\python backend\search_demo.py

# Interactive terminal conversational RAG chat
.\.venv\Scripts\python backend\chat_demo.py
```
