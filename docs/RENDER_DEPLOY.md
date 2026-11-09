# Deploying the backend on Render

Scope: **Tulane University Faculty Handbook** (+ appended Research Misconduct Policy). Not a staff handbook.

## How the index and model reach Render (path chosen: versioned index bundle)

Nothing is rebuilt on Render. The repo already tracks the verified bundle, and `render.yaml` builds from it:

| Needed at runtime | Where | Notes |
|---|---|---|
| `data/index/faiss.index`, `chunks_metadata.json`, `manifest.json` | Git | loaded once at startup |
| `data/processed/pages.json`, `chunks.json` | Git | **Required**: the loader hashes them against `manifest.json` (stale-index check) |
| `data/raw/*.pdf`, `handbook.txt`, `ingestion_report.json` | Git | not read at runtime (PDF only needed to regenerate) |
| MiniLM `all-MiniLM-L6-v2` | downloaded in build into `HF_HOME=/opt/render/project/src/.cache/huggingface` | same env var is used at runtime, so no download on boot |

- `.gitattributes` (`data/** -text`) is **mandatory**. The manifest fingerprints are hashes of CRLF files created on Windows,
  but Git had stored them as LF. Without it a Linux checkout fails with "Stale index detected" (verified, see below).
- `scripts/verify_artifacts.py` runs at the end of the build and fails the deploy if the index does not load/retrieve.
- **Redistribution blocker (yours to confirm):** the PDF is committed. Make sure the repo is private, or that you may
  redistribute the handbook. If not, remove `data/raw/` from Git (the runtime does not need it); the index text chunks still contain handbook text.
- To refresh content: rebuild locally (`extract_pdf.py`, `preprocess_chunk.py`, `build_index.py`), commit all of `data/`, redeploy.

## Plan, memory and storage

- **Free (512 MB, ephemeral disk, spins down after 15 min idle):** will not work. Measured on Windows: the embedding
  model + FAISS index alone use ~515 MB resident (before FastAPI, Gemini SDK, or requests). Chats would also be wiped on every restart, redeploy and spin-down, with ~30-60 s cold starts.
- **Starter (512 MB, supports a disk):** same memory problem, expect out-of-memory restarts. Linux numbers may differ somewhat, so you can try it, but I do not expect it to hold.
- **Standard (2 GB, `render.yaml` default):** lowest plan I recommend. Confirm actual usage in the Render Metrics tab after deploy; if it is well under 1 GB you could try downgrading.
- **Chat & Upload Storage:** SQLite at `/var/data/chat_history.db` and custom upload vector indexes at `/var/data/uploads`
  on the 1 GB disk (`CONVERSATION_DB_PATH` and automatic `UPLOAD_ARTIFACTS_DIR`). They survive restarts and redeploys.
  A service with a disk cannot scale to multiple instances and deploys have brief downtime. Without the disk (free plan,
  or `CONVERSATION_DB_PATH` unset) chats, citations, and uploaded document indexes disappear on restart.
- One Uvicorn worker only (model/index loaded once).
- Local default is unchanged: `data/conversations/chat_history.db` and `data/uploads/`; your local DB and uploads are gitignored.

## Access control (important)

The API has **no users**: conversation IDs are not access control, and `GET /api/conversations` lists every chat. A public URL
also allows anyone to spend your Gemini quota.

- **Private demo (implemented):** set `ACCESS_TOKEN` on Render. Every `/api/*` request except `/api/health` and `/api/ready`
  must send header `X-Access-Token: <token>`. Unset = open (local default).
  The web frontend now includes an in-app "🔑 Access Token" configuration dialog and automatic prompt upon 401s, allowing
  testers to enter the token once on their device (stored in `localStorage`) without baking any secrets into the APK build.
- **Public / multi-user (not implemented):** add real authentication (e.g. OIDC/Firebase/Auth0 JWT), store `owner_id` on
  conversations, filter list/get/send by owner, add rate limits and Gemini spend caps.
- **CORS** is separate: explicit origins only (`CORS_ORIGINS`). Android Capacitor WebView origin is `https://localhost`
  by default (`capacitor://localhost` for iOS). Add your final web origin if you host the frontend.

## Dashboard steps

1. Push the repo (including `render.yaml`, `.gitattributes`, `data/`) to GitHub.
2. Render → **New → Blueprint** → pick the repo/branch → Render reads `render.yaml`. (Or **New → Web Service** and enter the values below manually.)
3. Enter the secrets when prompted (`GEMINI_API_KEY`, `ACCESS_TOKEN`). Review plan = Standard and the disk.
4. Create. Watch the build log for `torch ... (cpu)`, `MiniLM cached`, `index OK: 749 chunks`.

| Setting | Value |
|---|---|
| Repository / Branch | your GitHub repo / `main` |
| Root Directory | *(blank = repo root)* |
| Runtime | Python 3 |
| Build Command | `bash scripts/render_build.sh` |
| Start Command | `python -m uvicorn backend.main:app --host 0.0.0.0 --port "$PORT"` |
| Health Check Path | `/api/ready` (503 until index + Gemini config are usable; `/api/health` always returns 200) |
| Plan | Standard |
| Disk | name `chat-data`, mount `/var/data`, 1 GB |
| Auto-Deploy | Off (until verified) |

## Environment variables

| Key | Value | Secret? | Needed at |
|---|---|---|---|
| `PYTHON_VERSION` | `3.11.11` | no | build |
| `HF_HOME` | `/opt/render/project/src/.cache/huggingface` | no | build **and** runtime |
| `TOKENIZERS_PARALLELISM` | `false` | no | runtime |
| `GEMINI_API_KEY` | your key | **yes** (`sync:false`) | runtime |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | no | runtime |
| `GEMINI_TIMEOUT_SECONDS` | `30` | no | runtime |
| `GEMINI_MAX_RETRIES` | `2` | no | runtime |
| `CONVERSATION_DB_PATH` | `/var/data/chat_history.db` | no | runtime |
| `CORS_ORIGINS` | `https://localhost,capacitor://localhost,http://localhost,http://localhost:5173,http://127.0.0.1:5173` | no | runtime |
| `ACCESS_TOKEN` | long random string (e.g. `python -c "import secrets;print(secrets.token_urlsafe(32))"`) | **yes** | runtime |

`PORT` is set by Render automatically; do not add it. The service has no `.env` on Render, so environment variables are the only config.

**After deploy** (add/change later as needed):
- Frontend/APK build env (not on Render): `VITE_API_BASE_URL=https://<service>.onrender.com`. Keep the LAN value in `frontend/.env` for local testing (e.g. use `.env.production.local` or pass it on the command line for the APK build).
- If you host the web frontend somewhere, append its origin to `CORS_ORIGINS`.
- To rotate the token/key: edit the variable in Render → it redeploys.
- Optional hardening later: `HF_HUB_OFFLINE=1` (runtime) once the first deploy proves the cache is accessible.

## Post-deploy smoke test (replace URL/token)

```powershell
$u="https://<service>.onrender.com"; $h=@{"X-Access-Token"="<ACCESS_TOKEN>"}
Invoke-RestMethod "$u/api/health"                      # status ok, index_loaded true, chunk_count 749
Invoke-RestMethod "$u/api/ready"                       # ready true
Invoke-RestMethod "$u/api/conversations"               # expect 401 without header (if ACCESS_TOKEN set)
$c = Invoke-RestMethod -Method Post "$u/api/conversations" -Headers $h -ContentType application/json -Body "{}"
$id = $c.conversation_id
$q1 = Invoke-RestMethod -Method Post "$u/api/conversations/$id/messages" -Headers $h -ContentType application/json -Body '{"message":"How much paid parental leave do faculty get?"}'
$q1.status; $q1.cited_sources | select label,pdf_pages
$q2 = Invoke-RestMethod -Method Post "$u/api/conversations/$id/messages" -Headers $h -ContentType application/json -Body '{"message":"Who approves it?"}'
$q2.answer
$r = Invoke-RestMethod "$u/api/conversations/$id/messages" -Headers $h
$r.messages.Count; $r.messages[-1].cited_sources.Count   # 4 messages, citations present
```
Persistence check: Render → Manual Deploy/Restart, then repeat the last GET: the conversation should still exist (disk).

## Criteria to hand the HTTPS URL to the APK build

1. `/api/ready` returns 200 and `chunk_count` is 749.
2. Parental-leave answer has cited pages; "Who approves it?" resolves to the same topic; reload keeps citations.
3. Conversation still present after a restart.
4. Request without token returns 401 (if the private demo token is enabled) and the cafeteria question is answered as unsupported.
5. The app has a way to enter the token (or you consciously accept an open API). Then: set `VITE_API_BASE_URL` to the URL, `npm run build`, `npx cap sync android`, rebuild the APK.

