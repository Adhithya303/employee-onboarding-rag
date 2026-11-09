#!/usr/bin/env bash
# Render build script (Linux). Reproducible, CPU-only. Run from the repository root.
set -euo pipefail

export PIP_DISABLE_PIP_VERSION_CHECK=1
# HF_HOME is set in render.yaml for BOTH build and runtime so the model cache downloaded here is
# the one the running service reads. Fall back to a repo-local path if run manually.
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"

echo "==> 1/4 CPU-only PyTorch (installed first so later installs keep it)"
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple "torch==2.15.0"

echo "==> 2/4 Backend dependencies (pinned via constraints; torch already satisfied)"
python -m pip install -c backend/constraints-render.txt -r backend/requirements.txt

echo "==> 3/4 Verify CPU-only torch survived and cache the MiniLM model"
python - <<'PY'
import torch
assert "+cpu" in torch.__version__, f"Expected CPU torch build, got {torch.__version__}"
assert not torch.cuda.is_available()
print("torch", torch.__version__, "(cpu)")
from sentence_transformers import SentenceTransformer
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
print("MiniLM cached")
PY

echo "==> 4/4 Verify committed index artifacts load and retrieve (no rebuild)"
python scripts/verify_artifacts.py

