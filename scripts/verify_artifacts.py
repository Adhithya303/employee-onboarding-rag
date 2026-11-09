"""Verify the committed index bundle loads (manifest + fingerprint checks) and retrieves a cited passage.

Never rebuilds anything. Exits non-zero on any failure so a Render build fails fast.
Usage (repo root): python scripts/verify_artifacts.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.retriever import HandbookRetriever  # noqa: E402


def main() -> int:
    retriever = HandbookRetriever()  # runs compatibility + stale-fingerprint checks
    n = int(retriever.index.ntotal)
    assert n == len(retriever.metadata) == int(retriever.manifest["num_chunks"]), "chunk count mismatch"
    hits = retriever.retrieve("paid parental leave for faculty", top_k=3)
    assert hits, "no passages retrieved"
    top = hits[0]
    print(f"index OK: {n} chunks; top hit {top['chunk_id']} pages={top.get('pdf_pages')}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"ARTIFACT VERIFICATION FAILED: {exc}", file=sys.stderr)
        sys.exit(1)

