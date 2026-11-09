"""
Stage 2A: Conservative text cleaning and paragraph-aware chunking for the
Tulane University Faculty Onboarding Assistant.

Dataset Note:
The source PDF (`data/raw/tulane_staff_handbook.pdf`) is the Tulane University Faculty Handbook
(pages 1-191) with an appended Research Misconduct Policy (pages 192-238). Faculty policies
must never be presented as general staff policies.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_FILE_REL = "data/raw/tulane_staff_handbook.pdf"
DOCUMENT_TITLE = "Tulane University Faculty Handbook"

SECTION_MAIN_HANDBOOK = "Faculty Handbook (Main)"
SECTION_APPENDED_RMP = "Appended Research Misconduct Policy"
EXPECTED_APPENDIX_START_PAGE = 192

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PAGES_JSON_PATH = PROCESSED_DIR / "pages.json"
CHUNKS_JSON_PATH = PROCESSED_DIR / "chunks.json"
INGESTION_REPORT_PATH = PROCESSED_DIR / "ingestion_report.json"

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
TARGET_CONTENT_TOKENS = 180
MAX_CONTENT_TOKENS = 195
OVERLAP_TOKENS = 30

# Recognizable standalone running headers/footers
STANDALONE_HEADER_FOOTER_RE = re.compile(
    r"^(?:Tulane University Faculty Handbook\s*\|\s*Page\s+\d+|RMP Page\s+\d+\s+of\s+\d+)$"
)

# Dotted leader lines found only in Table of Contents blocks (e.g., "1.1 Mission Statement ....... 4")
TOC_DOTTED_LINE_RE = re.compile(r"\.{5,}\s*\d+\s*(?:\d+)?\s*$")
TOC_HEADER_LINE_RE = re.compile(r"^(?:Table of Contents|Contents)$", re.IGNORECASE)
HORIZONTAL_RULE_RE = re.compile(r"^_{10,}$")

# Top-level heading patterns (Chapter, Part, Appendix, Senate Amendment, RMP Title, Endnotes)
CHAPTER_START_RE = re.compile(r"^\*?\s*CHAPTER\s+\d+[:.]\s+.+$")
PART_HEADING_RE = re.compile(r"^PART\s+[A-D](?:-\d+)?$")
APPENDIX_HEADING_RE = re.compile(r"^APPENDIX$")
SENATE_AMENDMENT_RE = re.compile(r"^AMENDMENT TO THE TULANE SENATE BY-LAWS:.*$")
RMP_MAIN_TITLE_RE = re.compile(
    r"^9\.6\s+Tulane University Policies and Procedures for Responding to Allegations.*$"
)
ENDNOTES_HEADING_RE = re.compile(r"^ENDNOTES$")

# Sub-heading patterns (numbered sections, Roman numeral sections, lettered subsections)
NUMBERED_HEADING_RE = re.compile(r"^(\d+\.\d+(?:\.\d+)?\.?)\s+([A-Z][^\n]{1,95})$")
ROMAN_HEADING_RE = re.compile(r"^([IVXLC]+)\.\s+([A-Z][^\n]{2,85})$")
LETTER_HEADING_RE = re.compile(r"^([A-N])\.\s+([A-Z][A-Za-z0-9\s,;:/&()–—'-]{2,80})$")

# Sentence verbs/phrases that indicate a line is a prose sentence or definition rather than a heading
NON_HEADING_PHRASE_RE = re.compile(
    r"\b(?:means|shall|will|must|may|is the|are the|includes|applies to|provides that)\b",
    re.IGNORECASE,
)

# List item / paragraph start patterns
LIST_ITEM_START_RE = re.compile(
    r"^(?:[•°o]\s+|\([a-z0-9]+\)\s+|\d+\)\s+|\d+\.\s+[A-Za-z]|[a-z]\.\s+[A-Za-z]|[ivx]+\.\s+[A-Za-z])"
)


@dataclass
class TextUnit:
    """A paragraph or sentence-split segment with its page and heading provenance."""

    text: str
    page: int
    document_section: str
    heading: str | None
    token_count: int


def format_embedding_input(
    document_title: str,
    document_section: str,
    heading: str | None,
    text: str,
) -> str:
    """
    Construct the complete string passed to the embedding model.
    Includes document title, section, and detected heading (if any) so the vector captures
    structural context while guaranteed to fit within `model.max_seq_length`.
    """
    header_parts = [f"{document_title} ({document_section})"]
    if heading:
        header_parts.append(f"Section: {heading}")
    prefix = " | ".join(header_parts)
    return f"{prefix}\n{text}"


def verify_section_boundary(pages_data: list[dict[str, Any]]) -> int:
    """
    Verify from actual extracted content where the appended Research Misconduct Policy starts.
    Confirm that it matches PDF page 192.
    """
    detected_start: int | None = None
    for entry in pages_data:
        page_num = entry["page"]
        raw_text = entry["text"]
        if "RMP Page 1 of 47" in raw_text and "Responding to Allegations" in raw_text:
            detected_start = page_num
            break

    if detected_start is None:
        raise ValueError(
            "Could not verify the start of the appended Research Misconduct Policy "
            "('RMP Page 1 of 47') in pages.json."
        )

    if detected_start != EXPECTED_APPENDIX_START_PAGE:
        raise ValueError(
            f"Expected appended Research Misconduct Policy at page {EXPECTED_APPENDIX_START_PAGE}, "
            f"but detected at page {detected_start}."
        )

    return detected_start


def clean_page_conservatively(
    page_num: int,
    raw_text: str,
) -> tuple[str, str | None, dict[str, Any]]:
    """
    Conservatively clean a single page's text:
    - Strip only recognizable standalone running header/footer lines.
    - Detect and exclude full Table of Contents pages (pages 2, 3, 193).
    - Strip partial Table of Contents dotted-leader blocks on hybrid pages (pages 192, 194)
      without removing policy metadata or body content.
    - Do not blindly join broken words or delete text after a footer.
    - Return (cleaned_text, skip_reason_or_None, page_cleaning_details).
    """
    lines = [line.strip() for line in raw_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]

    removed_headers_footers: list[str] = []
    content_lines: list[str] = []

    for line in lines:
        if STANDALONE_HEADER_FOOTER_RE.match(line):
            removed_headers_footers.append(line)
        else:
            # Collapse multiple horizontal spaces/tabs conservatively
            normalized_line = re.sub(r"[ \t\f\v]+", " ", line).strip()
            content_lines.append(normalized_line)

    non_empty_lines = [l for l in content_lines if l]
    if not non_empty_lines:
        return "", "blank_or_header_footer_only", {
            "removed_headers_footers": removed_headers_footers,
            "removed_toc_lines_count": 0,
        }

    # Check for Table of Contents lines
    dotted_indices = [i for i, l in enumerate(non_empty_lines) if TOC_DOTTED_LINE_RE.search(l)]

    # Pure TOC page check: pages 2, 3, and 193 consist entirely of TOC entries (plus wrapped TOC titles)
    if dotted_indices:
        # On pages 2 and 3, or page 193, either starts with "Table of Contents" or almost all lines are TOC entries
        starts_with_toc_header = TOC_HEADER_LINE_RE.match(non_empty_lines[0]) is not None
        dotted_ratio = len(dotted_indices) / len(non_empty_lines)
        if starts_with_toc_header or dotted_ratio >= 0.65:
            return "", "table_of_contents", {
                "removed_headers_footers": removed_headers_footers,
                "removed_toc_lines_count": len(non_empty_lines),
            }

        # Hybrid page with partial TOC block (e.g., page 192 has policy metadata then "Contents",
        # page 194 has 3 trailing TOC dotted lines before "I. Principles")
        filtered_lines: list[str] = []
        removed_toc_count = 0
        in_toc_block = False
        for line in content_lines:
            if not line:
                if not in_toc_block:
                    filtered_lines.append(line)
                continue
            if HORIZONTAL_RULE_RE.match(line):
                removed_toc_count += 1
                continue
            if TOC_HEADER_LINE_RE.match(line):
                in_toc_block = True
                removed_toc_count += 1
                continue
            if TOC_DOTTED_LINE_RE.search(line):
                removed_toc_count += 1
                continue
            if in_toc_block:
                # On page 192, everything after "Contents" to the end of page 192 is TOC
                removed_toc_count += 1
                continue
            filtered_lines.append(line)

        content_lines = filtered_lines

    cleaned_text = "\n".join(content_lines)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text).strip()

    if not re.sub(r"\s+", "", cleaned_text):
        return "", "blank_or_header_footer_only", {
            "removed_headers_footers": removed_headers_footers,
            "removed_toc_lines_count": 0,
        }

    return cleaned_text, None, {
        "removed_headers_footers": removed_headers_footers,
        "removed_toc_lines_count": len(dotted_indices),
    }


def is_reliable_heading(line: str, next_line: str | None = None) -> tuple[str | None, str | None, bool]:
    """
    Detect whether `line` (optionally combined with `next_line` for 2-line ALL-CAPS CHAPTER headers)
    is a reliable heading.
    Returns (heading_level, heading_text, consumed_next_line) where heading_level is
    'top', 'sub', or None.
    """
    stripped = line.strip()
    if not stripped:
        return None, None, False

    # 1. Top-level Chapter headings (may wrap onto a second ALL-CAPS line)
    if CHAPTER_START_RE.match(stripped):
        clean_chap = stripped.lstrip("* ").strip()
        if (
            next_line
            and next_line.strip()
            and next_line.strip().isupper()
            and len(next_line.strip()) <= 60
            and not CHAPTER_START_RE.match(next_line.strip())
        ):
            clean_chap = f"{clean_chap} {next_line.strip()}"
            return "top", clean_chap, True
        return "top", clean_chap, False

    if PART_HEADING_RE.match(stripped):
        return "top", stripped, False

    if APPENDIX_HEADING_RE.match(stripped):
        return "top", stripped, False

    if SENATE_AMENDMENT_RE.match(stripped):
        if next_line and next_line.strip().startswith("Discrimination"):
            return "top", f"{stripped} {next_line.strip()}", True
        return "top", stripped, False

    if RMP_MAIN_TITLE_RE.match(stripped):
        if next_line and "Research Misconduct" in next_line:
            return "top", f"{stripped} {next_line.strip()}".rstrip("*"), True
        return "top", stripped.rstrip("*"), False

    if ENDNOTES_HEADING_RE.match(stripped):
        return "top", stripped, False

    # Exclude lines that end with sentence period or contain prose verbs
    if stripped.endswith(".") or NON_HEADING_PHRASE_RE.search(stripped):
        return None, None, False

    # 2. Sub-level headings: Numbered sections (e.g., "1.1 Mission Statement", "10.3.2 Eligibility...")
    if NUMBERED_HEADING_RE.match(stripped):
        return "sub", stripped, False

    # 3. Sub-level headings: Roman numeral sections (e.g., "I. Principles", "X. Conducting the Investigation")
    if ROMAN_HEADING_RE.match(stripped):
        return "sub", stripped, False

    # 4. Sub-level headings: Lettered subsections (e.g., "A. Reporting Misconduct", "B. Information Required")
    if LETTER_HEADING_RE.match(stripped):
        return "sub", stripped, False

    return None, None, False


def segment_section_into_paragraphs(
    section_pages: list[tuple[int, str]],
    document_section: str,
) -> list[tuple[str, int, str | None]]:
    """
    Convert cleaned pages within a single document section into paragraph blocks:
    Returns list of (paragraph_text, page_num, active_heading).
    Never crosses the document_section boundary.
    """
    top_heading: str | None = None
    sub_heading: str | None = None

    def current_heading() -> str | None:
        if top_heading and sub_heading:
            return f"{top_heading} > {sub_heading}"
        return sub_heading or top_heading

    paragraphs: list[tuple[str, int, str | None]] = []

    for page_num, page_text in section_pages:
        if page_num == 1 and top_heading is None:
            top_heading = "Title Page"

        lines = page_text.split("\n")
        current_lines: list[str] = []
        para_heading: str | None = current_heading()

        def flush_paragraph() -> None:
            nonlocal current_lines, para_heading
            if current_lines:
                joined = "\n".join(current_lines).strip()
                if joined:
                    paragraphs.append((joined, page_num, para_heading))
                current_lines = []
            para_heading = current_heading()

        idx = 0
        while idx < len(lines):
            line = lines[idx].strip()
            next_line = lines[idx + 1].strip() if idx + 1 < len(lines) else None

            # Blank line marks an explicit paragraph boundary
            if not line:
                flush_paragraph()
                idx += 1
                continue

            # Check if this line is a reliable heading
            h_level, h_text, consumed_next = is_reliable_heading(line, next_line)
            if h_level is not None and h_text is not None:
                flush_paragraph()
                if h_level == "top":
                    top_heading = h_text
                    sub_heading = None
                else:
                    sub_heading = h_text
                para_heading = current_heading()
                if consumed_next and next_line is not None:
                    current_lines.append(f"{line}\n{next_line}")
                    idx += 2
                else:
                    current_lines.append(line)
                    idx += 1
                continue

            # Check if this line starts a new bullet/numbered list item
            if LIST_ITEM_START_RE.match(line) and current_lines:
                flush_paragraph()
                current_lines.append(line)
                idx += 1
                continue

            # Check if the previous line ended a paragraph (ended with '.', '.”', '."', ':' and was shorter
            # than a typical wrapped line or next line starts with a capital letter after a short line)
            if current_lines:
                prev_line = current_lines[-1]
                if (
                    re.search(r'[.!?]["”\']?$', prev_line)
                    and len(prev_line) < 68
                    and re.match(r"^[A-Z0-9“\"(]", line)
                ):
                    flush_paragraph()

            current_lines.append(line)
            idx += 1

        flush_paragraph()

    return paragraphs


def count_tokens(tokenizer: Any, text: str, add_special_tokens: bool = False) -> int:
    """Count tokens using the model's tokenizer without truncation or false-alarm warnings."""
    encoded = tokenizer(
        text,
        add_special_tokens=add_special_tokens,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
        verbose=False,
    )
    return len(encoded["input_ids"])


def split_oversized_paragraph(
    para_text: str,
    page_num: int,
    document_section: str,
    heading: str | None,
    tokenizer: Any,
    max_unit_tokens: int,
) -> list[TextUnit]:
    """
    Split an oversized paragraph at sentence boundaries when possible, with a token-based fallback
    so that every resulting TextUnit has <= max_unit_tokens content tokens.
    """
    tok_len = count_tokens(tokenizer, para_text, add_special_tokens=False)
    if tok_len <= max_unit_tokens:
        return [
            TextUnit(
                text=para_text,
                page=page_num,
                document_section=document_section,
                heading=heading,
                token_count=tok_len,
            )
        ]

    # First try splitting at sentence boundaries (keeping punctuation attached)
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9“\"(\[])", para_text)
    if len(sentences) == 1:
        # Fallback to line boundaries if no sentence boundary matched
        sentences = [s.strip() for s in para_text.split("\n") if s.strip()]

    units: list[TextUnit] = []
    current_sents: list[str] = []

    def flush_sents() -> None:
        nonlocal current_sents
        if not current_sents:
            return
        combined = " ".join(current_sents).strip()
        toks = count_tokens(tokenizer, combined, add_special_tokens=False)
        units.append(
            TextUnit(
                text=combined,
                page=page_num,
                document_section=document_section,
                heading=heading,
                token_count=toks,
            )
        )
        current_sents = []

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        sent_tok_len = count_tokens(tokenizer, sent, add_special_tokens=False)

        # If a single sentence still exceeds max_unit_tokens, use token-based word fallback
        if sent_tok_len > max_unit_tokens:
            flush_sents()
            words = sent.split()
            word_buf: list[str] = []
            for word in words:
                candidate = " ".join(word_buf + [word])
                cand_tok_len = count_tokens(tokenizer, candidate, add_special_tokens=False)
                if cand_tok_len <= max_unit_tokens:
                    word_buf.append(word)
                else:
                    if word_buf:
                        flushed_text = " ".join(word_buf)
                        flushed_tok_len = count_tokens(
                            tokenizer, flushed_text, add_special_tokens=False
                        )
                        units.append(
                            TextUnit(
                                text=flushed_text,
                                page=page_num,
                                document_section=document_section,
                                heading=heading,
                                token_count=flushed_tok_len,
                            )
                        )
                    word_buf = [word]
            if word_buf:
                flushed_text = " ".join(word_buf)
                flushed_tok_len = count_tokens(
                    tokenizer, flushed_text, add_special_tokens=False
                )
                units.append(
                    TextUnit(
                        text=flushed_text,
                        page=page_num,
                        document_section=document_section,
                        heading=heading,
                        token_count=flushed_tok_len,
                    )
                )
            continue

        candidate_text = " ".join(current_sents + [sent]).strip()
        cand_tok_len = count_tokens(tokenizer, candidate_text, add_special_tokens=False)
        if cand_tok_len <= max_unit_tokens:
            current_sents.append(sent)
        else:
            flush_sents()
            current_sents = [sent]

    flush_sents()
    return units


def select_chunk_heading(chunk_units: list[TextUnit]) -> str | None:
    """Select the most specific heading present in the chunk's units."""
    for u in chunk_units:
        if u.heading and " > " in u.heading:
            return u.heading
    for u in chunk_units:
        if u.heading:
            return u.heading
    return None


def top_level_heading_of(heading: str | None) -> str | None:
    """Return the top-level chapter/part component of a heading."""
    if not heading:
        return None
    return heading.split(" > ", 1)[0]


def build_section_chunks(
    units: list[TextUnit],
    document_section: str,
    section_prefix: str,
    tokenizer: Any,
    model_max_seq_length: int,
    target_tokens: int = TARGET_CONTENT_TOKENS,
    max_content_tokens: int = MAX_CONTENT_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> list[dict[str, Any]]:
    """
    Assemble paragraph/sentence TextUnits into chunks targeting ~180 content tokens with ~30 tokens
    of overlap. Ensures every complete embedding input (including prefix and special tokens) fits
    within `model_max_seq_length` without truncation.
    """
    chunks: list[dict[str, Any]] = []
    current_units: list[TextUnit] = []

    def can_fit_units(candidate_units: list[TextUnit]) -> tuple[bool, int, int]:
        cand_text = "\n\n".join(u.text for u in candidate_units)
        cand_heading = select_chunk_heading(candidate_units)
        content_toks = count_tokens(tokenizer, cand_text, add_special_tokens=False)
        emb_input = format_embedding_input(DOCUMENT_TITLE, document_section, cand_heading, cand_text)
        emb_toks = count_tokens(tokenizer, emb_input, add_special_tokens=True)
        fits = (content_toks <= max_content_tokens) and (emb_toks <= model_max_seq_length)
        return fits, content_toks, emb_toks

    def finalize_chunk(chunk_units: list[TextUnit]) -> None:
        if not chunk_units:
            return
        chunk_text = "\n\n".join(u.text for u in chunk_units).strip()
        if not chunk_text:
            return
        chunk_heading = select_chunk_heading(chunk_units)
        pdf_pages = sorted({u.page for u in chunk_units})
        content_toks = count_tokens(tokenizer, chunk_text, add_special_tokens=False)
        emb_input = format_embedding_input(DOCUMENT_TITLE, document_section, chunk_heading, chunk_text)
        emb_toks = count_tokens(tokenizer, emb_input, add_special_tokens=True)

        if emb_toks > model_max_seq_length:
            raise RuntimeError(
                f"Chunk embedding input ({emb_toks} tokens) exceeds model max_seq_length "
                f"({model_max_seq_length}) on pages {pdf_pages}."
            )

        idx = len(chunks) + 1
        digest = hashlib.sha256(
            f"{document_section}|{pdf_pages}|{chunk_heading}|{chunk_text}".encode("utf-8")
        ).hexdigest()[:10]
        chunk_id = f"{section_prefix}_{idx:04d}_{digest}"

        chunks.append(
            {
                "chunk_id": chunk_id,
                "text": chunk_text,
                "source_file": SOURCE_FILE_REL,
                "document_title": DOCUMENT_TITLE,
                "document_section": document_section,
                "pdf_pages": pdf_pages,
                "heading": chunk_heading,
                "token_count": content_toks,
            }
        )

    for unit in units:
        if not current_units:
            current_units = [unit]
            continue

        # Close chunk cleanly without overlap if a new top-level CHAPTER / PART / Title Page boundary is crossed
        prev_top = top_level_heading_of(current_units[-1].heading)
        curr_top = top_level_heading_of(unit.heading)
        if prev_top and curr_top and prev_top != curr_top:
            finalize_chunk(current_units)
            current_units = [unit]
            continue

        candidate = current_units + [unit]
        fits, cand_content_toks, _ = can_fit_units(candidate)
        _, curr_content_toks, _ = can_fit_units(current_units)

        # If adding `unit` still fits within limits and we haven't reached target_tokens yet
        # (or adding it stays close to target_tokens and within max_content_tokens):
        if fits and (curr_content_toks < target_tokens and cand_content_toks <= max_content_tokens):
            current_units.append(unit)
        else:
            finalize_chunk(current_units)

            # Build overlap tail (~30 tokens) from the end of `current_units` within the same top-level section
            overlap_units: list[TextUnit] = []
            overlap_tok_sum = 0
            for prev_u in reversed(current_units):
                if top_level_heading_of(prev_u.heading) != curr_top:
                    break
                if overlap_tok_sum + prev_u.token_count <= overlap_tokens + 25:
                    test_units = [prev_u] + overlap_units + [unit]
                    test_fits, _, _ = can_fit_units(test_units)
                    if test_fits:
                        overlap_units.insert(0, prev_u)
                        overlap_tok_sum += prev_u.token_count
                    else:
                        break
                else:
                    break

            current_units = overlap_units + [unit]

    if current_units:
        finalize_chunk(current_units)

    return chunks


def run_preprocessing_and_chunking(
    pages_json_path: Path = PAGES_JSON_PATH,
    chunks_json_path: Path = CHUNKS_JSON_PATH,
    ingestion_report_path: Path = INGESTION_REPORT_PATH,
    model: SentenceTransformer | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Read pages.json, conservatively clean text, exclude TOC and blank/header-only pages,
    chunk with the model's tokenizer, and save chunks.json and ingestion_report.json.
    """
    if not pages_json_path.exists():
        raise FileNotFoundError(
            f"Extracted pages file not found at: {pages_json_path}. "
            "Run `python backend/extract_pdf.py` first."
        )

    with pages_json_path.open("r", encoding="utf-8") as f:
        pages_data = json.load(f)

    if not isinstance(pages_data, list) or not pages_data:
        raise ValueError(f"Invalid or empty pages.json at: {pages_json_path}")

    appendix_start_page = verify_section_boundary(pages_data)

    if model is None:
        print(f"Loading tokenizer & embedding model '{MODEL_ID}' on CPU...")
        model = SentenceTransformer(MODEL_ID, device="cpu")

    tokenizer = model.tokenizer
    model_max_seq_length = int(model.max_seq_length)
    embedding_dim = int(model.get_embedding_dimension())

    main_section_pages: list[tuple[int, str]] = []
    rmp_section_pages: list[tuple[int, str]] = []
    skipped_pages: list[dict[str, Any]] = []
    partial_toc_cleaned_pages: list[int] = []

    for entry in pages_data:
        page_num = int(entry["page"])
        raw_text = str(entry["text"])
        section_name = (
            SECTION_MAIN_HANDBOOK if page_num < appendix_start_page else SECTION_APPENDED_RMP
        )

        cleaned_text, skip_reason, details = clean_page_conservatively(page_num, raw_text)
        if skip_reason is not None:
            skipped_pages.append(
                {
                    "page": page_num,
                    "document_section": section_name,
                    "reason": skip_reason,
                    "removed_headers_footers": details["removed_headers_footers"],
                }
            )
            continue

        if details["removed_toc_lines_count"] > 0:
            partial_toc_cleaned_pages.append(page_num)

        if page_num < appendix_start_page:
            main_section_pages.append((page_num, cleaned_text))
        else:
            rmp_section_pages.append((page_num, cleaned_text))

    # Segment each document section independently into paragraph blocks and split oversized paragraphs
    # Safe per-unit token limit leaves room for heading prefix + [CLS]/[SEP] tokens
    safe_unit_max_tokens = min(MAX_CONTENT_TOKENS, model_max_seq_length - 45)

    main_paragraphs = segment_section_into_paragraphs(main_section_pages, SECTION_MAIN_HANDBOOK)
    main_units: list[TextUnit] = []
    for p_text, p_page, p_heading in main_paragraphs:
        main_units.extend(
            split_oversized_paragraph(
                p_text, p_page, SECTION_MAIN_HANDBOOK, p_heading, tokenizer, safe_unit_max_tokens
            )
        )

    rmp_paragraphs = segment_section_into_paragraphs(rmp_section_pages, SECTION_APPENDED_RMP)
    rmp_units: list[TextUnit] = []
    for p_text, p_page, p_heading in rmp_paragraphs:
        rmp_units.extend(
            split_oversized_paragraph(
                p_text, p_page, SECTION_APPENDED_RMP, p_heading, tokenizer, safe_unit_max_tokens
            )
        )

    main_chunks = build_section_chunks(
        main_units,
        document_section=SECTION_MAIN_HANDBOOK,
        section_prefix="fh_main",
        tokenizer=tokenizer,
        model_max_seq_length=model_max_seq_length,
    )
    rmp_chunks = build_section_chunks(
        rmp_units,
        document_section=SECTION_APPENDED_RMP,
        section_prefix="fh_rmp",
        tokenizer=tokenizer,
        model_max_seq_length=model_max_seq_length,
    )

    all_chunks = main_chunks + rmp_chunks

    # Compute token stats across all chunks
    content_token_counts = [c["token_count"] for c in all_chunks]
    embedding_token_counts = [
        count_tokens(
            tokenizer,
            format_embedding_input(
                c["document_title"], c["document_section"], c["heading"], c["text"]
            ),
            add_special_tokens=True,
        )
        for c in all_chunks
    ]

    # Save chunks.json
    chunks_json_path.parent.mkdir(parents=True, exist_ok=True)
    with chunks_json_path.open("w", encoding="utf-8") as f_chunks:
        json.dump(all_chunks, f_chunks, indent=2, ensure_ascii=False)

    # Save ingestion_report.json
    report = {
        "project": "Tulane University Faculty Onboarding Assistant",
        "source_file": SOURCE_FILE_REL,
        "document_title": DOCUMENT_TITLE,
        "dataset_note": (
            "The source PDF is the Tulane University Faculty Handbook (pages 1-191) with the "
            "appended Research Misconduct Policy (pages 192-238). Faculty policies must never "
            "be presented as general staff policies."
        ),
        "total_pdf_pages": len(pages_data),
        "indexed_pages_count": len(main_section_pages) + len(rmp_section_pages),
        "indexed_pages_by_section": {
            SECTION_MAIN_HANDBOOK: len(main_section_pages),
            SECTION_APPENDED_RMP: len(rmp_section_pages),
        },
        "skipped_pages_count": len(skipped_pages),
        "skipped_pages": skipped_pages,
        "partial_toc_lines_removed_on_pages": partial_toc_cleaned_pages,
        "verified_appendix_boundary_page": appendix_start_page,
        "model_id": MODEL_ID,
        "model_max_seq_length": model_max_seq_length,
        "embedding_dimension": embedding_dim,
        "chunking_settings": {
            "target_content_tokens": TARGET_CONTENT_TOKENS,
            "max_content_tokens": MAX_CONTENT_TOKENS,
            "overlap_tokens": OVERLAP_TOKENS,
        },
        "chunks_summary": {
            "total_chunks": len(all_chunks),
            "main_handbook_chunks": len(main_chunks),
            "research_misconduct_policy_chunks": len(rmp_chunks),
            "min_content_tokens": min(content_token_counts) if content_token_counts else 0,
            "avg_content_tokens": round(
                sum(content_token_counts) / len(content_token_counts), 1
            )
            if content_token_counts
            else 0,
            "max_content_tokens": max(content_token_counts) if content_token_counts else 0,
            "max_embedding_input_tokens_with_special": max(embedding_token_counts)
            if embedding_token_counts
            else 0,
        },
    }

    with ingestion_report_path.open("w", encoding="utf-8") as f_rep:
        json.dump(report, f_rep, indent=2, ensure_ascii=False)

    return all_chunks, report


def main() -> int:
    try:
        chunks, report = run_preprocessing_and_chunking()
        print("Preprocessing & Chunking Summary")
        print("-" * 45)
        print(f"Total PDF pages: {report['total_pdf_pages']}")
        print(f"Indexed pages: {report['indexed_pages_count']}")
        print(f"Skipped pages ({report['skipped_pages_count']}): {[s['page'] for s in report['skipped_pages']]}")
        print(f"Verified appendix start page: {report['verified_appendix_boundary_page']}")
        print(f"Total chunks created: {len(chunks)}")
        print(f"  - {SECTION_MAIN_HANDBOOK}: {report['chunks_summary']['main_handbook_chunks']}")
        print(f"  - {SECTION_APPENDED_RMP}: {report['chunks_summary']['research_misconduct_policy_chunks']}")
        print(f"Model max sequence length: {report['model_max_seq_length']}")
        print(
            f"Max content tokens: {report['chunks_summary']['max_content_tokens']} | "
            f"Max full embedding input tokens (incl. prefix & [CLS]/[SEP]): "
            f"{report['chunks_summary']['max_embedding_input_tokens_with_special']}"
        )
        print(f"Saved chunks to: {CHUNKS_JSON_PATH}")
        print(f"Saved ingestion report to: {INGESTION_REPORT_PATH}")
        return 0
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

