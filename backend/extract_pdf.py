"""
Stage 1 PDF text extractor for the Tulane University Faculty Onboarding Assistant.

Dataset Note:
Although the source file is named `data/raw/tulane_staff_handbook.pdf`, its actual content is the
Tulane University Faculty Handbook (pages 1-191) with the appended Tulane University Policies and
Procedures for Responding to Allegations of Research Misconduct (pages 192-238). Faculty policies
from this document must never be presented as general staff policies.
"""
import json
import re
import sys
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError, PdfStreamError


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PDF_PATH = PROJECT_ROOT / "data" / "raw" / "tulane_staff_handbook.pdf"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PAGES_JSON_PATH = PROCESSED_DIR / "pages.json"
HANDBOOK_TXT_PATH = PROCESSED_DIR / "handbook.txt"

LOW_CHAR_THRESHOLD = 50


def normalize_text(raw_text: str) -> str:
    """Normalize unnecessary whitespace while preserving paragraph breaks and all content."""
    if not raw_text:
        return ""

    # Normalize line endings and replace non-breaking spaces / null bytes with standard spaces
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\x00", "")

    # Collapse multiple horizontal spaces/tabs within each line and strip line ends
    lines = [re.sub(r"[ \t\f\v]+", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Collapse 3 or more consecutive newlines into a single blank line (paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def count_non_whitespace(text: str) -> int:
    """Count non-whitespace characters in a string."""
    return len(re.sub(r"\s+", "", text))


def extract_pdf(
    pdf_path: Path = DEFAULT_PDF_PATH,
    pages_json_path: Path = PAGES_JSON_PATH,
    handbook_txt_path: Path = HANDBOOK_TXT_PATH,
) -> list[dict]:
    """Extract text page by page from the PDF and save structured outputs."""
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"Source PDF not found at: {pdf_path}\n"
            f"Please ensure the file exists at 'data/raw/tulane_staff_handbook.pdf' relative to the project root."
        )

    if not pdf_path.is_file():
        raise ValueError(f"Expected a file but found a directory or special path at: {pdf_path}")

    if pdf_path.stat().st_size == 0:
        raise ValueError(f"The PDF file is empty (0 bytes): {pdf_path}")

    try:
        reader = PdfReader(str(pdf_path))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:
                raise ValueError(f"PDF is encrypted and could not be decrypted: {pdf_path}") from exc
        total_pages = len(reader.pages)
    except (PdfReadError, PdfStreamError, Exception) as exc:
        raise ValueError(f"Invalid or corrupted PDF file at '{pdf_path}': {exc}") from exc

    if total_pages == 0:
        raise ValueError(f"PDF contains 0 pages: {pdf_path}")

    pages_data: list[dict] = []
    sparse_pages: list[int] = []
    total_char_count = 0
    total_non_ws_count = 0

    for idx, page in enumerate(reader.pages, start=1):
        try:
            raw_page_text = page.extract_text() or ""
        except Exception as exc:
            raise ValueError(f"Failed to extract text from page {idx} of '{pdf_path}': {exc}") from exc

        cleaned_text = normalize_text(raw_page_text)
        non_ws_chars = count_non_whitespace(cleaned_text)

        if non_ws_chars < LOW_CHAR_THRESHOLD:
            sparse_pages.append(idx)

        total_char_count += len(cleaned_text)
        total_non_ws_count += non_ws_chars

        pages_data.append({
            "page": idx,
            "text": cleaned_text,
        })

    # Ensure output directory exists
    pages_json_path.parent.mkdir(parents=True, exist_ok=True)
    handbook_txt_path.parent.mkdir(parents=True, exist_ok=True)

    # Save pages.json
    with pages_json_path.open("w", encoding="utf-8") as f_json:
        json.dump(pages_data, f_json, indent=2, ensure_ascii=False)

    # Save handbook.txt with clear page separators
    with handbook_txt_path.open("w", encoding="utf-8") as f_txt:
        for item in pages_data:
            page_num = item["page"]
            page_text = item["text"]
            separator = f"{'=' * 80}\nPAGE {page_num} OF {total_pages}\n{'=' * 80}"
            f_txt.write(f"{separator}\n\n{page_text}\n\n")

    # Print extraction summary
    print("PDF Extraction Summary")
    print("-" * 40)
    print(f"Source PDF: {pdf_path}")
    print(f"Total page count: {total_pages}")
    print(f"Total extracted character count: {total_char_count} ({total_non_ws_count} non-whitespace)")
    print(
        f"Pages with fewer than {LOW_CHAR_THRESHOLD} non-whitespace characters "
        f"({len(sparse_pages)} of {total_pages}): {sparse_pages if sparse_pages else 'None'}"
    )
    print(f"Saved JSON output to: {pages_json_path}")
    print(f"Saved text output to: {handbook_txt_path}")

    # Check if extraction is empty or mostly empty
    sparse_ratio = len(sparse_pages) / total_pages
    avg_non_ws_per_page = total_non_ws_count / total_pages

    if total_non_ws_count == 0 or sparse_ratio >= 0.5 or avg_non_ws_per_page < LOW_CHAR_THRESHOLD:
        raise RuntimeError(
            "WARNING / ERROR: Extracted text is empty or mostly empty "
            f"({len(sparse_pages)}/{total_pages} pages have < {LOW_CHAR_THRESHOLD} non-whitespace characters; "
            f"average {avg_non_ws_per_page:.1f} non-whitespace chars/page). "
            "This PDF appears to be scanned or image-based; OCR may be required."
        )

    return pages_data


def main() -> int:
    try:
        extract_pdf()
        return 0
    except (FileNotFoundError, ValueError, RuntimeError) as err:
        print(f"\nERROR: {err}", file=sys.stderr)
        return 1
    except Exception as err:
        print(f"\nUNEXPECTED ERROR: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

