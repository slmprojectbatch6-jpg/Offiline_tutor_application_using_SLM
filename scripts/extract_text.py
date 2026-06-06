"""
Phase 1: NCERT PDF Text Extraction
-----------------------------------
Reads PDFs from data/<subject>/ folders,
cleans text, and writes to extracted_text/<subject>.txt
"""

import re
import sys
from pathlib import Path

# Allow running as standalone script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

try:
    import fitz  # PyMuPDF
    PDF_BACKEND = "pymupdf"
except ImportError:
    try:
        import pdfplumber
        PDF_BACKEND = "pdfplumber"
    except ImportError:
        raise ImportError("Install PyMuPDF or pdfplumber: pip install PyMuPDF")

from backend.config import settings


# ── Helpers ─────────────────────────────────────────────────

def _clean_text(raw: str) -> str:
    """Remove headers, footers, page numbers, and noise, but preserve paragraph structure."""
    lines = raw.splitlines()
    cleaned = []

    for line in lines:
        line = line.strip()

        # Skip page numbers (lone digits or "Page X of Y")
        if re.fullmatch(r"\d+", line):
            continue
        if re.search(r"page\s+\d+\s*(of\s*\d+)?", line, re.IGNORECASE):
            continue

        # Skip very short lines (likely headers/footers)
        if len(line.split()) < 3 and len(line) < 30:
            continue

        # Skip common NCERT boilerplate
        skip_patterns = [
            r"^NCERT$", r"^www\.ncert\.nic\.in$",
            r"^Social Science$", r"^Class\s+\d+$",
            r"^Rationalised\s+", r"^not\s+to\s+be\s+republished",
            r"^©\s*NCERT", r"^Chapter\s+\d+$",
        ]
        if any(re.search(p, line, re.IGNORECASE) for p in skip_patterns):
            continue

        cleaned.append(line)

    # Join and normalize whitespace but preserve newlines
    # Instead of completely destroying paragraphs, we join lines that seem to belong to the same paragraph
    paragraphs = []
    current_para = []
    
    for line in cleaned:
        if not line:
            if current_para:
                paragraphs.append(" ".join(current_para))
                current_para = []
        else:
            current_para.append(line)
            
    if current_para:
         paragraphs.append(" ".join(current_para))

    text = "\n\n".join(paragraphs)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)  # Non-ASCII → space
    text = re.sub(r"\s+([.,;:])", r"\1", text)   # Fix punctuation spacing
    return text.strip()



def extract_with_pymupdf(pdf_path: Path) -> str:
    """Extract text using PyMuPDF."""
    doc = fitz.open(str(pdf_path))
    pages_text = []
    
    is_split_page = pdf_path.parent.name.lower() == "gk"
    
    for page in doc:
        if is_split_page:
            rect = page.rect
            left_rect = fitz.Rect(rect.x0, rect.y0, rect.x0 + rect.width / 2, rect.y1)
            right_rect = fitz.Rect(rect.x0 + rect.width / 2, rect.y0, rect.x1, rect.y1)
            left_text = page.get_text("text", clip=left_rect)
            right_text = page.get_text("text", clip=right_rect)
            pages_text.append(left_text)
            pages_text.append(right_text)
        else:
            pages_text.append(page.get_text("text"))
    doc.close()
    return "\n".join(pages_text)


def extract_with_pdfplumber(pdf_path: Path) -> str:
    """Extract text using pdfplumber."""
    import pdfplumber
    pages_text = []
    is_split_page = pdf_path.parent.name.lower() == "gk"
    
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            if is_split_page:
                width, height = page.width, page.height
                left_crop = page.crop((0, 0, width / 2, height))
                right_crop = page.crop((width / 2, 0, width, height))
                left_t = left_crop.extract_text()
                right_t = right_crop.extract_text()
                if left_t: pages_text.append(left_t)
                if right_t: pages_text.append(right_t)
            else:
                t = page.extract_text()
                if t:
                    pages_text.append(t)
    return "\n".join(pages_text)


def extract_pdf(pdf_path: Path) -> str:
    """Extract and clean text from a single PDF."""
    if PDF_BACKEND == "pymupdf":
        raw = extract_with_pymupdf(pdf_path)
    else:
        raw = extract_with_pdfplumber(pdf_path)
    return _clean_text(raw)


# ── Main Extraction ──────────────────────────────────────────

def extract_all_subjects():
    """Process all subjects in data/ directory."""
    settings.EXTRACTED_TEXT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}

    for subject in settings.SUBJECTS:
        subject_dir = settings.DATA_DIR / subject
        if not subject_dir.exists():
            logger.warning(f"Subject folder not found: {subject_dir}")
            continue

        pdf_files = list(subject_dir.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDFs found in {subject_dir}")
            continue

        logger.info(f"Processing {len(pdf_files)} PDF(s) for subject: {subject}")
        combined_text = []

        for pdf_path in sorted(pdf_files):
            logger.info(f"  → Extracting: {pdf_path.name}")
            try:
                text = extract_pdf(pdf_path)
                combined_text.append(f"\n\n=== {pdf_path.stem} ===\n\n{text}")
                logger.success(f"    ✓ {len(text.split())} words extracted")
            except Exception as e:
                logger.error(f"    ✗ Failed: {e}")

        if combined_text:
            output_path = settings.EXTRACTED_TEXT_DIR / f"{subject}.txt"
            full_text = "\n".join(combined_text)
            output_path.write_text(full_text, encoding="utf-8")
            word_count = len(full_text.split())
            summary[subject] = {"files": len(pdf_files), "words": word_count}
            logger.success(f"Saved {subject}.txt → {word_count} words total")

    return summary


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("PHASE 1: NCERT PDF TEXT EXTRACTION")
    logger.info("=" * 60)
    result = extract_all_subjects()
    logger.info("\n📊 EXTRACTION SUMMARY:")
    for subj, stats in result.items():
        logger.info(f"  {subj}: {stats['files']} PDFs → {stats['words']:,} words")
    logger.success("\n✅ Phase 1 Complete!")
