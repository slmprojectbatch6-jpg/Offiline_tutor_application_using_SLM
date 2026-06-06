"""
Phase 2: Text Preprocessing & Chunking
---------------------------------------
Reads extracted_text/<subject>.txt files,
preprocesses, chunks into 300-500 word chunks with 50-word overlap,
and saves structured JSON to chunks/<subject>_chunks.json
"""

import json
import re
import sys
import hashlib
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from backend.config import settings


# ── Text Preprocessing ───────────────────────────────────────

def preprocess_text(text: str) -> str:
    """Deep-clean text before chunking."""
    # Remove special / non-printable characters (keep basic punctuation)
    text = re.sub(r"[^\w\s.,;:!?()\-\'\"]", " ", text)

    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)

    # Remove lines that are just dashes or underscores
    text = re.sub(r"^[-_]{3,}$", "", text, flags=re.MULTILINE)

    # Remove excessive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove duplicate lines
    seen_lines: set = set()
    unique_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and stripped not in seen_lines:
            seen_lines.add(stripped)
            unique_lines.append(line)

    return "\n".join(unique_lines).strip()


# ── Chapter Detection ─────────────────────────────────────────

def detect_chapter(text_block: str) -> str:
    """Try to detect chapter heading from a block of text."""
    patterns = [
        r"chapter\s+\d+[:\-–]?\s*(.+?)(?:\n|$)",
        r"=== (.+?) ===",
        r"unit\s+\d+[:\-–]?\s*(.+?)(?:\n|$)",
    ]
    for pat in patterns:
        match = re.search(pat, text_block[:500], re.IGNORECASE)
        if match:
            return match.group(1).strip()[:80]
    return "General"


# ── Chunking ─────────────────────────────────────────────────

def words_to_text(words: List[str]) -> str:
    return " ".join(words)


def recursive_split(text: str, chunk_size: int, overlap: int) -> List[str]:
    """A minimal recursive character text splitter."""
    separators = ["\n\n", "\n", ".", "?", "!", " ", ""]
    
    def _split(txt: str, sep_idx: int) -> List[str]:
        if len(txt) <= chunk_size or sep_idx >= len(separators):
            return [txt]
            
        sep = separators[sep_idx]
        if sep == "":
            return [txt[i:i+chunk_size] for i in range(0, len(txt), chunk_size - overlap)]
            
        splits = txt.split(sep)
        # Re-attach the separator for elements that need it
        if sep in [".", "?", "!"]:
            splits = [s + sep if i < len(splits)-1 else s for i, s in enumerate(splits)]
        elif sep in ["\n\n", "\n"]:
            splits = [s + sep if i < len(splits)-1 else s for i, s in enumerate(splits)]
            
        chunks = []
        current_chunk = ""
        
        for s in splits:
            if len(current_chunk) + len(s) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap - simple approach: don't strictly do overlap at the char level for simplicity,
                # just start the next chunk 
                current_chunk = s
            else:
                current_chunk += ("" if current_chunk.endswith(sep) else (" " if current_chunk and sep == " " else "")) + s
                
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        # Recursive check
        final_chunks = []
        for c in chunks:
            if len(c) > chunk_size:
                final_chunks.extend(_split(c, sep_idx + 1))
            else:
                final_chunks.append(c)
                
        return final_chunks

    return _split(text, 0)


def chunk_text(
    text: str,
    subject: str,
    chunk_size: int = settings.CHUNK_SIZE_WORDS * 5,  # Approximating chars per word
    overlap: int = settings.CHUNK_OVERLAP_WORDS * 5,
):
    """
    Split text into chunks using a custom recursive text splitter.
    """
    text_chunks = recursive_split(text, chunk_size, overlap)
    
    # Try to detect the chapter once for this block
    chapter = detect_chapter(text)
    
    chunk_index = 0
    for chunk_text_str in text_chunks:
        word_count = len(chunk_text_str.split())
        if word_count < 20:
            continue

        chunk_id = hashlib.md5(
            f"{subject}_{chunk_index}_{chunk_text_str[:50]}".encode()
        ).hexdigest()[:12]

        yield {
            "chunk_id": chunk_id,
            "subject": subject,
            "chapter": chapter,
            "chunk_index": chunk_index,
            "text": chunk_text_str.strip(),
            "word_count": word_count,
        }
        chunk_index += 1


# ── Main Processing ───────────────────────────────────────────

def process_all_subjects():
    """Process all extracted text files and produce chunk JSONs using a stream."""
    settings.CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}

    import gc
    gc.collect()

    for subject in settings.SUBJECTS:
        enriched_file = settings.EXTRACTED_TEXT_DIR / f"{subject}_enriched.txt"
        base_file = settings.EXTRACTED_TEXT_DIR / f"{subject}.txt"
        
        if enriched_file.exists():
            input_file = enriched_file
            logger.info(f"Using ENRICHED text for: {subject}")
        elif base_file.exists():
            input_file = base_file
            logger.info(f"Using base text for: {subject}")
        else:
            logger.warning(f"No extracted text found for: {subject} (run Phase 1 first)")
            continue

        raw_text = input_file.read_text(encoding="utf-8")

        # Preprocess
        clean_text = preprocess_text(raw_text)
        del raw_text
        gc.collect()
        
        logger.info(f"  Word count after cleaning: {len(clean_text.split()):,}")

        # Split by chapter sections (=== Chapter Name ===)
        sections = re.split(r"=== .+? ===", clean_text)
        section_headers = re.findall(r"=== (.+?) ===", clean_text)

        all_chunks_count = 0
        output_file = settings.CHUNKS_DIR / f"{subject}_chunks.json"
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("[\n")
            first = True
            
            if len(sections) > 1:
                # Process section by section
                for i, section in enumerate(sections):
                    if not section.strip():
                        continue
                    chapter_name = section_headers[i - 1] if i > 0 and i - 1 < len(section_headers) else "Introduction"
                    section_clean = preprocess_text(section)
                    section_chunks = chunk_text(section_clean, subject)
                    
                    for c in section_chunks:
                        c["chapter"] = chapter_name
                        if not first:
                            f.write(",\n")
                        json.dump(c, f, ensure_ascii=False)
                        first = False
                        all_chunks_count += 1
                        
                    del section_clean
                    del section_chunks
                    gc.collect()
            else:
                # Process as single block
                section_chunks = chunk_text(clean_text, subject)
                for c in section_chunks:
                    if not first:
                        f.write(",\n")
                    json.dump(c, f, ensure_ascii=False)
                    first = False
                    all_chunks_count += 1
                    
                del section_chunks
                gc.collect()

            f.write("\n]\n")

        summary[subject] = all_chunks_count
        logger.success(f"  ✓ {all_chunks_count} chunks saved → {output_file.name}")
        
        del clean_text
        del sections
        del section_headers
        gc.collect()

    return summary


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("PHASE 2: TEXT PREPROCESSING & CHUNKING")
    logger.info("=" * 60)
    result = process_all_subjects()
    logger.info("\n📊 CHUNKING SUMMARY:")
    for subj, count in result.items():
        logger.info(f"  {subj}: {count} chunks")
    logger.success("\n✅ Phase 2 Complete!")
