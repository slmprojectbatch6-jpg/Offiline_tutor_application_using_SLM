"""
Phase 1.5: LLM-Based Text Enrichment (Optional)
-----------------------------------------------
Reads raw `extracted_text/{subject}.txt`, splits it into sections, and uses the 
local Phi-3 model to rewrite and enrich the text for better definitions and clarity.
NOTE: This takes a significant amount of time depending on hardware speed.
"""

import sys
import json
from pathlib import Path
from loguru import logger
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import settings

ENRICH_PROMPT = """You are an expert textbook editor and educator.
Please rewrite the following textbook section to be more structured, detailed, and clear. 
- Expand on any implied concepts.
- Add short, clear explanations for difficult terms to help students understand better. 
- Keep ALL original factual information completely intact.
- Use bullet points for lists and structured paragraphs.
- Do NOT add a conversational intro (e.g. "Here is the enriched text:"). Just output the raw enriched educational text.

TEXT TO ENRICH:
{text_block}
"""

def split_into_large_blocks(text: str, max_words=600):
    """Split text into blocks of roughly `max_words`."""
    paragraphs = text.split("\n\n")
    blocks = []
    current_block = []
    current_words = 0

    for para in paragraphs:
        para_words = len(para.split())
        if current_words + para_words > max_words and current_block:
            blocks.append("\n\n".join(current_block))
            current_block = [para]
            current_words = para_words
        else:
            current_block.append(para)
            current_words += para_words

    if current_block:
        blocks.append("\n\n".join(current_block))

    return blocks

def rewrite_block(text_block: str, model_name: str) -> str:
    """Send text block to Ollama for enrichment."""
    prompt = ENRICH_PROMPT.format(text_block=text_block)
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 1024,
                }
            },
            timeout=120
        )
        response.raise_for_status()
        res_json = response.json()
        return res_json.get("response", text_block).strip()
    except Exception as e:
        logger.error(f"Ollama API Error during enrichment: {e}")
        return text_block # Fallback to original text if error

def enrich_subject(subject: str):
    """Enrich the text for a specific subject incrementally."""
    input_file = settings.EXTRACTED_TEXT_DIR / f"{subject}.txt"
    output_file = settings.EXTRACTED_TEXT_DIR / f"{subject}_enriched.txt"
    progress_file = settings.EXTRACTED_TEXT_DIR / f"{subject}_enrich_progress.json"

    if not input_file.exists():
        logger.warning(f"No extracted text found for {subject}")
        return

    logger.info(f"Starting enrichment for: {subject}...")
    raw_text = input_file.read_text(encoding="utf-8")
    blocks = split_into_large_blocks(raw_text)

    logger.info(f"Total blocks to process: {len(blocks)}")

    # Load progress
    enriched_blocks = []
    start_idx = 0
    if progress_file.exists():
        try:
            with open(progress_file, "r", encoding="utf-8") as f:
                progress = json.load(f)
                enriched_blocks = progress.get("blocks", [])
                start_idx = len(enriched_blocks)
                logger.info(f"Resuming from block {start_idx}/{len(blocks)}")
        except Exception:
            pass

    for i in range(start_idx, len(blocks)):
        logger.info(f"Enriching block {i+1}/{len(blocks)} ({len(blocks[i].split())} words)...")
        enriched_text = rewrite_block(blocks[i], settings.OLLAMA_MODEL)
        enriched_blocks.append(enriched_text)

        # Save incremental progress
        with open(progress_file, "w", encoding="utf-8") as f:
            json.dump({"blocks": enriched_blocks}, f, ensure_ascii=False)

    # All blocks finished, save final enriched text
    full_enriched_text = "\n\n".join(enriched_blocks)
    output_file.write_text(full_enriched_text, encoding="utf-8")
    
    # Clean up progress file
    if progress_file.exists():
        progress_file.unlink()

    logger.success(f"Successfully enriched {subject}. Output saved to {output_file.name}")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("PHASE 1.5: TEXT ENRICHMENT VIA LLM")
    logger.info("=" * 60)
    
    for subject in settings.SUBJECTS:
        enrich_subject(subject)
        
    logger.success("✅ Enrichment complete! You can now run Phase 2 (preprocess_chunks.py) on the enriched text.")
