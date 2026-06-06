"""
Phase 5: Phi-3 Small Language Model Integration (via Ollama)
-------------------------------------------------------------
Offline LLM inference using a local Ollama server.
Falls back to a mock response if Ollama is unreachable.
"""

import json
import sys
from pathlib import Path
from typing import Generator, Optional

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from loguru import logger
from backend.config import settings


class Phi3Model:
    """Wrapper for Phi-3 using local Ollama API."""

    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL.rstrip('/')
        self.model_name = settings.OLLAMA_MODEL
        self._loaded = False
        self._mock_mode = False

    def load(self):
        """Check if Ollama is running and has the model."""
        if self._loaded:
            return

        try:
            logger.info(f"Checking Ollama server at {self.base_url} for '{self.model_name}'")
            
            # Use a short timeout so app startup isn't delayed if it's down
            res = httpx.get(f"{self.base_url}/api/tags", timeout=3.0)
            res.raise_for_status()
            
            # Check if model exists
            tags = res.json().get("models", [])
            has_model = any(m.get("name") == self.model_name or m.get("name", "").startswith(f"{self.model_name}:") for m in tags)
            
            if has_model:
                self._loaded = True
                self._mock_mode = False
                logger.success(f"Ollama connected successfully. Using model '{self.model_name}'")
            else:
                logger.warning(f"Ollama is running, but model '{self.model_name}' is missing!\n"
                               f"Run `ollama run {self.model_name}` to download it.")
                self._mock_mode = True
                self._loaded = True
                
        except (httpx.RequestError, httpx.HTTPError) as e:
            logger.warning(
                f"Count not connect to Ollama at {self.base_url}.\n"
                "Running in MOCK mode. To enable full AI:\n"
                "  1. Install & double-click Ollama\n"
                f"  2. Open terminal and run: ollama run {self.model_name}\n"
                f"Error: {str(e)}"
            )
            self._mock_mode = True
            self._loaded = True

    def generate(self, prompt: str, stream: bool = False) -> str:
        """Generate a response for the given prompt from Ollama."""
        if not self._loaded:
            self.load()

        if self._mock_mode:
            return self._mock_response(prompt)

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": settings.PHI3_TEMPERATURE,
                "num_ctx": settings.PHI3_CONTEXT_LENGTH,
                "num_predict": settings.PHI3_MAX_TOKENS,
                "repeat_penalty": 1.2,
                "stop": ["Question:", "Context:", "\n\n\n"]
            }
        }

        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(f"{self.base_url}/api/generate", json=payload)
                response.raise_for_status()
                return response.json().get("response", "").strip()
        except Exception as e:
            logger.error(f"Generate error: {e}")
            return f"❌ Failed to reach Ollama: {str(e)}"

    def stream(self, prompt: str) -> Generator[str, None, None]:
        """Stream response tokens from Ollama."""
        if not self._loaded:
            self.load()

        if self._mock_mode:
            # Simulate streaming for mock mode
            mock_text = self._mock_response(prompt)
            words = mock_text.split()
            import time
            for i, word in enumerate(words):
                yield word + (" " if i < len(words) - 1 else "")
                time.sleep(0.05)
            return

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": settings.PHI3_TEMPERATURE,
                "num_ctx": settings.PHI3_CONTEXT_LENGTH,
                "num_predict": settings.PHI3_MAX_TOKENS,
                "repeat_penalty": 1.2,
                "stop": ["Question:", "Context:", "\n\n\n"]
            }
        }

        try:
            with httpx.Client(timeout=60.0) as client:
                with client.stream("POST", f"{self.base_url}/api/generate", json=payload) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if line:
                            data = json.loads(line)
                            yield data.get("response", "")
        except Exception as e:
            logger.error(f"Stream generation error: {e}")
            yield f"❌ Stream failed: {str(e)}"

    def _mock_response(self, prompt: str) -> str:
        """Fallback mock response when Ollama is not available."""
        # Extract the question from the prompt
        q_start = prompt.find("Question:") + len("Question:")
        a_start = prompt.find("Answer:")
        question = prompt[q_start:a_start].strip() if q_start > 0 else "your question"

        return (
            f"⚠️ **Mock Mode Active** — Ollama server missing or model not found.\n\n"
            f"To get real AI answers:\n"
            f"1. Make sure the Ollama app is running\n"
            f"2. Run exactly: `ollama run {self.model_name}` in your command prompt.\n"
            f"3. Your question was: *{question}*\n\n"
            f"The RAG retrieval is working — relevant NCERT chunks have been found."
        )


# ── Singleton ─────────────────────────────────────────────────
_phi3_instance: Optional[Phi3Model] = None


def get_phi3_model() -> Phi3Model:
    global _phi3_instance
    if _phi3_instance is None:
        _phi3_instance = Phi3Model()
        _phi3_instance.load()
    return _phi3_instance
