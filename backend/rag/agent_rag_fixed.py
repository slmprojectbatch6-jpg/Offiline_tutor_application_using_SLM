"""
Agentic RAG Query Rewriting and Iterative Retrieval Layer - Fixed Version
Ensures strict separation between internal reasoning and external output.
"""

import re
import time
import requests
from typing import List, Dict, Any, Optional
import numpy as np

from backend.rag.agent_rag_fixes import AgentRAGFixesMixin


# ============================================================================
# Prompts
# ============================================================================

QUERY_GENERATION_PROMPT = """<|system|>
You are a search query generator for an Indian NCERT textbook database covering History, Polity, Geography, and Economics for Classes 6-10.

Your ONLY job is to generate search queries. Do NOT answer the question. Do NOT explain anything.

Rules:
- Generate exactly 4 short search queries (3-6 words each)
- Think about what WORDS the NCERT textbook would actually use
- For "after X" questions, generate queries about what came NEXT (not about X itself)
- For "first/last" questions, include specific names, years, or events
- For comparison questions, generate separate queries for each concept
- One query should be broad (the general topic)
- One query should be specific (names, dates, events)
- One query should cover the CONTEXT (what came before/after)
- One query should use ALTERNATIVE TERMS for the same concept
<|end|>
<|user|>
Generate 4 search queries for: {user_question}
<|end|>
<|assistant|>
QUERY1:"""

ANSWER_GENERATION_PROMPT = """<|system|>
You are Vidhya Mitra, an NCERT textbook tutor for Indian students (Classes 6-10).

Rules:
- Answer ONLY using the Context Passages provided below. Do NOT use any outside knowledge.
- Give a direct, clear answer. No reasoning process, no "THOUGHT:", no internal monologue.
- IGNORING ARTIFACTS: The passages will contain textbook artifacts like "Answer in 100-150 words", "Questions", or chapter headers. IGNORE them. Do NOT output textbook questions in your answer.
- Synthesize a clean, factual answer based ONLY on what the Student Question asks. Do not answer questions found inside the context text.
- If the context does NOT have enough information, say: "This topic is not covered in detail in your NCERT textbooks." and share whatever partial info you have based ONLY on the context.
- Keep answers concise and use simple language suitable for school students.
- Format: plain text paragraphs. Use bullet points ONLY for listing multiple items.
<|end|>
<|user|>
Context Passages:
---
{retrieved_chunks}
---

Student Question: {user_question}

Give a clean, factual answer. Ignore and strip away any textbook questions/headers found in the context.
<|end|>
<|assistant|>
Here is the factual answer based on the context:
"""

AGENT_THINK_PROMPT = """<|system|>
You are a retrieval-augmented tutor agent for NCERT textbooks.
You are evaluating if the retrieved context is sufficient to answer the student's question.

Rules:
- If the context contains the answer (directly or indirectly), output: ACTION: ANSWER
- If the context does NOT contain the answer and you need to search more, output: ACTION: SEARCH
- If ACTION: SEARCH, you MUST provide 3 new search queries that are DIFFERENT from previous queries.

Format your response EXACTLY like this:
If you need more info:
ACTION: SEARCH
QUERY1: <new query 1>
QUERY2: <new query 2>
QUERY3: <new query 3>

If you have enough info:
ACTION: ANSWER
<|end|>
<|user|>
Question: {user_question}
Previous Queries: {previous_queries}

Retrieved Context So Far:
---
{accumulated_context}
---

Do you have enough information to answer?
<|end|>
<|assistant|>
"""


# ============================================================================
# Main Agent Class
# ============================================================================

class AgentRAG(AgentRAGFixesMixin):
    """
    Three-stage agentic RAG pipeline:
    Stage 1: Query Rewriting (internal, never shown to user)
    Stage 2: FAISS Retrieval (internal, never shown to user)  
    Stage 3: Answer Generation (clean answer, shown to user)
    """
    def __init__(
        self,
        ollama_model: str,
        faiss_index: Any,
        embedder: Any,
        chunks: List[Dict[str, Any]],
        chunk_metadata: Optional[List[Dict[str, Any]]] = None,
        mode: str = "simple",
        debug: bool = False
    ):
        self.model_name = ollama_model
        self.faiss_index = faiss_index
        self.embedder = embedder
        self.chunks = chunks
        self.mode = mode
        self.debug = debug
        self.max_unique_chunks = 10

    def _log(self, msg: str):
        """Internal logging."""
        if self.debug:
            print(f"[AgentRAG Debug] {msg}")

    def _call_ollama(self, prompt: str, temperature: float = 0.3, stop_tokens: Optional[List[str]] = None) -> str:
        """Call Phi-3 via Ollama local API."""
        if stop_tokens is None:
            stop_tokens = ["<|end|>", "<|user|>", "QUERY5:"]
            
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "top_p": 0.9,
                        "num_predict": 512,
                        "repeat_penalty": 1.2,
                        "stop": stop_tokens
                    }
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except requests.exceptions.RequestException as e:
            self._log(f"Ollama API Error: {str(e)}")
            return f"Error connecting to local LLM: {str(e)}"
        except Exception as e:
            self._log(f"Error generating response: {str(e)}")
            return ""

        # THIS METHOD IS NOW IN AgentRAGFixesMixin

        # THIS METHOD IS NOW IN AgentRAGFixesMixin

    def _generate_search_queries(self, user_question: str) -> List[str]:
        """Stage 1: Generate rewritten search queries."""
        self._log("Stage 1: Generating search queries")
        prompt = QUERY_GENERATION_PROMPT.format(user_question=user_question)
        
        # We manually prefix QUERY1: in the prompt, so the model starts from the first query text
        raw_response = self._call_ollama(prompt, temperature=0.4, stop_tokens=["<|end|>", "<|user|>"])
        
        # We add QUERY1: back to make parsing uniform
        full_response = "QUERY1:" + raw_response 
        
        queries = []
        for line in full_response.split('\n'):
            line = line.strip()
            if line.startswith("QUERY") and ":" in line:
                q = line.split(":", 1)[1].strip()
                # Remove any quotes or trailing punctuation that might mess up FAISS
                q = re.sub(r'^["\']|["\']$', '', q) 
                if len(q) > 2:
                    queries.append(q)
        
        if not queries:
            # Fallback if parsing failed
            self._log("Model failed to output QUERY format. Splitting by newlines.")
            for line in raw_response.split('\n'):
                line = line.strip()
                if line and len(line) > 3 and not line.startswith("<|"):
                    # remove bullets if any
                    line = re.sub(r'^[-*•\d.]\s+', '', line)
                    queries.append(line)
        
        # Absolute fallback
        if not queries:
            self._log("Complete query generation failure. Using original question.")
            queries = [user_question]
            
        # Ensure we don't have too many
        queries = queries[:4]
        self._log(f"Generated Queries: {queries}")
        return queries

        # THIS METHOD IS NOW IN AgentRAGFixesMixin

    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Format chunks into a readable string context."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            subject = chunk.get("subject", "Unknown").capitalize()
            chapter = chunk.get("chapter", "Unknown")
            text = chunk.get("text", "")
            parts.append(f"[Source {i} | {subject} - {chapter}]\n{text}")
        
        # Truncate context if it gets too long (roughly 2500 tokens max)
        full_context = "\n\n".join(parts)
        # Assuming ~4 chars per token, limit to 8000 chars to be safe for phi3
        max_chars = 8000 
        if len(full_context) > max_chars:
             self._log("Notice: Truncating context to fit LLM window.")
             full_context = full_context[:max_chars] + "\n...[Context Truncated]..."
             
        return full_context

    def _generate_answer(self, user_question: str, chunks: List[Dict[str, Any]]) -> str:
        """Stage 3: Generate the final clean answer."""
        self._log("Stage 3: Generating final answer")
        
        context_str = self._format_context(chunks)
        if not context_str:
            context_str = "No relevant context found in the NCERT database."
            
        prompt = ANSWER_GENERATION_PROMPT.format(
            retrieved_chunks=context_str,
            user_question=user_question
        )
        
        # Low temperature for factual generation
        raw_answer = self._call_ollama(prompt, temperature=0.1)
        self._log(f"Raw Answer Length: {len(raw_answer)}")
        
        cleaned_answer = self._clean_answer(raw_answer)
        
        # Final verification
        if not self._verify_response_clean(cleaned_answer):
            self._log("WARNING: Cleaned answer still contains red flags. Running strict clean.")
            # Extreme fallback: take only the last paragraph or force a generic response
            lines = cleaned_answer.split('\n')
            safe_lines = [l for l in lines if not any(x in l.upper() for x in ["THOUGHT:", "ACTION:", "QUERY", "REASONING:"])]
            cleaned_answer = "\n".join(safe_lines).strip()
            
            if not cleaned_answer:
                cleaned_answer = "I found some information, but encountered an error formatting it securely. Please try asking in a different way."

        return cleaned_answer

        # Pipeline has been superseded by `query` which executes confidence checking

    def agent_loop(self, user_question: str, max_rounds: int = 3, subject_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Iterative pipeline with evaluation.
        Still returns ONLY the final answer in the main dict.
        """
        start_time = time.time()
        self._log("Starting Agent Loop")
        
        accumulated_chunks = []
        previous_queries = []
        
        # Round 1: Just do a normal retrieve
        current_queries = self._generate_search_queries(user_question)
        
        for round_num in range(1, max_rounds + 1):
            self._log(f"--- Agent Round {round_num} ---")
            previous_queries.extend(current_queries)
            
            # Retrieve
            new_chunks = self._retrieve_chunks(current_queries, subject_filter)
            
            # Deduplicate into accumulated
            existing_texts = {c.get("text", "") for c in accumulated_chunks}
            for c in new_chunks:
                if c.get("text", "") not in existing_texts:
                    accumulated_chunks.append(c)
                    existing_texts.add(c.get("text", ""))
                    
            # Keep top N overall
            accumulated_chunks = sorted(accumulated_chunks, key=lambda x: x.get("score", float('inf')))[:self.max_unique_chunks]
            
            if round_num == max_rounds:
                self._log("Max rounds reached. Breaking to generate answer.")
                break
                
            # Evaluate
            context_str = self._format_context(accumulated_chunks)
            eval_prompt = AGENT_THINK_PROMPT.format(
                user_question=user_question,
                previous_queries=", ".join(previous_queries),
                accumulated_context=context_str if context_str else "No context found yet."
            )
            
            eval_response = self._call_ollama(eval_prompt, temperature=0.3)
            self._log(f"Evaluation: {eval_response[:100]}...")
            
            if "ACTION: ANSWER" in eval_response.upper():
                self._log("Agent decided it has enough info.")
                break
                
            elif "ACTION: SEARCH" in eval_response.upper():
                self._log("Agent decided to search more.")
                # Parse new queries
                current_queries = []
                for line in eval_response.split('\n'):
                    if line.strip().upper().startswith("QUERY") and ":" in line:
                        q = line.split(":", 1)[1].strip()
                        q = re.sub(r'^["\']|["\']$', '', q) 
                        current_queries.append(q)
                
                if not current_queries:
                    self._log("Agent wanted to search but gave no queries. Forcing answer.")
                    break
            else:
                self._log("Failed to parse agent evaluation action. Forcing answer.")
                break
                
        # Final Generation
        answer = self._generate_answer(user_question, accumulated_chunks)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        result = {
            "answer": answer,
            "chunks": accumulated_chunks,
            "time_ms": int(elapsed_ms)
        }
        
        if self.debug:
            result["debug"] = {
                "total_rounds": round_num,
                "all_queries": previous_queries
            }
            
        return result

        # THIS METHOD IS NOW IN AgentRAGFixesMixin


# ============================================================================
# Test Cases
# ============================================================================
if __name__ == "__main__":
    print("AgentRAG Mock Test Runner")
    import numpy as np
    
    # Mock implementations for local testing without real models
    class MockEmbedder:
        def encode(self, texts, **kwargs):
            return np.random.rand(1, 384)

    class MockFaiss:
        def search(self, vec, k):
            return np.array([[0.9]]), np.array([[0]])

    mock_chunks = [
        {"subject": "history", "chapter": "1", "text": "The Maratha Empire emerged after the decline of the Mughals. The British East India Company gradually expanded its control over India."},
    ]

    # Initialize in simple mode
    agent = AgentRAG(
        ollama_model="phi3",
        faiss_index=MockFaiss(),
        embedder=MockEmbedder(),
        chunks=mock_chunks,
        mode="simple",
        debug=True
    )
    
    print("\n" + "="*50)
    print("Test 1: Indirect question (The original failing case)")
    print("="*50)
    res1 = agent.query("who ruled india after mughals")
    print(f"CLEAN ANSWER:\n{res1['answer']}")
    
    print("\n" + "="*50)
    print("Test 2: First type question")
    print("="*50)
    res2 = agent.query("who is the first president of india")
    print(f"CLEAN ANSWER:\n{res2['answer']}")
    
    print("\n" + "="*50)
    print("Test 3: Comparison question")
    print("="*50)
    res3 = agent.query("difference between fundamental rights and directive principles")
    print(f"CLEAN ANSWER:\n{res3['answer']}")

    print("\n" + "="*50)
    print("Test 4: Direct factual question")
    print("="*50)
    res4 = agent.query("what is the capital of Maharashtra")
    print(f"CLEAN ANSWER:\n{res4['answer']}")
    
    print("\n" + "="*50)
    print("Test 5: Not in DB")
    print("="*50)
    res5 = agent.query("explain quantum physics")
    print(f"CLEAN ANSWER:\n{res5['answer']}")
