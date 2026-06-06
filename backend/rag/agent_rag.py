"""
Agentic RAG Query Rewriting and Iterative Retrieval Layer
"""
import re
import requests
from typing import List, Dict, Any, Optional
import numpy as np

# ============================================================================
# Prompts
# ============================================================================

QUERY_GENERATION_PROMPT = """<|system|>
You are a search query generator for an Indian NCERT textbook knowledge base.

Your job: Given a student's question, generate exactly 3 search queries that will help retrieve the most relevant textbook passages.

Rules:
- Think about what INFORMATION is needed, not just rephrase the question
- Include specific names, dates, events, periods that might appear in textbooks
- If the question is INDIRECT (like "who was the first PM"), think about what the textbook ACTUALLY says (like "Nehru served as prime minister from 1947")
- Cover different angles: direct match, related context, and broader topic
<|user|>
Student Question: {user_question}

Respond ONLY in this exact format, nothing else:
QUERY1: <first search query>
QUERY2: <second search query>
QUERY3: <third search query>
<|assistant|>"""

ANSWER_GENERATION_PROMPT = """<|system|>
You are an NCERT textbook tutor helping Indian students prepare for competitive exams.

Your job: Answer the student's question using ONLY the provided context passages. Think step by step.

Rules:
- ONLY use information from the context below. Do not use outside knowledge.
- If the context contains the answer INDIRECTLY, reason through it step by step.
- If the context does NOT contain enough information, say "I don't have enough information in the textbook to answer this accurately."
- Be concise and exam-focused.
<|user|>
Context Passages:
{retrieved_chunks}

Student Question: {user_question}

Respond ONLY in this exact format:
REASONING: <work through what the context tells you, connect the dots>
ANSWER: <your final concise answer>
<|assistant|>"""

AGENT_PROMPT = """<|system|>
You are a retrieval-augmented tutor agent for NCERT textbooks covering Polity, History, Economy, and Geography.

You will work in a loop: THINK -> SEARCH -> EVALUATE -> either SEARCH AGAIN or ANSWER.

FORMAT - you must respond in EXACTLY one of these two formats:

Option A (need more info):
THOUGHT: <what you know so far and what is missing>
ACTION: SEARCH
QUERIES:
- <query 1>
- <query 2>
- <query 3>

Option B (ready to answer):
THOUGHT: <your step-by-step reasoning using the context>
ACTION: ANSWER
RESPONSE: <your final answer to the student>

RULES:
- Maximum 3 search rounds. After 3 rounds you MUST answer with whatever you have.
- Each search query should be SHORT (3-8 words), specific, and different from previous queries.
- When you get context back, CHECK if it actually answers the question. If not, search with DIFFERENT terms.
- For INDIRECT questions, think about what words the textbook would actually use.
- Never make up facts. Only use retrieved context.
<|user|>
CURRENT STATE:
Question: {user_question}
Search Round: {round_number} of 3
Previous Queries: {previous_queries}
Retrieved Context So Far: {accumulated_context}

Your response:
<|assistant|>"""

AGENT_FORCE_ANSWER_PROMPT = """<|system|>
You are an NCERT textbook tutor.

You have searched for information but reached the maximum number of search rounds.
Now, you MUST provide the best possible answer to the student's question based ONLY on the retrieved context.
<|user|>
Context Passages:
{retrieved_chunks}

Student Question: {user_question}

Respond ONLY in this exact format:
REASONING: <your reasoning>
ANSWER: <your answer>
<|assistant|>"""

# ============================================================================
# API Call Helper
# ============================================================================

def call_phi3(prompt: str, model: str = "phi3", temperature: float = 0.3) -> str:
    """
    Calls the local Ollama API to generate a text response.
    
    Args:
        prompt (str): The formatted prompt.
        model (str): The Ollama model name.
        temperature (float): The generation temperature.
        
    Returns:
        str: The generated response from the LLM.
    """
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "top_p": 0.9,
                    "num_predict": 512
                }
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json()["response"].strip()
    except Exception as e:
        print(f"Error calling Ollama API: {e}")
        return ""

# ============================================================================
# Parsing Helpers
# ============================================================================

def parse_agent_response(response: str) -> Dict[str, Any]:
    """
    Parse the agent's structured output.
    
    Args:
        response (str): Raw string output from the LLM.
        
    Returns:
        Dict: A dictionary containing the action, thought, and either queries or answer.
    """
    response = response.strip()
    
    # Check if agent wants to answer
    if "ACTION: ANSWER" in response:
        thought = ""
        if "THOUGHT:" in response:
            thought = response.split("THOUGHT:")[1].split("ACTION:")[0].strip()
        
        answer = ""
        if "RESPONSE:" in response:
            answer = response.split("RESPONSE:")[1].strip()
        else:
            # Fallback
            answer = response.split("ACTION: ANSWER")[-1].strip()
            
        return {
            "action": "ANSWER",
            "thought": thought,
            "answer": answer
        }
    
    # Check if agent wants to search
    elif "ACTION: SEARCH" in response:
        thought = ""
        if "THOUGHT:" in response:
            thought = response.split("THOUGHT:")[1].split("ACTION:")[0].strip()
        
        queries = []
        lines = response.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("- ") and len(line) > 2:
                queries.append(line[2:].strip())
            elif line.startswith("QUERY") and ":" in line:
                queries.append(line.split(":", 1)[1].strip())
        
        return {
            "action": "SEARCH",
            "thought": thought,
            "queries": queries[:3]
        }
    
    # Fallback - treat entire response as answer
    return {
        "action": "ANSWER",
        "thought": "Failed to parse agent response correctly.",
        "answer": response
    }

def parse_answer_response(response: str) -> Dict[str, str]:
    """Parse the reasoning and answer from the simple 2-stage pipeline's answer generation."""
    reasoning = ""
    answer = response
    
    if "REASONING:" in response and "ANSWER:" in response:
        reasoning = response.split("REASONING:")[1].split("ANSWER:")[0].strip()
        answer = response.split("ANSWER:")[1].strip()
    elif "ANSWER:" in response:
        answer = response.split("ANSWER:")[1].strip()
        
    return {"reasoning": reasoning, "answer": answer}

def parse_queries(response: str) -> List[str]:
    """Parse queries from the simple 2-stage pipeline's query generation."""
    queries = []
    lines = response.strip().split("\n")
    for line in lines:
        line = line.strip()
        if line.startswith("QUERY") and ":" in line:
            queries.append(line.split(":", 1)[1].strip())
        elif line.startswith("- "):
             queries.append(line[2:].strip())
    return queries[:3] # Ensure max 3 queries

# ============================================================================
# Main Agent Class
# ============================================================================

class AgentRAG:
    """
    Agentic RAG wrapper for an existing retrieval backend.
    Supports both a 'simple' two-stage generation pipeline and an 'agent' loop approach.
    """
    def __init__(
        self, 
        ollama_model: str, 
        faiss_index: Any, 
        embedder: Any, 
        chunks: List[Dict[str, Any]], 
        mode: str = "simple",
        verbose: bool = False
    ):
        self.ollama_model = ollama_model
        self.faiss_index = faiss_index
        self.embedder = embedder
        self.chunks = chunks
        self.mode = mode
        self.verbose = verbose
        
    def _log(self, msg: str):
        if self.verbose:
            print(f"[AgentRAG Log] {msg}")

    def _retrieve_chunks(self, query: str, top_k: int = 3, subject_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve chunks from the FAISS index using the embedder."""
        query_vec = self.embedder.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)
        
        search_k = top_k * 5 if subject_filter else top_k
        scores, indices = self.faiss_index.search(query_vec, search_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx].copy()
            
            if subject_filter and chunk.get("subject", "").lower() != subject_filter.lower():
                continue
                
            chunk["score"] = float(score)
            results.append(chunk)
            if len(results) >= top_k:
                break
            
        return results

    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Format chunks into a readable string context."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            subject = chunk.get("subject", "Unknown").capitalize()
            chapter = chunk.get("chapter", "Unknown")
            text = chunk.get("text", "")
            parts.append(f"[Source {i} | {subject} - {chapter}]\n{text}")
        return "\n\n".join(parts)

    def simple_pipeline(self, user_question: str, subject_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Stage 1: Generate rewritten queries.
        Stage 2: Retrieve chunks for each query.
        Stage 3: Answer based on aggregated context.
        """
        self._log("Running Simple Pipeline")
        
        # Stage 1: Query Generation
        q_prompt = QUERY_GENERATION_PROMPT.format(user_question=user_question)
        q_response = call_phi3(q_prompt, self.ollama_model)
        queries = parse_queries(q_response)
        
        if not queries:
            self._log("Fallback: generating single query from original question.")
            queries = [user_question]
            
        self._log(f"Generated queries: {queries}")
        
        # Stage 2: Retrieval
        gathered_chunks: Dict[int, Dict[str, Any]] = {}
        for q in queries:
            retrieved = self._retrieve_chunks(q, top_k=3, subject_filter=subject_filter)
            for chunk in retrieved:
                # Deduplicate by text content hash or chunk index if available
                chunk_id = hash(chunk.get("text", ""))
                if chunk_id not in gathered_chunks:
                    gathered_chunks[chunk_id] = chunk
                    
            if len(gathered_chunks) >= 10: # Cap at 10 unique chunks
                break
                
        final_chunks_list = list(gathered_chunks.values())[:10]
        self._log(f"Retrieved {len(final_chunks_list)} unique chunks.")
        
        # Stage 3: Answer Generation
        context_str = self._format_context(final_chunks_list)
        a_prompt = ANSWER_GENERATION_PROMPT.format(
            retrieved_chunks=context_str if context_str else "No context found.",
            user_question=user_question
        )
        
        a_response = call_phi3(a_prompt, self.ollama_model)
        parsed_ans = parse_answer_response(a_response)
        
        self._log(f"Reasoning: {parsed_ans['reasoning']}")
        return {
            "answer": parsed_ans["answer"],
            "chunks": final_chunks_list
        }

    def agent_loop(self, user_question: str, max_rounds: int = 3, subject_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Full agent loop mode: THINK -> SEARCH -> EVALUATE.
        """
        self._log("Running Agent Loop Mode")
        accumulated_context_str = ""
        previous_queries = []
        gathered_chunks: Dict[int, Dict[str, Any]] = {}
        
        for round_num in range(1, max_rounds + 1):
            self._log(f"--- Round {round_num} ---")
            
            prompt = AGENT_PROMPT.format(
                user_question=user_question,
                round_number=round_num,
                previous_queries=", ".join(previous_queries) if previous_queries else "None",
                accumulated_context=accumulated_context_str if accumulated_context_str else "None yet."
            )
            
            response = call_phi3(prompt, self.ollama_model)
            parsed = parse_agent_response(response)
            
            self._log(f"Agent Action: {parsed['action']}")
            self._log(f"Agent Thought: {parsed.get('thought', '')}")
            
            if parsed["action"] == "ANSWER":
                return {
                    "answer": parsed["answer"],
                    "chunks": list(gathered_chunks.values())[:10]
                }
                
            # Perform SEARCH
            queries = parsed.get("queries", [])
            self._log(f"Agent Search Queries: {queries}")
            
            if not queries:
                self._log("No valid queries found. Aborting round.")
                continue
                
            for q in queries:
                if q not in previous_queries:
                    previous_queries.append(q)
                retrieved = self._retrieve_chunks(q, top_k=3, subject_filter=subject_filter)
                for chunk in retrieved:
                    chunk_id = hash(chunk.get("text", ""))
                    if chunk_id not in gathered_chunks:
                        gathered_chunks[chunk_id] = chunk
                        
            # Rebuild accumulated context
            final_chunks_list = list(gathered_chunks.values())[:10]
            accumulated_context_str = self._format_context(final_chunks_list)

        # Force answer after max rounds
        self._log("Max rounds reached. Forcing answer.")
        final_prompt = AGENT_FORCE_ANSWER_PROMPT.format(
            retrieved_chunks=accumulated_context_str if accumulated_context_str else "No context found.",
            user_question=user_question
        )
        final_response = call_phi3(final_prompt, self.ollama_model)
        parsed_ans = parse_answer_response(final_response)
        return {
            "answer": parsed_ans["answer"],
            "chunks": list(gathered_chunks.values())[:10]
        }

    def query(self, user_question: str, subject_filter: Optional[str] = None) -> Dict[str, Any]:
        """Main entry point to answer a user question. Returns dict with dict['answer'] and dict['chunks']."""
        if self.mode == "agent":
            return self.agent_loop(user_question, subject_filter=subject_filter)
        else:
            return self.simple_pipeline(user_question, subject_filter=subject_filter)

# ============================================================================
# Test Cases
# ============================================================================
if __name__ == "__main__":
    pass
    # Examples to test functionality:
    # 
    # class MockEmbedder:
    #     def encode(self, texts, **kwargs):
    #         import numpy as np
    #         return np.random.rand(1, 384)
    #
    # class MockFaiss:
    #     def search(self, vec, k):
    #         import numpy as np
    #         return np.array([[0.9, 0.8, 0.7]]), np.array([[0, 1, 2]])
    #
    # mock_chunks = [
    #     {"subject": "history", "chapter": "m", "text": "Chunk 0"},
    #     {"subject": "history", "chapter": "m", "text": "Chunk 1"},
    #     {"subject": "history", "chapter": "m", "text": "Chunk 2"}
    # ]
    #
    # agent = AgentRAG(
    #     ollama_model="phi3",
    #     faiss_index=MockFaiss(),
    #     embedder=MockEmbedder(),
    #     chunks=mock_chunks,
    #     mode="simple",
    #     verbose=True
    # )
    # 
    # # Test 1: Indirect question
    # print(agent.query("Who is the first prime minister of India?"))
    #
    # # Test 2: Temporal/sequential question  
    # print(agent.query("Name the rulers/empires after the Mughal empire"))
    #
    # # Test 3: Comparison question
    # print(agent.query("What is the difference between fundamental rights and directive principles?"))
    #
    # # Test 4: Current affairs (if that data exists)
    # print(agent.query("Who is the current Chief Justice of India?"))
    #
    # # Test 5: Multi-hop
    # print(agent.query("Which constitutional amendment is related to the GST council?"))
