import re
import time
from typing import List, Dict, Any, Optional

NOT_IN_NCERT_MESSAGES = {
    "current_affairs": (
        "📚 This is a current affairs question.\n\n"
        "NCERT textbooks do not contain information about current "
        "political leaders, recent appointments, or present-day events.\n\n"
        "📖 For current affairs, please refer to:\n"
        "• Pratiyogita Darpan\n"
        "• Chronicle Magazine\n"
        "• The Hindu / Indian Express newspapers\n"
        "• Vajiram & Ravi current affairs notes"
    ),
    
    "not_in_syllabus": (
        "📚 DATA NOT PRESENT IN NCERT\n\n"
        "This topic is not covered in NCERT textbooks for Classes 6-10.\n"
        "The information you're looking for may be in:\n"
        "• Higher class textbooks (Class 11-12)\n"
        "• Specialized reference books\n"
        "• Current affairs magazines"
    ),
    
    "low_confidence": (
        "📚 LIMITED DATA FOUND\n\n"
        "I found some related information in NCERT textbooks, but it may "
        "not fully answer your question. Here's what I found:\n\n"
        "{partial_answer}\n\n"
        "⚠️ This answer may be incomplete. Please verify from your textbook."
    )
}

CURRENT_AFFAIRS_PATTERNS = [
    # Direct current affairs keywords
    "latest budget",
    "current gdp",
    "population of india"
]


class AgentRAGFixesMixin:
    """
    Drop-in replacement methods for AgentRAG that fix regex crashes,
    add confidence scoring, and handle out-of-domain questions.
    """
    
    HARD_THRESHOLD = 40       # Below this = DATA NOT PRESENT
    SOFT_THRESHOLD = 45       # Below this = answer with disclaimer
    CONFIDENT_THRESHOLD = 50  # Above this = answer normally

    def _is_current_affairs(self, question: str) -> bool:
        """
        Check if question is asking about dynamic info
        that textbooks would never contain.
        """
        q_lower = question.lower().strip()
        
        # Check direct pattern matches
        for pattern in CURRENT_AFFAIRS_PATTERNS:
            if pattern in q_lower:
                return True
        
        return False

    def _check_chunk_relevance(self, question: str, chunk: str) -> bool:
        """
        Quick keyword overlap check. 
        Detects if history is injected via 'User Question: ' and parses just the query.
        """
        # If history was prepended, only look at the actual user question
        if "User Question:" in question:
            parsed_question = question.split("User Question:")[-1].strip()
        else:
            parsed_question = question.strip()
            
        stopwords = {
            "what", "who", "is", "the", "a", "an", "of", "in", "was", "were",
            "are", "list", "all", "tell", "me", "about", "give", "name",
            "how", "which", "when", "where", "why", "do", "does", "did",
            "can", "could", "will", "would", "should", "has", "have", "had",
            "this", "that", "these", "those", "it", "its", "and", "or",
            "for", "to", "from", "with", "by", "at", "on", "current",
            "present", "first", "last", "our", "previous", "context", "user", 
            "question", "explain", "describe", "discuss"
        }
        
        question_words = set(parsed_question.lower().split()) - stopwords
        chunk_lower = chunk.lower()
        
        matches = sum(1 for word in question_words if word in chunk_lower)
        
        if len(question_words) == 0:
            return True
            
        # For short queries, if even ONE important word matches, consider it relevant
        if len(question_words) <= 5 and matches >= 1:
            return True
        
        overlap_ratio = matches / len(question_words)
        return overlap_ratio >= 0.15

    def _clean_answer(self, raw_answer: str) -> str:
        """
        Remove internal reasoning tokens, prompt leakage, and formatting artifacts.
        Each pattern is applied separately with proper flag handling to avoid
        'global flags not at the start of expression' errors.
        """
        cleaned = raw_answer.strip()
        
        # Define patterns as tuples: (pattern_string, flags, replacement)
        # NEVER put (?i) or (?s) inside the pattern string
        patterns = [
            (r"^THOUGHT:.*$", re.MULTILINE | re.IGNORECASE, ""),
            (r"^ACTION:.*$", re.MULTILINE | re.IGNORECASE, ""),
            (r"^REASONING:.*$", re.MULTILINE | re.IGNORECASE, ""),
            (r"^QUERY\d*:.*$", re.MULTILINE | re.IGNORECASE, ""),
            (r"^SEARCH:.*$", re.MULTILINE | re.IGNORECASE, ""),
            
            # Phi-3 special tokens
            (r"<\|system\|>", 0, ""),
            (r"<\|user\|>", 0, ""),
            (r"<\|assistant\|>", 0, ""),
            (r"<\|end\|>", 0, ""),
            (r"<\|endoftext\|>", 0, ""),
            
            # Prompt leakage / Hallucinated prompt structures
            (r"^Context.*?:.*$", re.MULTILINE | re.IGNORECASE, ""),
            (r"^Student Question:.*$", re.MULTILINE | re.IGNORECASE, ""),
            (r"^User Question:.*$", re.MULTILINE | re.IGNORECASE, ""),
            (r"^Question:.*$", re.MULTILINE | re.IGNORECASE, ""),
            (r"^Give a clear, direct answer.*$", re.MULTILINE | re.IGNORECASE, ""),
            (r"^Give a clean, factual answer.*$", re.MULTILINE | re.IGNORECASE, ""),
            (r"^---\s*$", re.MULTILINE, ""),
            
            # Preamble phrases
            (r"^Based on the context[,.]?\s*", re.IGNORECASE, ""),
            (r"^Based on the retrieved[,.]?\s*", re.IGNORECASE, ""),
            (r"^Based on the passages?[,.]?\s*", re.IGNORECASE, ""),
            (r"^According to the context[,.]?\s*", re.IGNORECASE, ""),
            (r"^According to the passages?[,.]?\s*", re.IGNORECASE, ""),
            (r"^According to the NCERT[,.]?\s*", re.IGNORECASE, ""),
        ]
        
        # Run three passes to catch nested tags
        for _ in range(3):
            old_cleaned = cleaned
            for pattern_str, flags, replacement in patterns:
                try:
                    cleaned = re.sub(pattern_str, replacement, cleaned, flags=flags)
                except re.error as e:
                    if hasattr(self, 'debug') and self.debug:
                        print(f"[DEBUG] Regex error for pattern '{pattern_str}': {e}")
                    continue
            
            if cleaned == old_cleaned:
                break
                
        # Clean up resulting whitespace
        try:
            cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        except re.error:
            pass
            
        cleaned = cleaned.strip()

        # Remove semantic duplicate sentences and paragraphs (halts LLM loops)
        lines = cleaned.split('\n')
        dedup_lines: List[str] = []
        seen_items: List[Any] = []
        
        for line in lines:
            line_strip = line.strip()
            if not line_strip:
                dedup_lines.append(line)
                continue
            
            # Remove isolated bullet points and numbers (e.g. "3.", "-", "• ")
            if len(re.sub(r'[-*•\d.\s()]', '', line_strip)) == 0:
                continue
            
            # Simple exact match for list elements
            if re.match(r'^[-*•\d.]\s+', line_strip):
                compare_str = re.sub(r'^[-*•\d.]\s*', '', line_strip).lower()
                is_dup = False
                for seen in seen_items:
                    if isinstance(seen, str) and seen == compare_str:
                        is_dup = True
                        break
                if not is_dup:
                    dedup_lines.append(line)
                    seen_items.append(compare_str)
                continue

            # For normal text blocks, split into sentences and fuzzy match
            sentences = re.split(r'(?<=[.!?])\s+', line_strip)
            valid_sentences = []
            
            for s in sentences:
                s_strip = s.strip()
                if not s_strip:
                    continue
                    
                words = set(re.findall(r'\b\w+\b', s_strip.lower()))
                if len(words) < 5:
                    valid_sentences.append(s) # Too short to reliably fuzzy match
                    continue
                    
                is_dup = False
                for seen in seen_items:
                    if isinstance(seen, set):
                        overlap = len(words.intersection(seen))
                        min_len = min(len(words), len(seen))
                        if min_len > 0 and (overlap / min_len) > 0.8:
                            is_dup = True
                            break
                
                if not is_dup:
                    valid_sentences.append(s)
                    seen_items.append(words)
            
            if valid_sentences:
                para = ' '.join(valid_sentences).strip()
                dedup_lines.append(para)

        # Drop trailing incomplete sentence/paragraph if it exists at the very end
        while dedup_lines and not dedup_lines[-1].strip():
            dedup_lines.pop()
            
        if dedup_lines:
            last_line = dedup_lines[-1].strip()
            if last_line and not last_line.endswith(('.', '!', '?', '"', "'", ')')):
                cutoff_words = {'of', 'the', 'and', 'in', 'to', 'a', 'an', 'is', 'are', 'was', 'were', 'with', 'for', 'by', 'on', 'at', 'from', 'as', 'but', 'or', 'which', 'that', 'such', 'like', 'these'}
                last_word = last_line.split()[-1].lower() if last_line.split() else ""
                
                # Drop the last paragraph if it ends in a sudden cutoff mid-thought
                if last_word in cutoff_words or len(last_line.split()) < 6:
                    if len(dedup_lines) > 1:
                        dedup_lines.pop()
                    else:
                        dedup_lines[-1] = last_line + "..."
                else:
                    dedup_lines[-1] = last_line + "..."
                    
        cleaned = '\n'.join(dedup_lines)
        
        if not cleaned or len(cleaned) < 10:
            return "I could not generate a proper answer for this question. Please try rephrasing."
            
        return cleaned

    def _aggressive_clean(self, text: str) -> str:
        """
        Last resort cleaning if _clean_answer() still leaves artifacts.
        Strips everything before the first actual sentence.
        """
        lines = text.split("\n")
        clean_lines = []
        
        skip_prefixes = [
            "thought:", "action:", "reasoning:", "query",
            "search:", "context passages:", "student question:",
            "---", "based on the context", "based on the retrieved",
            "<|"
        ]
        
        for line in lines:
            stripped = line.strip().lower()
            if not stripped:
                continue
            if any(stripped.startswith(prefix) for prefix in skip_prefixes):
                continue
            clean_lines.append(line.strip())
        
        result = "\n".join(clean_lines).strip()
        
        if not result or len(result) < 10:
            return "I could not generate a proper answer. Please try rephrasing your question."
            
        return result

    def _verify_response_clean(self, answer: str) -> bool:
        """Check that no internal tokens leaked into the response."""
        red_flags = [
            "THOUGHT:", "ACTION:", "SEARCH:", "REASONING:",
            "QUERY1:", "QUERY2:", "QUERY3:", "QUERY4:",
            "<|system|>", "<|user|>", "<|end|>", "<|assistant|>",
            "Context Passages:", "Student Question:",
            "Based on the retrieved"
        ]
        answer_upper = answer.upper()
        return not any(flag.upper() in answer_upper for flag in red_flags)

    def _retrieve_chunks(self, search_queries: List[str], subject_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieve chunks from FAISS and compute confidence scores.
        """
        all_chunks = []
        all_raw_chunks = []
        all_scores = []
        seen_chunks = set()
        
        for query in search_queries:
            try:
                # Embed query
                query_vec = self.embedder.encode(
                    [query], normalize_embeddings=True, convert_to_numpy=True
                ).astype('float32')
                
                search_k = 15 if subject_filter else 3
                distances, indices = self.faiss_index.search(query_vec, search_k)
                
                for dist, idx in zip(distances[0], indices[0]):
                    if idx < 0 or idx >= len(self.chunks):
                        continue
                        
                    chunk_data = self.chunks[idx].copy()
                    
                    if subject_filter and chunk_data.get("subject", "").lower() != subject_filter.lower():
                        continue
                        
                    chunk_text = chunk_data.get("text", "")
                    chunk_key = chunk_text[:100]
                    
                    if chunk_key not in seen_chunks:
                        seen_chunks.add(chunk_key)
                        
                        # FAISS is built with IndexFlatIP using normalized embeddings.
                        # This means 'dist' is exactly the Cosine Similarity (typically 0.0 to 1.0).
                        # We convert this directly to a percentage.
                        dist_val = float(dist)
                        similarity_pct = max(0.0, dist_val) * 100.0
                        
                        chunk_data["score"] = similarity_pct
                        all_chunks.append(chunk_text)
                        all_raw_chunks.append(chunk_data)
                        all_scores.append(similarity_pct)
                        
            except Exception as e:
                if hasattr(self, 'debug') and self.debug:
                    print(f"[DEBUG] Retrieval error for query '{query}': {e}")
                continue

        if not all_raw_chunks:
            return {
                "chunks": [],
                "raw_chunks": [],
                "scores": [],
                "best_score": 0.0,
            }
            
        best_score = max(all_scores)
            
        paired = sorted(zip(all_scores, all_chunks, all_raw_chunks), key=lambda x: x[0], reverse=True)
            
        sorted_chunks = [chunk for _, chunk, _ in paired][:8]
        sorted_raw = [raw for _, _, raw in paired][:8]
        sorted_scores = [score for score, _, _ in paired][:8]
        
        return {
            "chunks": sorted_chunks,
            "raw_chunks": sorted_raw,
            "scores": sorted_scores,
            "best_score": best_score,
        }

    def query(self, user_question: str, subject_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Main entry point with strict confidence gating.
        """
        start_time = time.time()
        
        # ==========================================
        # GATE 1: Current affairs early detection
        # ==========================================
        if self._is_current_affairs(user_question):
            elapsed = int((time.time() - start_time) * 1000)
            return {
                "answer": NOT_IN_NCERT_MESSAGES["current_affairs"],
                "confidence": "none",
                "chunks": [],
                "time_ms": elapsed
            }
        
        # ==========================================
        # STEP 1: Generate search queries (hidden)
        # ==========================================
        search_queries = self._generate_search_queries(user_question)
        
        # ==========================================
        # STEP 2: Retrieve from FAISS
        # ==========================================
        retrieval_result = self._retrieve_chunks(search_queries, subject_filter)
        
        chunks = retrieval_result["chunks"]
        scores = retrieval_result["scores"]  # These are percentage similarities (0-100)
        best_score = retrieval_result["best_score"]
        raw_chunks = retrieval_result["raw_chunks"]
        
        if hasattr(self, 'debug') and self.debug:
            print(f"[DEBUG] Best score: {best_score}%")
            print(f"[DEBUG] All scores: {scores}")
            
        # ==========================================
        # GATE 2: Hard score threshold
        # ==========================================
        
        # STRICT RULE: If best chunk is below 65%, refuse to answer
        # Scores around 49-55% are essentially random noise
        if best_score < self.HARD_THRESHOLD:
            elapsed = int((time.time() - start_time) * 1000)
            return {
                "answer": NOT_IN_NCERT_MESSAGES["not_in_syllabus"],
                "confidence": "none",
                "chunks": raw_chunks, # Can pass the chunks for UI debugging
                "time_ms": elapsed
            }
        
        # ==========================================
        # GATE 3: Keyword relevance check
        # ==========================================
        # Even above threshold, verify chunks actually relate to the question
        relevant_chunks = []
        relevant_raw_chunks = []
        relevant_scores = []
        
        for chunk, raw_chunk, score in zip(chunks, raw_chunks, scores):
            if self._check_chunk_relevance(user_question, chunk):
                relevant_chunks.append(chunk)
                relevant_raw_chunks.append(raw_chunk)
                relevant_scores.append(score)
        
        # If no chunks pass relevance check, refuse to answer
        if not relevant_chunks:
            elapsed = int((time.time() - start_time) * 1000)
            return {
                "answer": NOT_IN_NCERT_MESSAGES["not_in_syllabus"],
                "confidence": "none",
                "chunks": raw_chunks, # Passed for debugging on UI
                "time_ms": elapsed
            }
        
        # ==========================================
        # STEP 3: Generate answer with relevant chunks only
        # ==========================================
        
        if hasattr(self, '_format_context'):
            context_str = self._format_context(relevant_raw_chunks)
        else:
            context_str = "\n---\n".join(relevant_chunks)

        try:
             raw_answer = self._generate_answer(user_question, relevant_raw_chunks)
        except Exception:
             raw_answer = self._generate_answer(user_question, context_str)
        
        # Clean the answer
        final_answer = self._clean_answer(raw_answer)
        
        # Verify clean
        if not self._verify_response_clean(final_answer):
            final_answer = self._aggressive_clean(final_answer)
        
        # Decide response format based on confidence level
        if best_score < self.HARD_THRESHOLD:
            # BLOCK: No relevant data found at all
            elapsed = int((time.time() - start_time) * 1000)
            return {
                "answer": NOT_IN_NCERT_MESSAGES["not_in_syllabus"],
                "confidence": "none",
                "chunks": relevant_raw_chunks,
                "time_ms": elapsed
            }

        elif best_score < self.SOFT_THRESHOLD:
            # LOW CONFIDENCE: Answer exists but may be incomplete
            # Add disclaimer ONLY for this range
            final_answer = (
                "📚 LIMITED DATA FOUND\n\n"
                "I found some related information in NCERT textbooks, "
                "but it may not fully answer your question. "
                "Here's what I found:\n\n"
                f"{final_answer}\n\n"
                "⚠️ This answer may be incomplete. Please verify from your textbook."
            )

        else:
            # HIGH CONFIDENCE: Good match found, answer normally
            # NO disclaimer, NO warning, just the clean answer
            # Do nothing — final_answer is already clean
            pass
        
        elapsed = int((time.time() - start_time) * 1000)
        
        result = {
            "answer": final_answer,
            "confidence": "high" if best_score >= self.CONFIDENT_THRESHOLD else "medium",
            "chunks": relevant_raw_chunks,
            "time_ms": elapsed
        }
        
        if hasattr(self, 'debug') and self.debug:
            result["debug"] = {
                "search_queries": search_queries,
                "chunks_count": len(relevant_raw_chunks),
                "best_score": best_score,
            }
            
        return result


def calibrate_thresholds(faiss_index, embedder, chunks):
    """
    Test with known good and bad questions to find threshold values.
    """
    good_questions = [
        "what is the Indian constitution",
        "who were the Mughals",
        "what are fundamental rights",
        "explain the water cycle",
        "what is GDP"
    ]
    
    bad_questions = [
        "current chief minister of Andhra Pradesh",
        "who won the cricket world cup 2024",
        "explain quantum computing",
        "latest iPhone specifications",
        "what is ChatGPT"
    ]
    
    import numpy as np
    
    print("=== GOOD QUESTIONS (should have HIGH Percentage Score) ===")
    for q in good_questions:
        embedding = embedder.encode([q], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        distances, indices = faiss_index.search(embedding, k=3)
        best_dist = distances[0][0]
        best_pct = max(0.0, (1.0 - best_dist / 2.0)) * 100.0
        print(f"  Q: {q}")
        print(f"  Best L2 Dist: {best_dist:.4f} | Conv Pct: {best_pct:.2f}%")
        preview = chunks[indices[0][0]].get("text", "")[:80] if isinstance(chunks[indices[0][0]], dict) else chunks[indices[0][0]][:80]
        print(f"  Top chunk preview: {preview}...")
        print()
    
    print("=== BAD QUESTIONS (should have LOW Percentage Score) ===")
    for q in bad_questions:
        embedding = embedder.encode([q], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        distances, indices = faiss_index.search(embedding, k=3)
        best_dist = distances[0][0]
        best_pct = max(0.0, (1.0 - best_dist / 2.0)) * 100.0
        print(f"  Q: {q}")
        print(f"  Best L2 Dist: {best_dist:.4f} | Conv Pct: {best_pct:.2f}%")
        preview = chunks[indices[0][0]].get("text", "")[:80] if isinstance(chunks[indices[0][0]], dict) else chunks[indices[0][0]][:80]
        print(f"  Top chunk preview: {preview}...")
        print()
    
    print("=== THRESHOLD RECOMMENDATION ===")
    print("Set HARD_THRESHOLD just above the highest 'bad question' percentage.")
    print("Set CONFIDENT_THRESHOLD just below the lowest 'good question' percentage.")
