from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from src.llm_client import SafeLLMClient
from src.utils import parse_story_xml

class NaiveRagStrategy:
    def __init__(self, story_file_path: str, llm_client: SafeLLMClient):
        print("Loading Embedding Model (this may take a moment)...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        print(f"Parsing story from {story_file_path}...")
        self.chunks = parse_story_xml(story_file_path)
        
        # Pre-calculate embeddings for all chunks (messages)
        print("Embedding story chunks...")
        self.chunk_texts = [chunk['full_text'] for chunk in self.chunks]
        self.chunk_embeddings = self.embedder.encode(self.chunk_texts)
        
        self.llm_client = llm_client

    def ask(self, question: str) -> str:
        """
        Main entry point: 
        1. Retrieve relevant chunks.
        2. Construct context respecting the character limit.
        3. Query LLM.
        """
        # 1. Embed the Question
        question_embedding = self.embedder.encode(question)
        
        # 2. Calculate Cosine Similarity
        scores = np.dot(self.chunk_embeddings, question_embedding)
        
        # 3. Sort chunks by highest score
        top_k_indices = np.argsort(scores)[::-1]
        
        # 4. Construct Context respecting the 3000 char limit
        relevant_context = self._build_safe_context(top_k_indices, question)
        
        # 5. Call LLM (context is auto-truncated if needed)
        return self.llm_client.generate_answer(relevant_context, question)

    def _build_safe_context(self, sorted_indices: np.ndarray, question: str) -> str:
        """
        Iterates through sorted chunks and adds them to context 
        UNTIL adding another one would break the SafeLLMClient limit.
        """
        # Estimate overhead (System prompt + structure chars in SafeLLMClient)
        # System instruction (~200 chars) + "Context:\n" + "\n\nQuestion: " (~20 chars)
        # We must be conservative to account for the actual prompt structure.
        system_overhead = 250 
        current_used_chars = system_overhead + len(question)
        max_limit = SafeLLMClient.MAX_CHAR_LIMIT
        
        selected_chunks = []
        
        for idx in sorted_indices:
            chunk_text = self.chunks[idx]['full_text']
            chunk_len = len(chunk_text) + 2  # +2 for newlines
            
            if current_used_chars + chunk_len < max_limit:
                selected_chunks.append(chunk_text)
                current_used_chars += chunk_len
            else:
                # we can't fit the next best chunk, stop.
                break
        
        return "\n\n".join(selected_chunks)