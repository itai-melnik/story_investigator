from openai import OpenAI


class SafeLLMClient:
    """Client for making safe LLM API calls with automatic context truncation (3000 char limit)."""
    MAX_CHAR_LIMIT = 3000

    def __init__(self, api_key=None):
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)

    def generate_answer(self, context: str, question: str) -> str:
        """
        Constructs the prompt and calls OpenAI. 
        Automatically truncates context if it exceeds the character limit.
        """
        
        # 1. Define the System Persona
        system_instruction = (
            "You are AI Investigator 1.0. Answer the question based ONLY on the story chunks provided.\n"
            "You must strictly follow this format:\n"
            "[Answer]. Here is why:\n"
            "[Exact quote from the text or reasoning why]\n"
            "If answer is not in the text or you cannot find a conclusive answer, state that you don't know and explain why."
        )

        # 2. Calculate available space for context
        # Overhead: system_instruction  + "Context:\n" + "\n\nQuestion: "  + question
        overhead = len(system_instruction) + len("Context:\n") + len("\n\nQuestion: ") + len(question)
        max_context_length = self.MAX_CHAR_LIMIT - overhead
        
        # 3. Truncate context if needed
        if len(context) > max_context_length:
            # Leave room for "..." indicator
            context = context[:max_context_length - 3] + "..."
        
        # 4. Build user content
        user_content = f"Context:\n{context}\n\nQuestion: {question}"

        # 4. Call OpenAI API
        return self._call_llm_api(system_instruction, user_content)

    def _call_llm_api(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model="gpt-5-mini",  # TODO: try with gpt-5-nano (to see how effective the graphRAG is)
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error calling OpenAI: {str(e)}"