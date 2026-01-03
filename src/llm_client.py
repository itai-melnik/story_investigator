# src/llm_client.py

class PromptTooLongError(Exception):
    """Custom exception raised when the LLM prompt exceeds the safe limit."""
    pass

class SafeLLMClient:
    """Client for making safe LLM API calls with prompt limit enforcement (3000 tokens)."""
    MAX_CHAR_LIMIT = 3000

    def __init__(self, api_key=None):
        self.api_key = api_key

    def generate_answer(self, context: str, question: str) -> str:
        """
        Constructs the prompt and sends it to the LLM.
        Raises PromptTooLongError if the constructed prompt is > 3000 chars.
        """
        
        # 1. Construct the prompt template
        # We need to count the *exact* characters sent to the model
        system_instruction = "AI Investigator 1.0. Answer the question based on the story chunks below."
        full_prompt = f"{system_instruction}\n\nContext:\n{context}\n\nQuestion: {question}"

        # 2 Check the constraint
        current_length = len(full_prompt)
        
        if current_length > self.MAX_CHAR_LIMIT:
            raise PromptTooLongError(
                f"Prompt length ({current_length}) exceeds limit of {self.MAX_CHAR_LIMIT} characters."
            )

        # 3. Call Actual LLM (Mocked for now)
        return self._call_llm_api(full_prompt)

    def _call_llm_api(self, prompt: str) -> str:
        """
        Placeholder for the actual API call (OpenAI, Gemini, etc.)
        """
        # TODO: Replace with actual API call later
        return "This is a mock response from the AI Investigator."