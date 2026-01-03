import pytest
import sys
import os
from src.llm_client import SafeLLMClient, PromptTooLongError

@pytest.fixture
def client():
    """Fixture to provide a fresh client instance for each test."""
    return SafeLLMClient()

def test_prompt_under_limit_success(client):
    """Test that a prompt under 3000 chars passes successfully."""
    context = "Short context."
    question = "Short question?"
    
    # Simple assert checks
    result = client.generate_answer(context, question)
    assert isinstance(result, str)
    assert len(result) > 0

def test_prompt_exceeds_limit_raises_error(client):
    """Test that a prompt over 3000 chars raises PromptTooLongError."""
    # 2500 chars context + 600 chars question = 3100 chars (over limit)
    long_context = "a" * 2500
    long_question = "b" * 501
    
    # checking for exceptions
    with pytest.raises(PromptTooLongError) as excinfo:
        client.generate_answer(long_context, long_question)
    
    # Check the error message contains helpful info
    assert "exceeds limit" in str(excinfo.value)