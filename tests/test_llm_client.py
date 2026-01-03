import pytest
import sys
import os
from unittest.mock import patch, MagicMock
from src.llm_client import SafeLLMClient, PromptTooLongError

@pytest.fixture
def client():
    """
    Fixture to provide a client instance.
    We mock the environment variable so it doesn't complain about missing keys during tests.
    """
    with patch.dict(os.environ, {"OPENAI_API_KEY": "fake-test-key"}):
        return SafeLLMClient()

@patch("src.llm_client.OpenAI")
def test_prompt_under_limit_success(mock_openai_class):
    """
    Test that a prompt under 3000 chars passes logic and 'calls' the API.
    We mock the OpenAI client so no real network request happens.
    """
    # 1. Setup the Mock Response
    # We need to mimic: client.chat.completions.create().choices[0].message.content
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Mocked Answer"
    
    # Attach this mock response to the create method
    mock_instance = mock_openai_class.return_value
    mock_instance.chat.completions.create.return_value = mock_response

    # 2. Create the client INSIDE the patch context so it uses the mock
    with patch.dict(os.environ, {"OPENAI_API_KEY": "fake-test-key"}):
        client = SafeLLMClient()

    # 3. Run the method
    context = "Short context."
    question = "Short question?"
    result = client.generate_answer(context, question)

    # 4. Assertions
    assert result == "Mocked Answer"
    
    # Verify we actually called the API (conceptually)
    mock_instance.chat.completions.create.assert_called_once()

def test_prompt_exceeds_limit_raises_error(client):
    """
    Test that a prompt over 3000 chars raises PromptTooLongError.
    This logic happens BEFORE the API call, so we don't strictly need to mock the API here,
    but it's safe to rely on the client fixture.
    """
    # 2500 chars context + 501 chars question = 3001 chars (over limit)
    long_context = "a" * 2500
    long_question = "b" * 501
    
    with pytest.raises(PromptTooLongError) as excinfo:
        client.generate_answer(long_context, long_question)
    
    assert "exceeds limit" in str(excinfo.value)