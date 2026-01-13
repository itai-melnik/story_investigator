import pytest
import sys
import os
from unittest.mock import patch, MagicMock
from src.llm_client import SafeLLMClient

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

@patch("src.llm_client.OpenAI")
def test_long_context_is_truncated(mock_openai_class):
    """
    Test that a long context is automatically truncated to fit within the limit.
    The method should NOT raise an error, but truncate and call the API.
    """
    # Setup the Mock Response
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Mocked Answer"
    mock_instance = mock_openai_class.return_value
    mock_instance.chat.completions.create.return_value = mock_response

    with patch.dict(os.environ, {"OPENAI_API_KEY": "fake-test-key"}):
        client = SafeLLMClient()

    # Very long context that exceeds the limit
    long_context = "a" * 5000
    question = "Short question?"
    
    # Should NOT raise an error - context gets truncated
    result = client.generate_answer(long_context, question)
    
    # Verify the API was still called
    assert result == "Mocked Answer"
    mock_instance.chat.completions.create.assert_called_once()
    
    # Verify the context was truncated (contains "..." before the question)
    call_args = mock_instance.chat.completions.create.call_args
    user_message = call_args.kwargs["messages"][1]["content"]
    assert "...\n\nQuestion:" in user_message