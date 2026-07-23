import pytest
import requests
from unittest.mock import patch, MagicMock
from backend.llm.ollama_client import OllamaClient
from backend.core.exceptions import LLMError

@patch("backend.llm.ollama_client.requests.post")
def test_ollama_generate_success(mock_post):
    mock_response = MagicMock()
    mock_response.json.return_value = {"response": "This is a mock LLM answer."}
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    client = OllamaClient(model_name="llama3")
    result = client.generate("Hello AgriBot")

    assert result == "This is a mock LLM answer."
    mock_post.assert_called_once()
    
    # Assert payload format
    call_kwargs = mock_post.call_args.kwargs
    assert "json" in call_kwargs
    payload = call_kwargs["json"]
    assert payload["model"] == "llama3"
    assert payload["prompt"] == "Hello AgriBot"
    assert payload["stream"] is False

@patch("backend.llm.ollama_client.requests.post")
def test_ollama_connection_error(mock_post):
    mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")
    
    client = OllamaClient()
    
    with pytest.raises(LLMError, match="Failed to connect to Ollama"):
        client.generate("Test prompt")
