import pytest
from unittest.mock import MagicMock
from backend.chat_engine.engine import ZenithChatEngine
from backend.models.domain import UserQuery, ChatMessage

def test_chat_engine_flow():
    # Mock all dependencies
    mock_translator = MagicMock()
    mock_retriever = MagicMock()
    mock_prompt_builder = MagicMock()
    mock_llm = MagicMock()
    mock_memory = MagicMock()
    
    # Setup mock returns
    mock_translator.detect_language.return_value = "te"
    mock_translator.translate.side_effect = ["Translate to English", "Translate to Telugu"]
    
    mock_retriever.retrieve.return_value = ["MockSearchResult"]
    mock_memory.get_history.return_value = [ChatMessage(role="user", content="hello")]
    mock_prompt_builder.build_prompt.return_value = "Final Prompt String"
    mock_llm.generate.return_value = "LLM English Response"
    
    engine = ZenithChatEngine(
        translator=mock_translator,
        retriever=mock_retriever,
        prompt_builder=mock_prompt_builder,
        llm=mock_llm,
        memory=mock_memory
    )
    
    query = UserQuery(query="Hello in Telugu")
    response = engine.process_chat(query)
    
    # Assert data flow
    mock_translator.detect_language.assert_called_once_with("Hello in Telugu")
    mock_translator.translate.assert_any_call("Hello in Telugu", source_lang="te", target_lang="en")
    
    mock_retriever.retrieve.assert_called_once_with("Translate to English")
    mock_memory.get_history.assert_called_once()
    
    mock_prompt_builder.build_prompt.assert_called_once_with(
        query="Translate to English",
        context=["MockSearchResult"],
        chat_history=[ChatMessage(role="user", content="hello")],
        disease_context=None
    )
    
    mock_llm.generate.assert_called_once_with("Final Prompt String")
    mock_translator.translate.assert_any_call("LLM English Response", source_lang="en", target_lang="te")
    
    # Memory should save both messages
    assert mock_memory.add_message.call_count == 2
    
    assert response.response == "Translate to Telugu"
    assert response.language == "te"
    assert response.source_chunks == ["MockSearchResult"]
    assert response.processing_time_ms is not None
