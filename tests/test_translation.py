import pytest
from unittest.mock import patch, MagicMock
from backend.translation.local_translator import DeepTranslatorService
from backend.core.exceptions import TranslationError

@patch('backend.translation.local_translator.detect')
def test_detect_language_success(mock_detect):
    mock_detect.return_value = 'hi'
    
    service = DeepTranslatorService()
    lang = service.detect_language("नमस्ते")
    
    assert lang == 'hi'
    mock_detect.assert_called_once_with("नमस्ते")

@patch('backend.translation.local_translator.detect')
def test_detect_language_fallback(mock_detect):
    # If detection raises an exception, it should fallback to 'en'
    mock_detect.side_effect = Exception("Langdetect error")
    
    service = DeepTranslatorService()
    lang = service.detect_language("Some weird string 1234")
    
    assert lang == 'en'

@patch('backend.translation.local_translator.GoogleTranslator')
def test_translate_success(mock_gt):
    mock_translator_instance = MagicMock()
    mock_translator_instance.translate.return_value = "Hello"
    mock_gt.return_value = mock_translator_instance
    
    service = DeepTranslatorService()
    result = service.translate("नमस्ते", source_lang='hi', target_lang='en')
    
    assert result == "Hello"
    mock_gt.assert_called_once_with(source='hi', target='en')
    mock_translator_instance.translate.assert_called_once_with("नमस्ते")

def test_translate_same_language():
    service = DeepTranslatorService()
    result = service.translate("Hello", source_lang='en', target_lang='en')
    
    assert result == "Hello" # Should short-circuit and not call the API
