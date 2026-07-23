from langdetect import detect, DetectorFactory
from deep_translator import GoogleTranslator
from backend.core.interfaces import BaseTranslator
from backend.core.exceptions import TranslationError
from backend.logging.logger import translate_logger

# Ensure consistent results from langdetect
DetectorFactory.seed = 0

class DeepTranslatorService(BaseTranslator):
    """
    Lightweight translation service for scaffolding using deep-translator.
    In a full production environment, this would be replaced with local MarianMT or NLLB models.
    """

    def detect_language(self, text: str) -> str:
        try:
            # langdetect returns ISO 639-1 codes (e.g. 'en', 'hi', 'te')
            lang_code = detect(text)
            translate_logger.debug(f"Detected language '{lang_code}' for text of length {len(text)}")
            return lang_code
        except Exception as e:
            translate_logger.error(f"Failed to detect language: {str(e)}")
            # Default to English if detection fails
            return "en"

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        if not text.strip() or source_lang == target_lang:
            return text

        translate_logger.info(f"Translating from {source_lang} to {target_lang}")
        try:
            # deep-translator uses 'auto' for source, but we explicitly pass it if known
            translator = GoogleTranslator(source=source_lang, target=target_lang)
            translated_text = translator.translate(text)
            return translated_text
        except Exception as e:
            translate_logger.error(f"Translation failed: {str(e)}")
            raise TranslationError(f"Failed to translate text: {str(e)}")
