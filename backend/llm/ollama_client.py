import requests
from backend.core.interfaces import BaseLLM
from backend.core.exceptions import LLMError
from backend.config.settings import settings
from backend.logging.logger import llm_logger

class OllamaClient(BaseLLM):
    """
    Client to interact with a local Ollama instance running an open-source LLM.
    """

    def __init__(self, model_name: str = None, host: str = "http://localhost:11434"):
        self.model_name = model_name or settings.LLM_MODEL_NAME
        self.host = host
        self.api_url = f"{self.host}/api/generate"
        llm_logger.info(f"Initialized OllamaClient targeting {self.api_url} with model {self.model_name}")

    def generate(self, prompt: str, **kwargs) -> str:
        temperature = kwargs.get("temperature", settings.LLM_TEMPERATURE)
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }

        try:
            llm_logger.debug(f"Sending prompt to Ollama (len={len(prompt)}).")
            response = requests.post(self.api_url, json=payload, timeout=120)
            response.raise_for_status()
            
            data = response.json()
            generated_text = data.get("response", "")
            
            llm_logger.debug(f"Received response from Ollama (len={len(generated_text)}).")
            return generated_text.strip()
            
        except requests.exceptions.ConnectionError:
            error_msg = f"Failed to connect to Ollama at {self.host}. Is Ollama running?"
            llm_logger.error(error_msg)
            raise LLMError(error_msg)
        except requests.exceptions.RequestException as e:
            error_msg = f"HTTP request to Ollama failed: {str(e)}"
            llm_logger.error(error_msg)
            raise LLMError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error during LLM generation: {str(e)}"
            llm_logger.error(error_msg)
            raise LLMError(error_msg)
