import os
import requests
from backend.core.interfaces import BaseLLM
from backend.core.exceptions import LLMError
from backend.core_logging.logger import llm_logger

class GroqClient(BaseLLM):
    """
    Client to interact with Groq's high-speed LPU API.
    """
    def __init__(self, model_name: str = "llama3-70b-8192"):
        self.model_name = model_name
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.api_key = os.environ.get("GROQ_API_KEY")
        llm_logger.info(f"Initialized GroqClient targeting model {self.model_name}")

    def generate(self, prompt: str, **kwargs) -> str:
        if not self.api_key:
            # Fallback for development if no key is provided
            llm_logger.warning("GROQ_API_KEY not found. Returning mock response.")
            return "**No Groq API Key found!** Please set the `GROQ_API_KEY` environment variable. \n\n*This is a mock RAG response since I couldn't reach the API.*"
            
        temperature = kwargs.get("temperature", 0.7)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": 1024
        }

        try:
            llm_logger.debug(f"Sending prompt to Groq (len={len(prompt)}).")
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            generated_text = data['choices'][0]['message']['content']
            
            llm_logger.debug(f"Received response from Groq (len={len(generated_text)}).")
            return generated_text.strip()
            
        except Exception as e:
            error_msg = f"Failed to connect to Groq API: {str(e)}"
            llm_logger.error(error_msg)
            raise LLMError(error_msg)
