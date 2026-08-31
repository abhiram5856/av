import requests
import asyncio
from backend.config.settings import settings
from backend.logging.logger import api_logger

class WeatherService:
    @staticmethod
    async def fetch_current_weather(lat: float, lon: float) -> dict:
        """
        Fetches live weather data from OpenWeatherMap API.
        Falls back to default values if the API key is missing or request fails.
        """
        default_weather = {
            "temperature": 25.0,
            "humidity": 60.0
        }

        api_key = settings.OPENWEATHER_API_KEY
        if not api_key:
            api_logger.warning("OPENWEATHER_API_KEY not set. Using default static weather values.")
            return default_weather

        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        
        try:
            # Use asyncio.to_thread to prevent blocking the async event loop with a sync request
            response = await asyncio.to_thread(requests.get, url, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            return {
                "temperature": float(data["main"]["temp"]),
                "humidity": float(data["main"]["humidity"])
            }
        except Exception as e:
            api_logger.error(f"Weather fetch failed: {e}. Falling back to default values.")
            return default_weather
