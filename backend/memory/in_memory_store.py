from typing import List, Dict
from backend.core.interfaces import BaseMemory
from backend.models.domain import ChatMessage

class InMemorySessionStore(BaseMemory):
    """
    Lightweight, temporary in-memory store for chat history.
    Data is lost upon server restart.
    """
    def __init__(self):
        # Maps session_id (str) to a list of ChatMessages
        self._store: Dict[str, List[ChatMessage]] = {}

    def add_message(self, session_id: str, message: ChatMessage) -> None:
        if session_id not in self._store:
            self._store[session_id] = []
        self._store[session_id].append(message)

    def get_history(self, session_id: str, limit: int = 10) -> List[ChatMessage]:
        if session_id not in self._store:
            return []
        
        history = self._store[session_id]
        # Return the last 'limit' messages
        return history[-limit:] if limit > 0 else history
