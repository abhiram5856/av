from uuid import UUID, uuid4
import time
from typing import Optional

from backend.core.interfaces import BaseTranslator, BaseRetriever, BasePromptBuilder, BaseLLM, BaseMemory
from backend.models.domain import UserQuery, ChatResponse, ChatMessage
from backend.logging.logger import setup_logger

logger = setup_logger("zenith.chat_engine")

class ZenithChatEngine:
    """
    Orchestrates the entire RAG and Chat pipeline.
    """
    def __init__(
        self,
        translator: BaseTranslator,
        retriever: BaseRetriever,
        prompt_builder: BasePromptBuilder,
        llm: BaseLLM,
        memory: BaseMemory
    ):
        self.translator = translator
        self.retriever = retriever
        self.prompt_builder = prompt_builder
        self.llm = llm
        self.memory = memory

    def process_chat(self, query: UserQuery) -> ChatResponse:
        start_time = time.time()
        logger.info(f"Processing new chat query...")
        
        session_id = str(query.conversation_id) if query.conversation_id else str(uuid4())
        
        # 1. Detect and Translate
        original_lang = query.language or self.translator.detect_language(query.query)
        english_query = self.translator.translate(query.query, source_lang=original_lang, target_lang="en")
        
        # 2. Retrieve Context
        try:
            context_results = self.retriever.retrieve(english_query)
        except Exception as e:
            logger.warning(f"Retrieval failed or empty: {e}")
            context_results = []
            
        # 3. Fetch History
        history = self.memory.get_history(session_id, limit=5)
        
        # 4. Build Prompt
        prompt = self.prompt_builder.build_prompt(
            query=english_query,
            context=context_results,
            chat_history=history,
            disease_context=query.disease_context
        )
        
        # 5. Generate LLM Response
        english_response = self.llm.generate(prompt)
        
        # 6. Translate Back
        final_response = self.translator.translate(english_response, source_lang="en", target_lang=original_lang)
        
        # 7. Save to Memory
        user_msg = ChatMessage(role="user", content=query.query, language=original_lang)
        bot_msg = ChatMessage(role="assistant", content=final_response, language=original_lang)
        self.memory.add_message(session_id, user_msg)
        self.memory.add_message(session_id, bot_msg)
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        logger.info(f"Finished processing in {processing_time_ms}ms")
        
        return ChatResponse(
            response=final_response,
            language=original_lang,
            source_chunks=context_results,
            processing_time_ms=processing_time_ms
        )
