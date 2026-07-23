from fastapi import APIRouter, Depends
from backend.models.domain import UserQuery, ChatResponse

from backend.chat_engine.engine import ZenithChatEngine
from backend.translation.local_translator import DeepTranslatorService
from backend.embeddings.local_embeddings import SentenceTransformerModel
from backend.vectorstore.faiss_store import FaissVectorStore
from backend.retrieval.vector_retriever import VectorRetriever
from backend.prompt_builder.rag_prompt import StandardPromptBuilder
from backend.llm.ollama_client import OllamaClient
from backend.memory.in_memory_store import InMemorySessionStore

router = APIRouter()

# Global instances for the scaffold (In production, use dependency injection frameworks)
translator = DeepTranslatorService()
vector_store = FaissVectorStore()
embedder = SentenceTransformerModel()
retriever = VectorRetriever(embedder, vector_store)
prompt_builder = StandardPromptBuilder()
llm = OllamaClient()
memory = InMemorySessionStore()

chat_engine = ZenithChatEngine(
    translator=translator,
    retriever=retriever,
    prompt_builder=prompt_builder,
    llm=llm,
    memory=memory
)

def get_engine():
    return chat_engine

@router.post("/", response_model=ChatResponse)
async def process_chat(query: UserQuery, engine: ZenithChatEngine = Depends(get_engine)):
    """
    Process a chat message. 
    Flow: Lang Detect -> Translate -> Retrieve -> Prompt -> LLM -> Translate -> Response.
    """
    response = engine.process_chat(query)
    return response
