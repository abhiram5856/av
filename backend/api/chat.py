"""
AgriVision AI — Genuine RAG Chatbot Endpoint
=============================================
Replaces the previous keyword-matching mock (which was NOT RAG).

Pipeline:
  1. Receive farmer question + optional disease/env context
  2. Embed query using LocalEmbedder (all-MiniLM-L6-v2)
  3. Retrieve top-K semantically similar agronomic knowledge chunks from FAISS
  4. Build structured prompt: question + retrieved chunks + disease context
  5. Generate grounded response via GroqClient (llama3-70b-8192)
  6. Return response WITH retrieved chunk IDs and similarity scores for audit

Safety rules enforced:
  - If model confidence < 60%: response explicitly recommends agronomist
  - If GROQ_API_KEY missing: honest fallback (no fake RAG)
  - If FAISS empty: honest fallback with retrieval status in response
  - Retrieved chunk IDs and scores always included in debug_retrieval field
"""
import os
import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from backend.embeddings.local_embedder import LocalEmbedder
from backend.vectorstore.faiss_store import FaissVectorStore
from backend.retrieval.vector_retriever import VectorRetriever
from backend.llm.groq_client import GroqClient
from backend.logging.logger import api_logger
from backend.core.exceptions import LLMError

router = APIRouter()

# ── Module-level singletons (lazy-initialised) ─────────────────────────────
_embedder: Optional[LocalEmbedder] = None
_vector_store: Optional[FaissVectorStore] = None
_retriever: Optional[VectorRetriever] = None
_llm: Optional[GroqClient] = None


def _get_rag_components():
    """Lazy-initialise RAG components once on first call."""
    global _embedder, _vector_store, _retriever, _llm
    if _retriever is None:
        api_logger.info("Initialising RAG pipeline components...")
        _embedder     = LocalEmbedder()
        _vector_store = FaissVectorStore()
        _retriever    = VectorRetriever(_embedder, _vector_store)
        _llm          = GroqClient()
        api_logger.info(
            f"RAG pipeline ready. FAISS index size: {_vector_store.index.ntotal} vectors."
        )
    return _retriever, _llm, _vector_store


# ── Request / Response schemas ────────────────────────────────────────────
class ChatRequest(BaseModel):
    query: str
    language: Optional[str] = "en"
    disease_context: Optional[str] = None       # e.g. "tomato_late_blight"
    confidence: Optional[float] = None          # model confidence 0-1
    crop: Optional[str] = None
    growth_stage: Optional[str] = None
    severity_indicator: Optional[str] = None
    temperature: Optional[float] = None
    humidity: Optional[float] = None


class RetrievedChunkDebug(BaseModel):
    chunk_id: str
    similarity_score: float
    text_preview: str


class ChatResponse(BaseModel):
    response: str
    language: str
    retrieval_executed: bool
    retrieved_chunks_count: int
    low_confidence_warning: bool
    recommends_agronomist: bool
    debug_retrieval: List[RetrievedChunkDebug]
    processing_time_ms: int


CONFIDENCE_THRESHOLD = 0.60
LOW_CONF_ADVISORY = (
    "\n\n---\n"
    "**Important:** The AI disease classifier returned a confidence score below 60%. "
    "This recommendation is based on a potentially uncertain diagnosis. "
    "Please consult a qualified agronomist or the nearest Krishi Vigyan Kendra (KVK) "
    "before applying any treatment."
)


def _build_prompt(
    query: str,
    retrieved_chunks: list,
    disease_context: Optional[str],
    confidence: Optional[float],
    crop: Optional[str],
    growth_stage: Optional[str],
    severity_indicator: Optional[str],
    temperature: Optional[float],
    humidity: Optional[float],
) -> str:
    """
    Build a grounded RAG prompt. The LLM must answer based on retrieved context
    and must not invent information not present in the chunks.
    """
    # Format retrieved knowledge
    knowledge_block = ""
    if retrieved_chunks:
        knowledge_block = "\n\n--- RETRIEVED AGRONOMIC KNOWLEDGE ---\n"
        for i, result in enumerate(retrieved_chunks, 1):
            text = result.chunk.metadata.get("text", result.chunk.text)
            knowledge_block += f"\n[Source {i}]: {text}\n"
        knowledge_block += "\n--- END OF RETRIEVED KNOWLEDGE ---\n"
    else:
        knowledge_block = "\n\n[No relevant knowledge chunks were retrieved for this query.]\n"

    # Build context block
    context_parts = []
    if disease_context:
        conf_str = f" (confidence: {confidence*100:.1f}%)" if confidence else ""
        context_parts.append(f"Diagnosed disease: {disease_context}{conf_str}")
    if crop:
        context_parts.append(f"Crop: {crop}")
    if growth_stage:
        context_parts.append(f"Growth Stage: {growth_stage}")
    if severity_indicator:
        context_parts.append(f"Estimated Severity: {severity_indicator}")
    if temperature is not None:
        context_parts.append(f"Field temperature: {temperature}°C")
    if humidity is not None:
        context_parts.append(f"Field humidity: {humidity*100:.0f}%")

    context_block = ""
    if context_parts:
        context_block = "\n\n--- FIELD CONTEXT ---\n" + "\n".join(context_parts) + "\n"

    prompt = f"""You are AgriBot, an agricultural assistant for Telangana farmers. 
Your role is to provide practical, safe, and evidence-based advice about crop diseases and farm management.

INSTRUCTIONS:
- Answer ONLY based on the retrieved agronomic knowledge provided below.
- If the retrieved knowledge does not cover the question, say so clearly.
- Do NOT invent fungicide names, dosages, or crop science facts.
- Keep advice practical for smallholder farmers in Telangana.
- Use simple, clear language. If helpful, use bullet points.
- Do not claim certainty beyond what the retrieved knowledge supports.
{context_block}{knowledge_block}

FARMER QUESTION: {query}

ANSWER:"""
    return prompt


@router.post("/", response_model=ChatResponse)
async def process_chat(request: ChatRequest):
    t_start = time.perf_counter()

    low_confidence = (
        request.confidence is not None
        and request.confidence < CONFIDENCE_THRESHOLD
    )
    recommends_agronomist = low_confidence
    retrieval_executed     = False
    retrieved_chunks       = []
    debug_chunks: List[RetrievedChunkDebug] = []
    response_text = ""

    try:
        retriever, llm, vector_store = _get_rag_components()

        # ── Step 1: Retrieve relevant chunks ─────────────────────────────────
        if vector_store.index.ntotal == 0:
            api_logger.warning(
                "FAISS index is empty. RAG retrieval skipped. "
                "Run scripts/ingest_knowledge_base.py to populate."
            )
            retrieval_executed = False
        else:
            retrieved_chunks   = retriever.retrieve(request.query, top_k=3)
            retrieval_executed = True
            api_logger.info(
                f"RAG retrieval: {len(retrieved_chunks)} chunks retrieved for query: "
                f"'{request.query[:60]}...'"
            )

            # Build debug output with chunk IDs and scores (for audit trail)
            for result in retrieved_chunks:
                debug_chunks.append(RetrievedChunkDebug(
                    chunk_id=str(result.chunk.id),
                    similarity_score=round(float(result.score), 4),
                    text_preview=result.chunk.metadata.get("text", result.chunk.text)[:120],
                ))

        # ── Step 2: Check LLM availability ───────────────────────────────────
        groq_key = os.environ.get("GROQ_API_KEY")
        if not groq_key:
            # Honest fallback — do NOT pretend to use RAG
            api_logger.warning("GROQ_API_KEY not set. Returning honest fallback.")
            chunk_info = ""
            if debug_chunks:
                chunk_info = (
                    f"\n\nI retrieved {len(debug_chunks)} relevant knowledge entries "
                    f"from the agricultural database, but cannot generate a synthesised "
                    f"response without a configured LLM provider. "
                    f"Please set the GROQ_API_KEY environment variable."
                )
            response_text = (
                f"The agricultural knowledge retrieval system is operational"
                f"{' and found relevant information' if retrieval_executed else ''}. "
                f"However, the language model (LLM) is not configured. "
                f"Please set GROQ_API_KEY to enable full AI-generated responses."
                f"{chunk_info}"
            )
            if low_confidence:
                response_text += LOW_CONF_ADVISORY
                recommends_agronomist = True

            elapsed = int((time.perf_counter() - t_start) * 1000)
            return ChatResponse(
                response=response_text,
                language=request.language or "en",
                retrieval_executed=retrieval_executed,
                retrieved_chunks_count=len(debug_chunks),
                low_confidence_warning=low_confidence,
                recommends_agronomist=recommends_agronomist,
                debug_retrieval=debug_chunks,
                processing_time_ms=elapsed,
            )

        # ── Step 3: Build grounded prompt ────────────────────────────────────
        prompt = _build_prompt(
            query=request.query,
            retrieved_chunks=retrieved_chunks,
            disease_context=request.disease_context,
            confidence=request.confidence,
            crop=request.crop,
            growth_stage=request.growth_stage,
            severity_indicator=request.severity_indicator,
            temperature=request.temperature,
            humidity=request.humidity,
        )

        # ── Step 4: Generate response via LLM ────────────────────────────────
        response_text = llm.generate(prompt, temperature=0.3)

        # ── Step 5: Append low-confidence advisory if needed ─────────────────
        if low_confidence:
            response_text += LOW_CONF_ADVISORY
            recommends_agronomist = True

    except LLMError as e:
        api_logger.error(f"LLM generation failed: {e}")
        response_text = (
            "I was unable to generate a response at this time due to a connection issue "
            "with the language model. Please try again shortly. "
            "If this persists, consult your nearest Krishi Vigyan Kendra (KVK)."
        )
        recommends_agronomist = True
    except Exception as e:
        api_logger.error(f"RAG chatbot error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat service error: {str(e)}")

    elapsed = int((time.perf_counter() - t_start) * 1000)
    api_logger.info(
        f"RAG chat complete | retrieval={retrieval_executed} | "
        f"chunks={len(debug_chunks)} | llm_used={bool(groq_key)} | "
        f"latency={elapsed}ms"
    )

    return ChatResponse(
        response=response_text,
        language=request.language or "en",
        retrieval_executed=retrieval_executed,
        retrieved_chunks_count=len(debug_chunks),
        low_confidence_warning=low_confidence,
        recommends_agronomist=recommends_agronomist,
        debug_retrieval=debug_chunks,
        processing_time_ms=elapsed,
    )
