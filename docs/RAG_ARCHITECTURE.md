# AgriVision AI — RAG Architecture

**Status:** Implemented (2026-08-26)  
**Replaces:** Keyword-matching mock (`chat.py` v1)

---

## Architecture Overview

```
Farmer Question
       |
       v
ChatRequest (FastAPI POST /api/v1/chat/)
  - query: str
  - disease_context: str (optional, from diagnosis)
  - confidence: float (optional, 0-1)
  - crop: str (optional)
  - temperature, humidity (optional, from IoT/weather)
       |
       v
LocalEmbedder (all-MiniLM-L6-v2)
  → Encode query to 384-dim dense vector
       |
       v
FaissVectorStore (data/faiss_index)
  → IndexFlatL2 similarity search
  → Returns top-3 semantically similar chunks
  → Each result: chunk_id, similarity_score, text
       |
       v
Prompt Builder
  → Formats: question + retrieved chunks + field context
  → Instructs LLM to answer ONLY from retrieved knowledge
       |
       v
GroqClient (llama3-70b-8192)
  → Configurable via GROQ_API_KEY env var
  → Returns grounded response
       |
       v
ChatResponse
  - response: str
  - retrieval_executed: bool  ← AUDIT FIELD
  - retrieved_chunks_count: int  ← AUDIT FIELD
  - debug_retrieval: [{chunk_id, similarity_score, text_preview}]  ← AUDIT FIELD
  - low_confidence_warning: bool
  - recommends_agronomist: bool
```

---

## Knowledge Base

**Source:** `knowledge_base/agronomic_rules.json`  
**Coverage:** All 24 active disease classes  
**Content per entry:** visual symptoms, environmental conditions, Telangana relevance, treatment recommendations, agronomic logic rule  
**Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim)  
**Vector store:** FAISS IndexFlatL2 (`data/faiss_index`)  
**Vectors stored:** 24  

**Re-ingest command:** `python scripts/ingest_knowledge_base.py`

---

## Audit Validation

Every ChatResponse includes `debug_retrieval` — a list of the actual chunks retrieved, their FAISS similarity scores, and text previews. This field exists specifically so retrieval can be independently validated:

```json
{
  "retrieval_executed": true,
  "retrieved_chunks_count": 3,
  "debug_retrieval": [
    {
      "chunk_id": "uuid-xxxx",
      "similarity_score": 0.1423,
      "text_preview": "Crop Disease Rule: tomato_late_blight. If Tomato_Late_Blight..."
    }
  ]
}
```

---

## Safety Behaviours

| Condition | Behaviour |
|-----------|-----------|
| `GROQ_API_KEY` not set | Returns honest message — does NOT pretend to use LLM |
| FAISS index empty | Returns honest message — does NOT pretend retrieval occurred |
| `confidence < 60%` | Appends explicit agronomist recommendation advisory |
| LLM API error | Returns graceful error message with KVK referral |
| Retrieved chunks irrelevant | LLM instructed to say so; must not invent information |

---

## What Was Replaced

The previous `chat.py` (v1) contained this logic:

```python
if "paddy" in user_query or "rice" in user_query:
    reply = "For Paddy (Rice) in Telangana..."
elif "cotton" in user_query:
    reply = "For Cotton crops..."
```

This is keyword matching — **not RAG**. It was labelled "Mock process a chat message" in its own docstring. It has been fully replaced.
