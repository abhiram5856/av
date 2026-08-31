# Kissan AI: Deployment, Integration, and RAG Guide

This guide provides the necessary steps to deploy the frontend, correctly integrate it with the backend, and implement the RAG (Retrieval-Augmented Generation) system for the AI assistant.

---

## 1. Frontend Deployment Guide (Next.js)

Since the frontend is built using Next.js (as noted in the docs), the easiest and most performant way to deploy it is via **Vercel**.

### Deployment Steps (Vercel):
1. **Push to GitHub**: Ensure all your frontend code is pushed to a GitHub repository.
2. **Create a Vercel Account**: Sign in to [vercel.com](https://vercel.com) using your GitHub account.
3. **Import the Project**: Click **Add New... > Project** and select your `kissan_ai` repository.
4. **Configure Project Settings**:
   - **Framework Preset**: Select `Next.js`.
   - **Root Directory**: If your frontend is inside a specific folder (e.g., `frontend/`), select it.
   - **Environment Variables**: Add your backend URL so the frontend knows where to send API requests:
     - `NEXT_PUBLIC_API_URL` = `https://your-backend-url.onrender.com` (or wherever your backend is hosted).
5. **Deploy**: Click **Deploy**. Vercel will automatically build the Next.js app and provide you with a live URL.

*Note: Whenever you push new code to the `main` branch, Vercel will automatically redeploy the frontend.*

---

## 2. Frontend-Backend Integration Summary

Your existing `FRONTEND_DOCS.md` and `BACKEND_DOCS.md` heavily cover the API contract. Here is a summary of how the integration must be handled:

### 1. CORS (Cross-Origin Resource Sharing)
For the deployed frontend (e.g., `https://kissan-ai.vercel.app`) to communicate with the backend API, the backend (FastAPI) MUST have CORS properly configured.

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    # Add your deployed frontend URL here:
    allow_origins=["http://localhost:3000", "https://kissan-ai.vercel.app"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 2. The API Fetch Utility (Frontend)
Your frontend should use a single utility to handle all requests. This ensures the `Authorization: Bearer <token>` is always sent.

```typescript
// frontend/src/utils/api.ts
const BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export const apiFetch = async (endpoint: string, options: RequestInit = {}) => {
  const token = localStorage.getItem("access_token");
  
  const headers = {
    ...options.headers,
    "Authorization": token ? `Bearer ${token}` : "",
  };

  const response = await fetch(`${BASE_URL}${endpoint}`, { ...options, headers });
  
  if (response.status === 401) {
    // Redirect to login if token is expired
    window.location.href = "/login";
  }
  
  return response.json();
};
```

---

## 3. RAG (LangChain) Implementation Guide

To build an intelligent assistant that can answer questions based on specific agricultural data (like localized farming guides or PDF manuals), you need a **Retrieval-Augmented Generation (RAG)** pipeline.

### Prerequisites (Backend)
Install the required LangChain and Vector Database libraries:
```bash
pip install langchain langchain-openai chromadb sentence-transformers pypdf
```

### Step 1: Ingesting Data (Vector Database)
You need to convert your agricultural PDFs/texts into vector embeddings so the AI can search them. You can run this script once to populate the database.

```python
# scripts/ingest_data.py
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# 1. Load the PDF manual
loader = PyPDFLoader("data/agricultural_manual.pdf")
documents = loader.load()

# 2. Split into smaller chunks (for better retrieval)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)

# 3. Create Embeddings & Store in ChromaDB
# We use a free HuggingFace model for embeddings to save costs
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
print("Knowledge base created successfully!")
```

### Step 2: The LangChain Query Engine
Now, expose an endpoint in your FastAPI backend that uses LangChain to answer user queries based on the stored vectors.

```python
# backend/api/rag_routes.py
from fastapi import APIRouter, Depends
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from pydantic import BaseModel

router = APIRouter()

# Define the request body
class ChatRequest(BaseModel):
    query: str
    disease_context: str = None # Optional: The disease detected by CausalAg-Net

# Load the vector database into memory
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Setup LLM (You need an OPENAI_API_KEY)
llm = ChatOpenAI(temperature=0.2, model_name="gpt-3.5-turbo")

# Custom Prompt template forcing the AI to act as an agricultural expert
prompt_template = """
You are Kissan AI, an expert agricultural assistant. Use the provided context to answer the farmer's question.
If the farmer is asking about a specific disease, tailor your advice to treat that disease.
If you don't know the answer based on the context, state that you don't know.

Context: {context}
Farmer's Question: {question}

Answer in a helpful and instructional tone:
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Create the LangChain Retrieval System
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_db.as_retriever(search_kwargs={"k": 3}), # Fetch top 3 relevant chunks
    chain_type_kwargs={"prompt": PROMPT}
)

@router.post("/chat")
async def chat_with_assistant(request: ChatRequest):
    # If the app detected a disease, append it to the query for better context
    final_query = request.query
    if request.disease_context:
        final_query = f"I am dealing with {request.disease_context}. {request.query}"
        
    # Get answer from LangChain
    result = qa_chain({"query": final_query})
    
    return {
        "status": "success",
        "answer": result["result"]
    }
```

### Step 3: Frontend Usage
On the frontend, when the farmer uses the chat feature:

```javascript
// Example Frontend call
const askQuestion = async (userQuestion) => {
    const response = await apiFetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            query: userQuestion,
            disease_context: "Tomato_Late_Blight" // Pass the latest detected disease
        })
    });
    
    console.log("AI Answer:", response.answer);
};
```

This ensures the Chatbot isn't just a generic AI, but an expert RAG system deeply integrated with the `CausalAg-Net` disease detection model.
