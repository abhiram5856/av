# CausalAg-Net: RAG & LangChain Guide

This document outlines the architecture for integrating an LLM (Large Language Model) via LangChain to generate human-readable agricultural advice based on the CausalAg-Net diagnostic output.

## 1. Why RAG (Retrieval-Augmented Generation)?
While our PyTorch model can identify "Tomato Late Blight" and our symbolic engine can verify that humidity is high, the final step is giving the farmer **actionable advice**. 
By using RAG, we can connect our LLM to real agricultural databases (like ICAR guidelines) so the AI doesn't hallucinate pesticide recommendations.

## 2. The LangChain Workflow

When a diagnosis is completed in `api/diagnose.py`, the backend will trigger a LangChain process before returning the final JSON.

### Step A: The Vector Database
You must index agricultural PDFs and documents into a Vector Database (e.g., ChromaDB, Pinecone). 
```python
# Pseudo-code for document ingestion (run once)
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

loader = PyPDFLoader("data/icar_tomato_guidelines.pdf")
docs = loader.load_and_split()
vectorstore = Chroma.from_documents(docs, OpenAIEmbeddings())
```

### Step B: The Prompt Context
Inside the `/diagnose` endpoint, you construct a prompt containing the hard evidence from your PyTorch model.

```python
# Pseudo-code inside your FastAPI endpoint
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(temperature=0.2)
qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())

prompt = f"""
You are an expert agronomist. 
Our vision model has detected Tomato Late Blight with a severity of {severity_score}%.
The logic engine confirms high humidity ({humidity_float}) is driving the disease.
Based on the retrieved ICAR documents, what is the immediate chemical and cultural control method the farmer should apply today?
"""

ai_explanation = qa_chain.run(prompt)
```

## 3. Connecting to the Frontend
Once LangChain returns the `ai_explanation` string, simply attach it to the JSON response that is sent back to the Next.js frontend:

```json
{
  "status": "success",
  "hypothesis": "Tomato_Late_Blight",
  ...
  "ai_explanation": "According to ICAR guidelines, apply Mancozeb 75 WP at 2.5g/L immediately. Prune lower leaves to improve airflow..."
}
```
The frontend is already configured to read `results.ai_explanation` and display it in the bottom summary box!
