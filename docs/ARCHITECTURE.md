# Zenith AgriBot - Architecture Design

## Overview
Zenith AgriBot is an enterprise-grade, multilingual Retrieval-Augmented Generation (RAG) chatbot designed to run locally using open-source models, avoiding reliance on paid external APIs.

## Architecture Diagram

```mermaid
flowchart TD
    %% User Input
    User((User)) -->|Sends Query| LangDetect[Language Detection]
    
    %% Multilingual Pre-processing
    LangDetect -->|Detects Non-English| PreTranslate[Translation to EN]
    LangDetect -->|Detects English| Retriever
    PreTranslate --> Retriever
    
    %% Retrieval Pipeline
    subgraph RAG Pipeline
        Retriever[Retriever Engine] -->|Embeds Query| EmbeddingModel[Sentence Transformers]
        EmbeddingModel -->|Vector Search| FAISS[(FAISS Vector Store)]
        FAISS -->|Returns top-k Chunks| Retriever
        Retriever --> PromptBuilder[Prompt Builder]
    end
    
    %% Context Enhancement (Future)
    subgraph Future Context
        CNN[CNN Disease Prediction Model] -.->|Predicted Disease Context| PromptBuilder
    end
    
    %% Generation
    PromptBuilder -->|Injects Context + Query| LocalLLM[Local LLM / Ollama]
    LocalLLM -->|Generates English Response| PostTranslate[Translation to Original Lang]
    
    %% Output
    PostTranslate -->|Returns Response| User
```

## Data Flow Diagram

```mermaid
sequenceDiagram
    participant User
    participant API as FastAPI Router
    participant Engine as Chat Engine
    participant Trans as Translation Module
    participant Memory as Conversation Memory
    participant Retr as Retrieval Engine
    participant VS as Vector Store
    participant Prompt as Prompt Builder
    participant LLM as Local LLM
    
    User->>API: POST /api/chat {query, lang}
    API->>Engine: process_chat()
    
    Engine->>Memory: Fetch History
    
    Engine->>Trans: Detect & Translate to EN
    Trans-->>Engine: EN Query
    
    Engine->>Retr: Retrieve Context (EN Query)
    Retr->>VS: search(embedded_query)
    VS-->>Retr: top_k Chunks
    Retr-->>Engine: SearchResults
    
    Engine->>Prompt: build_prompt(EN Query, Context, History)
    Prompt-->>Engine: Final Prompt String
    
    Engine->>LLM: generate(Prompt)
    LLM-->>Engine: EN Response String
    
    Engine->>Trans: Translate to original lang
    Trans-->>Engine: Final Localized Response
    
    Engine->>Memory: Save Q & A
    
    Engine-->>API: ChatResponse
    API-->>User: JSON Output
```

## Core Principles
1. **Clean Architecture**: Separation of concerns. The API layer knows nothing about how FAISS works.
2. **SOLID**: Abstract base classes define behavior; specific implementations handle exact technologies.
3. **Local First**: Prioritizing offline, secure inference to respect data privacy and bandwidth constraints.
