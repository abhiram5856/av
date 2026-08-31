# NOVA System Architecture Blueprint

This document details the architectural layers and data flow patterns of the **Neural Optimized Vision Assistant (NOVA)**.

---

## 🏗️ Structural Overview

NOVA is structured around a decoupled, local-first multi-modal processing pipeline.

```mermaid
flowchart TD
    %% User Input & Image Ingestion
    User((User / Farmer)) -->|Uploads Leaf Image + Context| API[FastAPI Gateway]
    
    %% API router layers
    subgraph Multi-Modal Processing
        API -->|1. Image Quality Assessment| IQA[IQA Engine]
        API -->|2. Crop Disease Classification| CNN[MobileNetV3 small Classifier]
        API -->|3. Feature Attribution| CAM[Grad-CAM Generator]
        API -->|4. Risk Scorer| Severity[Multimodal Concern Scorer]
    end
    
    %% AIContext composition
    subgraph Context Integration
        CNN -->|Disease & Confidence| Builder[AIContext Builder]
        CAM -->|Lesion Ratios & Heatmaps| Builder
        Severity -->|Final Concern Score & Level| Builder
        API -->|Weather Temp/Humidity/Soil pH| Builder
    end
    
    %% RAG Retrieval
    subgraph RAG Knowledge Ingestion
        Builder -->|Queries vector DB| VectorDB[(FAISS Vector Store)]
        VectorDB -->|Retrieves ICAR manual chunks| Builder
    end
    
    %% LLM Response Compile
    subgraph LLM Generation & Output
        Builder -->|Aggregated AIContext| Prompt[Prompt Builder]
        Prompt -->|Context-Aware Instruction| LLM[Groq Llama3 / Ollama]
        LLM -->|Agronomist Recommendations| Output[JSON / PDF Report]
    end
    
    %% Root Cause Placeholder
    subgraph Future Expansion
        Output -.->|Abstract Payload| RootCause[Root Cause Engine Placeholder]
    end
    
    Output -->|Download / Render| User
```

---

## 🔄 Multi-Modal Data Flow Sequence

1. **Leaf Image Ingestion**: The FastAPI router `/api/v1/diagnose/` accepts a multipart form containing the leaf image, current average temperature, humidity, soil pH, GPS coordinates, and user identifier.
2. **Vision Prediction & Feature Map Extraction**:
   * The image is routed to the MobileNetV3 small neural network.
   * Test-Time Augmentation (TTA) computes averaged class logits.
   * Grad-CAM intercepts the final convolutional layer (`features.7`) to produce heatmaps highlighting visual attention.
3. **Multimodal Concern Scoring**: The `MultimodalConcernScorer` computes a final score combining classification confidence, lesion area ratios, humidity, pH, temperature, and specific growth stages.
4. **Context Construction**: The `AIContextBuilder` aggregates vision outputs, GradCAM coordinates, weather parameters, and RAG knowledge.
5. **Report Compilation**: The PDF report endpoint `/api/v1/report` compiles a highly structured ReportLab document embedding the leaf image, GradCAM visual heatmap overlay, and RAG recommendations.
6. **Structured Production Logging**: Every request is monitored, logging execution latencies, device tags, and transaction identifiers.
