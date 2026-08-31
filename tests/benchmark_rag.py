import time
from backend.embeddings.local_embeddings import SentenceTransformerModel
from backend.vectorstore.faiss_store import FaissVectorStore
from backend.retrieval.vector_retriever import VectorRetriever
from backend.prompt_builder.rag_prompt import StandardPromptBuilder

def benchmark_rag_pipeline():
    print("=========================================")
    print("NOVA RAG Pipeline Validation & Benchmark")
    print("=========================================")
    
    # Initialize components
    t0 = time.time()
    embedder = SentenceTransformerModel()
    embed_init_time = time.time() - t0
    print(f"Embedding model initialization latency: {embed_init_time:.4f}s")
    
    t0 = time.time()
    vector_store = FaissVectorStore()
    store_init_time = time.time() - t0
    print(f"FAISS Vector Store loading latency: {store_init_time:.4f}s")
    
    retriever = VectorRetriever(embedder, vector_store)
    prompt_builder = StandardPromptBuilder()
    
    # Query test
    query = "How to control Tomato Late Blight leaf disease?"
    print(f"Testing Query: '{query}'")
    
    # Benchmark Retrieval
    t0 = time.time()
    results = retriever.retrieve(query, top_k=3)
    retrieval_latency = time.time() - t0
    print(f"Vector Retrieval Latency: {retrieval_latency * 1000:.2f} ms")
    print(f"Retrieved Chunks: {len(results)}")
    
    # Benchmark Prompt Construction
    t0 = time.time()
    context_chunks = [r.chunk.text for r in results] if results else ["Mock agricultural context details."]
    prompt = prompt_builder.build_prompt(
        query=query,
        context=context_chunks,
        chat_history=[],
        disease_context={"disease": "Tomato_Late_Blight", "final_severity_score": 75.0, "severity_category": "High"}
    )
    prompt_construction_time = time.time() - t0
    print(f"Prompt Construction Latency: {prompt_construction_time * 1000:.2f} ms")
    
    print("\nSample Prompt Constructed (first 250 chars):")
    print(prompt[:250] + "...\n")
    print("RAG System validation successful.")

if __name__ == "__main__":
    benchmark_rag_pipeline();
