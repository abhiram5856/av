import pytest
from uuid import uuid4
from backend.prompt_builder.rag_prompt import StandardPromptBuilder
from backend.models.domain import SearchResult, Chunk, ChatMessage, DiseasePrediction

def test_prompt_builder_basic():
    builder = StandardPromptBuilder()
    prompt = builder.build_prompt("What is NPK?", [], [])
    
    assert "System:" in prompt
    assert "Zenith AgriBot" in prompt
    assert "User: What is NPK?" in prompt
    assert "Assistant:" in prompt
    assert "RETRIEVED KNOWLEDGE" not in prompt

def test_prompt_builder_with_context():
    builder = StandardPromptBuilder()
    
    mock_chunk = Chunk(document_id=uuid4(), text="NPK stands for Nitrogen, Phosphorus, Potassium.", chunk_index=0)
    context = [SearchResult(chunk=mock_chunk, score=0.9)]
    
    prompt = builder.build_prompt("What is NPK?", context, [])
    
    assert "RETRIEVED KNOWLEDGE" in prompt
    assert "[1] NPK stands for Nitrogen, Phosphorus, Potassium." in prompt
    assert "User: What is NPK?" in prompt

def test_prompt_builder_with_cnn_disease():
    builder = StandardPromptBuilder()
    
    disease = DiseasePrediction(
        disease_name="Tomato Blight",
        confidence=0.95,
        treatment_recommendations=["Use fungicide"]
    )
    
    prompt = builder.build_prompt("How do I fix this?", [], [], disease_context=disease)
    
    assert "VISION AI CONTEXT" in prompt
    assert "Tomato Blight" in prompt
    assert "95.0%" in prompt
    assert "Use fungicide" in prompt
    assert "User: How do I fix this?" in prompt
