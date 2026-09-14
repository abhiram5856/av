import asyncio
from backend.models.concern_scorer import MultimodalConcernScorer
from backend.services.multimodal_evidence_engine import MultimodalEvidenceEngine
from pathlib import Path

BASE_DIR = Path('c:/Users/ABHIRAM MODUKURU/OneDrive/Desktop/AgriVision-AI')
RULES_PATH = BASE_DIR / 'knowledge_base' / 'agronomic_rules.json'

async def test_concern_and_rag():
    print('Testing RAG Engine...')
    try:
        engine = MultimodalEvidenceEngine(str(RULES_PATH))
        env_evidence = engine.evaluate(
            disease_name='tomato_late_blight',
            temperature=26.5,
            humidity=85.0,
            ph=6.5
        )
        print(f'RAG env_evidence: {env_evidence}')
    except Exception as e:
        print(f'RAG failed: {e}')
        env_evidence = {}

    print('\nTesting Concern Scorer...')
    try:
        scorer = MultimodalConcernScorer()
        score = scorer.calculate_concern(
            disease_name='tomato_late_blight',
            is_healthy=False,
            ml_confidence=0.99,
            lesion_area_ratio=0.3,
            environmental_evidence=env_evidence,
            growth_stage='Unknown',
            image_quality_status='Good'
        )
        print(f'Concern Score result: {score}')
    except Exception as e:
        print(f'Concern Scorer failed: {e}')

asyncio.run(test_concern_and_rag())
