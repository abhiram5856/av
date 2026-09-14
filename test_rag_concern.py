import asyncio
from backend.models.concern_scorer import MultimodalConcernScorer
from backend.services.multimodal_evidence_engine import MultimodalEvidenceEngine
from pathlib import Path

BASE_DIR = Path('c:/Users/ABHIRAM MODUKURU/OneDrive/Desktop/AgriVision-AI')
RULES_PATH = BASE_DIR / 'knowledge_base' / 'agronomic_rules.json'

async def test_concern_and_rag():
    print('Testing Concern Scorer...')
    try:
        scorer = MultimodalConcernScorer()
        score = scorer.calculate_concern(
            disease='tomato_late_blight',
            confidence=0.99,
            temperature=26.5,
            humidity=85.0,
            growth_stage='Unknown',
            lesion_ratio=0.3
        )
        print(f'Concern Score result: {score}')
    except Exception as e:
        print(f'Concern Scorer failed: {e}')

    print('\nTesting RAG Engine...')
    try:
        engine = MultimodalEvidenceEngine(str(RULES_PATH))
        context = await engine.evaluate(
            disease='tomato_late_blight',
            confidence=0.99,
            temperature=26.5,
            humidity=85.0,
            ph_level=6.5,
            lesion_ratio=0.3
        )
        print(f'RAG Context: {context.keys() if hasattr(context, "keys") else type(context)}')
    except Exception as e:
        print(f'RAG failed: {e}')

asyncio.run(test_concern_and_rag())
