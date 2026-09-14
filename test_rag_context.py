import asyncio
from backend.services.context_builder import AIContextBuilder

async def test_ai_context_builder():
    print('Testing AI Context Builder...')
    try:
        env_evidence = {'overall_compatibility': 'Incompatible', 'factors': []}
        concern_result = {'score': 59.6, 'level': 'High Concern', 'evidence_quality': 'Poor'}
        
        context = await AIContextBuilder.build_context(
            disease_name='tomato_late_blight',
            temperature=26.5,
            humidity=85.0,
            ph=6.5,
            env_evidence=env_evidence,
            concern_result=concern_result
        )
        print(f'RAG Context keys: {context.keys()}')
        if 'treatment' in context:
            print('Treatment present: YES')
            print(context['treatment'])
        else:
            print('Treatment present: NO')
    except Exception as e:
        print(f'AI Context Builder failed: {e}')

asyncio.run(test_ai_context_builder())
