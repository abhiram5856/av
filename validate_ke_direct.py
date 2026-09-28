from backend.services.knowledge_engine import knowledge_engine

test_classes = [
    'tomato_early_blight',
    'rice_leaf_blast',
    'potato_late_blight',
    'cotton_leaf_curl_virus',
    'tomato_healthy',
    'rice_bacterial_leaf_blight',
    'rice_sheath_blight',
    'groundnut_rust',
    'tomato_yellow_leaf_curl_virus',
]

all_pass = True
for cls in test_classes:
    summary = knowledge_engine.get_disease_summary(cls)
    prevention = knowledge_engine.get_prevention_guidance(cls)
    sources = knowledge_engine.get_source_references(cls)
    env_ctx = knowledge_engine.get_environmental_context(cls)
    lookalikes = knowledge_engine.get_lookalikes(cls)
    completeness = knowledge_engine.get_knowledge_completeness_score(cls)

    dtype = summary.get("disease_type")
    distinctive = str(summary.get("distinctive_pattern", ""))[:90]
    env_temp = env_ctx.get("temperature_context") if env_ctx else "N/A"
    source_names = [s["organization"] for s in sources]
    prev_snippet = prevention[:100]

    # Checks
    generic_prevention = "Ensure proper spacing and use certified disease-free seeds. Rotate crops every"
    if generic_prevention in prevention:
        print(f"FAIL {cls}: Still using old generic prevention text")
        all_pass = False
    if not distinctive and dtype not in ("healthy",):
        print(f"WARN {cls}: No distinctive pattern recorded")
    if completeness not in ("HIGH", "MEDIUM", "LOW", "INSUFFICIENT"):
        print(f"FAIL {cls}: Invalid completeness score: {completeness}")
        all_pass = False

    print(f"=== {cls} ===")
    print(f"  disease_type: {dtype}")
    print(f"  completeness: {completeness}")
    print(f"  distinctive: {distinctive}")
    print(f"  lookalikes: {lookalikes[:2]}")
    print(f"  prevention: {prev_snippet}")
    print(f"  env_temp: {env_temp}")
    print(f"  sources: {source_names}")
    print()

print("ALL CHECKS PASSED" if all_pass else "SOME CHECKS FAILED")
