import os

with open('backend/api/diagnose.py', 'r') as f:
    content = f.read()

if 'from backend.services.knowledge_engine import knowledge_engine' not in content:
    content = content.replace('from backend.models.class_registry import', 'from backend.services.knowledge_engine import knowledge_engine\nfrom backend.models.class_registry import')

new_response = """
        # ── KNOWLEDGE ENGINE & ACTION PRIORITY ─────────────────────────────
        farmer_support = knowledge_engine.format_farmer_support(predicted_disease)
        action_priority = "LOW" if is_healthy(predicted_disease) else ("MONITOR" if is_low_confidence else "ATTENTION")

        # ── Response ───────────────────────────────────────────────────────
        return JSONResponse(content={
            "status": "success",
            "request_id": request_id,
            "is_demo_mode": not has_real_model,
            
            "VISION": {
                "prediction": predicted_disease if not is_low_confidence else "Unknown",
                "display_name": class_to_display(predicted_disease) if not is_low_confidence else "Uncertain",
                "is_healthy": is_healthy(predicted_disease) if not is_low_confidence else False,
                "confidence": round(confidence * 100, 2),
                "visual_evidence": {
                    "gradcam_available": True,
                    "attention_indicator": lesion_ratio,
                    "heatmap_b64": heatmap_b64,
                    "leaf_region_estimate": "whole leaf" if lesion_ratio > 0.5 else "localized spots",
                    "symptom_region_estimate": "Detected regions of interest on the leaf"
                },
                "low_confidence_warning": (
                    "Insufficient visual evidence for a reliable diagnosis. "
                    "Please retake the photo in better lighting or capture a closer leaf image."
                ) if is_low_confidence else None
            },
            
            "ENVIRONMENT": {
                "temperature": temperature,
                "humidity": humidity,
                "soil_ph": ph_level,
                "compatibility": env_evidence.get("overall_compatibility", "Insufficient Evidence"),
                "explanation": "Current environmental conditions are supportive of disease development." if env_evidence.get("overall_compatibility") == "Supportive" else "No major environmental conflict detected."
            },
            
            "IOT": {
                "observations": [],
                "status": "UNAVAILABLE"
            },
            
            "KNOWLEDGE": {
                "disease_knowledge": knowledge_engine.get_disease_knowledge(predicted_disease),
                "prevention": knowledge_engine.get_prevention_guidance(predicted_disease),
                "chemical_control": knowledge_engine.get_chemical_control(predicted_disease)
            },
            
            "DECISION": {
                "action_priority": action_priority,
                "what_to_do_now": farmer_support.get("what_to_do_now"),
                "monitor": farmer_support.get("monitor"),
                "prevention": farmer_support.get("prevention"),
                "expert_review": farmer_support.get("expert_review"),
                "rescan_instructions": "Move closer, focus on one leaf, use natural light, hold camera steady." if is_low_confidence else None
            },
            
            # Legacy fields for backward compatibility
            "prediction": {
                "disease": predicted_disease,
                "confidence": round(confidence * 100, 2),
                "display_name": class_to_display(predicted_disease)
            },
            "ai_context": ai_context.to_dict(),
        })
"""

start_idx = content.find('        # ── Response ───────────────────────────────────────────────────────\n        return JSONResponse(content={')
if start_idx == -1:
    print('Could not find start index for replacement.')
else:
    end_idx = content.find('        })', start_idx) + 10
    if end_idx < 10:
        print('Could not find end index.')
    else:
        new_content = content[:start_idx] + new_response.strip() + '\n\n' + content[end_idx:]
        with open('backend/api/diagnose.py', 'w') as f:
            f.write(new_content)
        print('Updated diagnose.py successfully.')
