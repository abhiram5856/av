from typing import List, Dict, Any, Optional

class StandardPromptBuilder:
    """
    NOVA-Enhanced Prompt Builder.
    Constructs the exact system and user prompts for the local LLM.
    """
    
    def __init__(self):
        # Base Persona for the AI
        self.system_prompt = (
            "You are NOVA (Neural Optimized Vision Assistant), an expert agricultural AI. "
            "Your goal is to provide precise, actionable, and empathetic advice to farmers. "
            "You must rely strictly on the provided Context (RAG) and the Computer Vision Diagnostic results. "
            "Always structure your response with clear headings: 'Diagnosis', 'Concern Assessment', "
            "'Immediate Treatment', and 'Preventive Measures'. "
            "CRITICAL: Do NOT invent unsupported environmental relationships. "
            "CRITICAL: Do NOT invent pesticide dosages. Rely only on retrieved sources. "
            "CRITICAL: If the AI confidence score is below 60%, OR if environmental evidence strongly conflicts, you MUST explicitly recommend agronomist verification."
        )

    def build_prompt(
        self, 
        query: str, 
        context: List[Any], 
        chat_history: List[Any], 
        disease_context: Optional[Any] = None,
        iot_context: Optional[Any] = None
    ) -> str:
        """
        Builds the final prompt string to feed to the LLM.
        """
        
        # 1. Start with the System Prompt
        prompt = f"System: {self.system_prompt}\n\n"
        
        # 2. Inject Computer Vision & Concern Context (Crucial for NOVA)
        if disease_context:
            if hasattr(disease_context, "model_dump"):
                d_ctx = disease_context.model_dump()
            elif hasattr(disease_context, "dict"):
                d_ctx = disease_context.dict()
            elif isinstance(disease_context, dict):
                d_ctx = disease_context
            else:
                d_ctx = {}

            disease_name = d_ctx.get('disease') or d_ctx.get('disease_name') or 'Unknown'
            confidence = d_ctx.get('confidence')
            
            # Legacy and new fields
            concern_score = d_ctx.get('concern_score') or d_ctx.get('score') or d_ctx.get('final_severity_score')
            concern_level = d_ctx.get('concern_level') or d_ctx.get('level') or d_ctx.get('severity_category')
            env_conflict = d_ctx.get('environmental_conflict')
            contributing = d_ctx.get('contributing_factors', [])
            limiting = d_ctx.get('limiting_factors', [])
            
            prompt += "--- VISION AI & MULTIMODAL CONTEXT ---\n"
            prompt += f"Detected Disease: {disease_name}\n"
            if confidence is not None:
                conf_val = float(confidence)
                conf_str = f"{conf_val * 100:.1f}%" if conf_val <= 1.0 else f"{conf_val}%"
                prompt += f"Confidence Score: {conf_str}\n"
            
            if concern_score is not None:
                prompt += f"Multimodal Concern Score: {concern_score}/100\n"
            if concern_level:
                prompt += f"Concern Level: {concern_level}\n"
            if env_conflict:
                prompt += f"Environmental Conflict: YES (Environmental conditions do not strongly support diagnosis)\n"
                
            if contributing:
                prompt += "Contributing Factors:\n"
                for factor in contributing:
                    prompt += f"- {factor}\n"
                    
            if limiting:
                prompt += "Limiting Factors:\n"
                for factor in limiting:
                    prompt += f"- {factor}\n"
                    
            prompt += "Based on these diagnostic factors, tailor your treatment recommendations appropriately.\n\n"

        # 3. Inject IoT / Environmental Context
        if iot_context:
            prompt += "--- IOT / SENSOR DATA ---\n"
            if isinstance(iot_context, dict):
                for k, v in iot_context.items():
                    prompt += f"{k}: {v}\n"
            elif hasattr(iot_context, "model_dump"):
                for k, v in iot_context.model_dump().items():
                    prompt += f"{k}: {v}\n"
            prompt += "Use this current environmental data to inform your advice (e.g., watering schedules, fungal risk).\n\n"
            
        # 4. Inject RAG Knowledge Base Context
        if context:
            prompt += "--- RETRIEVED KNOWLEDGE (RAG) ---\n"
            for i, chunk in enumerate(context):
                if isinstance(chunk, dict):
                    chunk_text = chunk.get('text', str(chunk))
                elif hasattr(chunk, 'chunk') and hasattr(chunk.chunk, 'text'):
                    chunk_text = chunk.chunk.text
                elif hasattr(chunk, 'text'):
                    chunk_text = chunk.text
                else:
                    chunk_text = str(chunk)
                prompt += f"[{i+1}] {chunk_text}\n"
            prompt += "\nUse the above verified agricultural data to answer the farmer.\n\n"
            
        # 5. Inject Chat History
        if chat_history:
            prompt += "--- CONVERSATION HISTORY ---\n"
            for msg in chat_history:
                role = getattr(msg, 'role', 'user')
                content = getattr(msg, 'content', '')
                prompt += f"{role.capitalize()}: {content}\n"
            prompt += "\n"
            
        # 6. Add the User's Latest Query
        prompt += f"User Query: {query}\n"
        prompt += "NOVA:"
        
        return prompt
