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
            "Always structure your response with clear headings: 'Diagnosis', 'Severity Assessment', "
            "'Immediate Treatment', and 'Preventive Measures'."
        )

    def build_prompt(
        self, 
        query: str, 
        context: List[Any], 
        chat_history: List[Any], 
        disease_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Builds the final prompt string to feed to the LLM.
        """
        
        # 1. Start with the System Prompt
        prompt = f"System: {self.system_prompt}\n\n"
        
        # 2. Inject Computer Vision & Severity Context (Crucial for NOVA)
        if disease_context:
            prompt += "--- DIAGNOSTIC CONTEXT (FROM COMPUTER VISION) ---\n"
            prompt += f"Detected Disease: {disease_context.get('disease', 'Unknown')}\n"
            prompt += f"Severity Score: {disease_context.get('final_severity_score', 'N/A')}/100\n"
            prompt += f"Severity Category: {disease_context.get('severity_category', 'Unknown')}\n"
            prompt += f"Environmental Risk Factor: {disease_context.get('environmental_risk_factor', 'N/A')}\n"
            prompt += f"Soil Stress Factor: {disease_context.get('soil_stress_factor', 'N/A')}\n"
            prompt += "Based on these diagnostic factors, tailor your treatment recommendations appropriately.\n\n"
            
        # 3. Inject RAG Knowledge Base Context
        if context:
            prompt += "--- KNOWLEDGE BASE CONTEXT (RAG) ---\n"
            for i, chunk in enumerate(context):
                # Assuming chunk is an object with a 'text' property or a dictionary
                chunk_text = chunk.get('text', str(chunk)) if isinstance(chunk, dict) else str(chunk)
                prompt += f"Source {i+1}: {chunk_text}\n"
            prompt += "\nUse the above verified agricultural data to answer the farmer.\n\n"
            
        # 4. Inject Chat History
        if chat_history:
            prompt += "--- CONVERSATION HISTORY ---\n"
            for msg in chat_history:
                # Assuming msg is a ChatMessage object with role and content
                role = getattr(msg, 'role', 'user')
                content = getattr(msg, 'content', '')
                prompt += f"{role.capitalize()}: {content}\n"
            prompt += "\n"
            
        # 5. Add the User's Latest Query
        prompt += f"User Query: {query}\n"
        prompt += "NOVA:"
        
        return prompt
