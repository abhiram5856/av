"""
NOVA Root Cause Dataset (NOVA-RCD) — JSONL Converter
=====================================================
Wrapper script that imports the context generator logic
and converts annotations into JSONL format.
"""

from backend.research.trace_rce_v2.ai_context_generator import generate_contexts

if __name__ == "__main__":
    generate_contexts()
