"""
NOVA Root Cause Engine — TRACE-RCE
====================================
Temporal-Relational Adaptive Causal Explainer — Root Cause Engine

This package is NOVA's primary original research contribution (Phase 2).
It implements the TRACE-RCE algorithm for multi-modal causal attribution
of agricultural plant diseases.

Architecture:
    ontology        — Causal Hypothesis Set (CHS) for each disease family
    evidence        — Evidence Extraction Module (EEM) per modality
    fusion          — Adaptive Evidence Fusion (AEF) operator
    calibration     — Confidence Calibration Module (CCM)
    conflict        — Conflict Resolution Layer (CRL)
    ranking         — Causal Ranking Engine (CRE) + Intervention Priority Scheduler
    models          — Pydantic output schemas
    engine          — TRACE-RCE main engine (AbstractRootCauseEngine implementation)
    evaluation      — Evaluation harness and ablation study framework
"""
