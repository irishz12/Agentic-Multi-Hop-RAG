"""Phase 8B: the complete Adaptive Cost-Aware Multi-Hop Retrieval pipeline
— Hybrid RRF -> frozen Phase 8A.2 learned router -> the cheapest backend
each question's router decision actually needs, always finishing with the
same Qwen final-answer generation used by Always-Agentic. See
`mhrag.adaptive.pipeline`.
"""
