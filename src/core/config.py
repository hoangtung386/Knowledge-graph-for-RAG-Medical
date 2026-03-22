"""Centralized configuration for PJI RAG system.

All magic numbers, thresholds, and environment-based settings live here.
"""

import os

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# LLM Provider
# ---------------------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "huggingface").lower()

# HuggingFace
HF_BASE_URL = "https://router.huggingface.co/v1"
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_LLM_MODEL = os.getenv("HF_LLM_MODEL", "Qwen/Qwen2.5-72B-Instruct")
HF_LLM_MODEL_QUERY = os.getenv("HF_LLM_MODEL_QUERY", "")
HF_EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
HF_EMBEDDING_DIM = int(os.getenv("HF_EMBEDDING_DIM", "384"))
HF_EMBEDDING_BATCH_SIZE = int(os.getenv("HF_EMBEDDING_BATCH_SIZE", "32"))
HF_EMBEDDING_MAX_WAIT_SEC = 30

# Jina AI (embedding + reranker)
JINA_API_KEY = os.getenv("JINA_API_KEY", "")
JINA_EMBEDDING_MODEL = os.getenv("JINA_EMBEDDING_MODEL", "jina-embeddings-v5-text-small")
JINA_EMBEDDING_DIM = int(os.getenv("JINA_EMBEDDING_DIM", "1024"))
JINA_RERANKER_MODEL = os.getenv("JINA_RERANKER_MODEL", "jina-reranker-v3")

# OpenAI
OPENAI_LLM_INDEXING = os.getenv("LLM_MODEL_INDEXING", "gpt-4o-mini")
OPENAI_LLM_QUERY = os.getenv("LLM_MODEL_QUERY", "gpt-4o")

# ---------------------------------------------------------------------------
# RAG Engine
# ---------------------------------------------------------------------------
RAG_WORKING_DIR = os.getenv("RAG_WORKING_DIR", "./rag_storage")
RAG_CHUNK_TOKEN_SIZE = 1200
RAG_CHUNK_OVERLAP_TOKEN_SIZE = 200
RAG_ENTITY_EXTRACT_MAX_GLEANING = 2
RAG_DEFAULT_TOP_K = 60
RAG_QUERY_MODE = "mix"

RAG_ENTITY_TYPES = [
    "organism",
    "biomarker",
    "diagnostic_criteria",
    "procedure",
    "joint",
    "antibiotic",
    "clinical_sign",
    "lab_test",
    "guideline",
]

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# ---------------------------------------------------------------------------
# Medical Thresholds (for risk factor detection)
# ---------------------------------------------------------------------------
DIABETES_HBA1C_THRESHOLD = 6.5  # %
HYPERGLYCEMIA_GLUCOSE_THRESHOLD = 11.0  # mmol/L (pre-operative target)
RENAL_CREATININE_THRESHOLD = 120  # μmol/L
HEPATIC_ALT_THRESHOLD = 40  # U/L
HEPATIC_AST_THRESHOLD = 40  # U/L
MALNUTRITION_ALBUMIN_THRESHOLD = 35  # g/L
HYPERKALEMIA_THRESHOLD = 5.0  # mmol/L (caution with TMP-SMX)
ANEMIA_RBC_THRESHOLD = 3.5  # x10^12/L

# D-Dimer unit conversion: web sends mg/L FEU, ICM uses ng/mL
D_DIMER_MGL_TO_NGML = 1000

# Acute infection threshold (days since arthroplasty)
ACUTE_INFECTION_DAYS_THRESHOLD = 90

# ---------------------------------------------------------------------------
# Qualitative positive values (for ICM scoring)
# ---------------------------------------------------------------------------
QUALITATIVE_POSITIVE_VALUES = frozenset(
    {"positive", "++", "+++", "detected", "true", "1"}
)

# Truncation limits for RAG queries
CLINICAL_CONTEXT_TRUNCATION = 500
CULTURE_SUMMARY_TRUNCATION_SYSTEMIC = 300
CULTURE_SUMMARY_TRUNCATION_LOCAL = 200
RECOMMENDATION_SUMMARY_TRUNCATION = 300

# Default document directory
DEFAULT_DOCUMENT_DIR = "data/pdf"
