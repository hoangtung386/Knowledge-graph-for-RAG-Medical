"""LightRAG wrapper for PJI medical knowledge retrieval.

Supports OpenAI and HuggingFace LLM providers.
Embedding: Jina AI (primary) or HuggingFace (fallback).
Reranker: Jina AI (if API key provided).
"""

import logging
import os
from functools import partial

import numpy as np
from lightrag import LightRAG, QueryParam
from lightrag.llm.jina import jina_embed
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.rerank import jina_rerank
from lightrag.utils import EmbeddingFunc, setup_logger, wrap_embedding_func_with_attrs

from src.core.config import (
    HF_BASE_URL,
    HF_EMBEDDING_BATCH_SIZE,
    HF_EMBEDDING_DIM,
    HF_EMBEDDING_MAX_WAIT_SEC,
    HF_EMBEDDING_MODEL,
    HF_LLM_MODEL,
    HF_LLM_MODEL_QUERY,
    HF_TOKEN,
    JINA_API_KEY,
    JINA_EMBEDDING_DIM,
    JINA_EMBEDDING_MODEL,
    JINA_RERANKER_MODEL,
    LLM_PROVIDER,
    OPENAI_LLM_INDEXING,
    OPENAI_LLM_QUERY,
    RAG_CHUNK_OVERLAP_TOKEN_SIZE,
    RAG_CHUNK_TOKEN_SIZE,
    RAG_DEFAULT_TOP_K,
    RAG_ENTITY_EXTRACT_MAX_GLEANING,
    RAG_ENTITY_TYPES,
    RAG_QUERY_MODE,
    RAG_WORKING_DIR,
)

setup_logger("lightrag", level="INFO")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HuggingFace LLM functions (OpenAI-compatible API)
# ---------------------------------------------------------------------------


async def _hf_llm_complete(
    prompt,
    system_prompt=None,
    history_messages=[],
    keyword_extraction=False,
    model: str | None = None,
    **kwargs,
) -> str:
    """Call HuggingFace LLM via OpenAI-compatible endpoint."""
    return await openai_complete_if_cache(
        model or HF_LLM_MODEL,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=HF_TOKEN,
        base_url=HF_BASE_URL,
        **kwargs,
    )


async def hf_llm_indexing(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    """HF LLM for indexing (default model)."""
    return await _hf_llm_complete(
        prompt, system_prompt, history_messages, keyword_extraction, **kwargs
    )


async def hf_llm_query(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    """HF LLM for queries (optionally stronger model)."""
    return await _hf_llm_complete(
        prompt,
        system_prompt,
        history_messages,
        keyword_extraction,
        model=HF_LLM_MODEL_QUERY or None,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# HuggingFace Embedding function (fallback when no JINA_API_KEY)
# ---------------------------------------------------------------------------


@wrap_embedding_func_with_attrs(
    embedding_dim=HF_EMBEDDING_DIM,
    max_token_size=8192,
    model_name=HF_EMBEDDING_MODEL,
)
async def hf_embedding_func(texts: list[str]) -> np.ndarray:
    """Call HuggingFace Inference API for sentence embeddings.

    Uses the native HF inference endpoint with batch input format.
    HF does NOT support OpenAI-compatible /v1/embeddings for most models.
    """
    import asyncio

    import aiohttp

    url = f"https://router.huggingface.co/hf-inference/models/{HF_EMBEDDING_MODEL}"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }
    params = {"task": "feature-extraction"}

    all_embeddings = []
    async with aiohttp.ClientSession() as session:
        for i in range(0, len(texts), HF_EMBEDDING_BATCH_SIZE):
            batch = texts[i : i + HF_EMBEDDING_BATCH_SIZE]
            embeddings = await _fetch_hf_embeddings(session, url, headers, params, batch)
            all_embeddings.extend(embeddings)

    return np.array(all_embeddings, dtype=np.float32)


async def _fetch_hf_embeddings(
    session, url: str, headers: dict, params: dict, batch: list[str]
) -> list[list[float]]:
    """Fetch embeddings for a single batch, with 503 retry."""
    import asyncio

    payload = {"inputs": batch}

    async with session.post(url, headers=headers, json=payload, params=params) as resp:
        if resp.status == 200:
            return _parse_embedding_response(await resp.json())

        if resp.status == 503:
            error = await resp.json()
            wait_time = min(error.get("estimated_time", 20), HF_EMBEDDING_MAX_WAIT_SEC)
            logger.warning("HF model loading, waiting %ds...", wait_time)
            await asyncio.sleep(wait_time)

            async with session.post(url, headers=headers, json=payload, params=params) as retry:
                retry.raise_for_status()
                return _parse_embedding_response(await retry.json())

        error_text = await resp.text()
        raise RuntimeError(f"HF embedding error ({resp.status}): {error_text[:200]}")


def _parse_embedding_response(result: list) -> list[list[float]]:
    """Normalize HF embedding response to list of vectors."""
    if isinstance(result, list) and len(result) > 0:
        if isinstance(result[0], list):
            return result
        if isinstance(result[0], float):
            return [result]
    raise ValueError(f"Unexpected embedding response shape: {type(result)}")


# ---------------------------------------------------------------------------
# OpenAI LLM functions
# ---------------------------------------------------------------------------


async def openai_llm_indexing(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        OPENAI_LLM_INDEXING,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def openai_llm_query(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        OPENAI_LLM_QUERY,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Embedding selection: Jina (primary) → HuggingFace (fallback)
# ---------------------------------------------------------------------------


def _select_embedding_func():
    """Select embedding function based on available API keys.

    Priority: Jina AI (if JINA_API_KEY set) → HuggingFace (fallback).
    """
    if JINA_API_KEY:
        logger.info(
            "Using Jina embedding: %s (dim=%d)",
            JINA_EMBEDDING_MODEL, JINA_EMBEDDING_DIM,
        )
        return EmbeddingFunc(
            embedding_dim=JINA_EMBEDDING_DIM,
            max_token_size=32000,
            model_name=JINA_EMBEDDING_MODEL,
            func=partial(
                jina_embed.func,
                model=JINA_EMBEDDING_MODEL,
                embedding_dim=JINA_EMBEDDING_DIM,
                api_key=JINA_API_KEY,
            ),
        )

    logger.info(
        "Using HuggingFace embedding: %s (dim=%d)",
        HF_EMBEDDING_MODEL, HF_EMBEDDING_DIM,
    )
    return hf_embedding_func


def _select_rerank_func():
    """Select reranker function. Jina if API key set, else None."""
    if JINA_API_KEY:
        logger.info("Using Jina reranker: %s", JINA_RERANKER_MODEL)
        return partial(jina_rerank, model=JINA_RERANKER_MODEL, api_key=JINA_API_KEY)
    return None


# ---------------------------------------------------------------------------
# Provider registry (LLM only — embedding/reranker selected separately)
# ---------------------------------------------------------------------------

_LLM_PROVIDERS = {
    "openai": {
        "indexing_llm": openai_llm_indexing,
        "query_llm": openai_llm_query,
    },
    "huggingface": {
        "indexing_llm": hf_llm_indexing,
        "query_llm": hf_llm_query,
    },
}


class PJIRagEngine:
    """Wrapper around LightRAG for PJI medical knowledge retrieval.

    LLM: HuggingFace (default) or OpenAI — set via LLM_PROVIDER env var.
    Embedding: Jina AI (if JINA_API_KEY set) or HuggingFace.
    Reranker: Jina AI (if JINA_API_KEY set) or disabled.
    """

    def __init__(self, working_dir: str | None = None):
        self.working_dir = working_dir or RAG_WORKING_DIR

        llm_provider = _LLM_PROVIDERS.get(LLM_PROVIDER, _LLM_PROVIDERS["huggingface"])
        self._query_llm_func = llm_provider["query_llm"]

        self.rag = LightRAG(
            working_dir=self.working_dir,
            llm_model_func=llm_provider["indexing_llm"],
            embedding_func=_select_embedding_func(),
            rerank_model_func=_select_rerank_func(),
            addon_params={
                "language": "English",
                "entity_types": RAG_ENTITY_TYPES,
            },
            chunk_token_size=RAG_CHUNK_TOKEN_SIZE,
            chunk_overlap_token_size=RAG_CHUNK_OVERLAP_TOKEN_SIZE,
            entity_extract_max_gleaning=RAG_ENTITY_EXTRACT_MAX_GLEANING,
        )

    async def initialize(self):
        """Must be called before any insert/query operations."""
        await self.rag.initialize_storages()

    async def finalize(self):
        """Clean up storage connections."""
        await self.rag.finalize_storages()

    async def ingest(self, text: str, source: str | None = None):
        """Insert document text into LightRAG.

        LightRAG automatically: chunks text -> extracts entities/relations
        via LLM -> builds knowledge graph -> embeds into vector store.
        """
        kwargs = {}
        if source:
            kwargs["file_paths"] = [source]
        await self.rag.ainsert(text, **kwargs)

    async def retrieve_context(self, query: str, top_k: int = RAG_DEFAULT_TOP_K) -> str:
        """Retrieve raw context without LLM generation."""
        return await self.rag.aquery(
            query,
            param=QueryParam(
                mode=RAG_QUERY_MODE,
                top_k=top_k,
                only_need_context=True,
            ),
        )

    async def query(
        self,
        query: str,
        user_prompt: str | None = None,
        conversation_history: list[dict[str, str]] | None = None,
        top_k: int = RAG_DEFAULT_TOP_K,
    ) -> str:
        """Full RAG query: retrieve from KG + vector -> rerank -> LLM answer."""
        param = QueryParam(mode=RAG_QUERY_MODE, top_k=top_k)
        param.model_func = self._query_llm_func

        if user_prompt:
            param.user_prompt = user_prompt
        if conversation_history:
            param.conversation_history = conversation_history

        return await self.rag.aquery(query, param=param)

    async def llm_complete(self, prompt: str, system_prompt: str | None = None) -> str:
        """Call LLM directly, bypassing RAG retrieval pipeline.

        Use this when you already have the context (e.g., from retrieve_context)
        and only need LLM generation. Avoids double-retrieval overhead.
        """
        return await self._query_llm_func(
            prompt, system_prompt=system_prompt,
        )

    async def delete_document(self, doc_id: str):
        """Delete a document and rebuild affected KG entities/relations."""
        await self.rag.adelete_by_doc_id(doc_id)

    async def delete_entity(self, entity_name: str):
        """Delete an entity and all its associated relations from the KG."""
        await self.rag.adelete_by_entity(entity_name)
