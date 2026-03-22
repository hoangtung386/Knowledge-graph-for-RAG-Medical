"""PJI Diagnosis System - Main pipeline orchestrator.

Combines LightRAG retrieval with ICM 2018 scoring and structured prompts.
"""

import asyncio
import json
import logging

from dotenv import load_dotenv

load_dotenv()

from src.api.schemas import (
    AiRagCitation,
    DiagnosticTestRecommendation,
    LocalAntibioticRecommendation,
    ProcessSnapshotResponse,
    RecommendationItems,
    SnapshotDataJson,
    SurgeryProcedureRecommendation,
    SystemicAntibioticRecommendation,
)
from src.core.config import (
    CLINICAL_CONTEXT_TRUNCATION,
    CULTURE_SUMMARY_TRUNCATION_LOCAL,
    CULTURE_SUMMARY_TRUNCATION_SYSTEMIC,
    DEFAULT_DOCUMENT_DIR,
    RECOMMENDATION_SUMMARY_TRUNCATION,
)
from src.core.document_loader import ingest_directory
from src.core.rag_engine import PJIRagEngine
from src.medical.pji_criteria import ICM2018Criteria, format_icm_summary
from src.medical.prompt_templates import (
    build_guideline_comparison_prompt,
    build_lab_interpretation_prompt,
    build_pji_diagnosis_prompt,
)
from src.medical.recommendation_prompts import (
    build_citations_prompt,
    build_diagnostic_test_prompt,
    build_local_antibiotic_prompt,
    build_surgery_prompt,
    build_systemic_antibiotic_prompt,
)
from src.medical.completeness_checker import check_data_completeness
from src.medical.snapshot_parser import (
    build_clinical_context,
    build_culture_summary,
    determine_is_acute,
    extract_icm_lab_values,
    extract_major_criteria,
    extract_patient_risk_factors,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# user_prompt constants — injected into LightRAG's LLM call AFTER retrieval.
# These do NOT affect retrieval/search, only how LLM formats the response.
# ---------------------------------------------------------------------------
PJI_DIAGNOSIS_USER_PROMPT = """\
You are a clinical decision support system for PJI diagnosis.
Base your response ONLY on the provided context. Do NOT speculate.
Always cite the specific guideline and year.

Respond in this format:
1. **Applicable Criteria** (cite guideline + year)
2. **Assessment** (Infected / Inconclusive / Not Infected / Insufficient Data)
3. **Key Findings** (which markers met/not met)
4. **Recommended Next Steps** (per guideline)
5. **Limitations** (what cannot be determined)

If information is insufficient, state this clearly rather than guessing."""

GUIDELINE_COMPARE_USER_PROMPT = """\
Compare how different PJI guidelines (MSIS 2011, ICM 2018, EBJIS 2021) address this question.
Present a comparison table where possible.
Cite specific guidelines. State clearly if any guideline's data is missing from context."""

LAB_INTERPRET_USER_PROMPT = """\
Interpret each lab result against PJI diagnostic thresholds.
For each marker: state the value, applicable threshold per guideline, and whether it's positive/negative.
Then give an overall assessment. List missing tests that would help.
Mention conditions that could cause false positives or negatives."""


class PJIDiagnosisSystem:
    """Main orchestrator for PJI clinical decision support.

    Two query strategies available:
    - `diagnose()`: Full pipeline — retrieves context, calculates ICM score,
      then lets LightRAG call LLM with user_prompt for structured output.
    - `diagnose_context_only()`: Returns raw context + ICM score + prompt
      for cases where you want to call your own LLM.
    """

    def __init__(self):
        self.rag = PJIRagEngine()
        self.criteria = ICM2018Criteria()
        self._initialized = False

    async def initialize(self):
        """Initialize RAG engine. Must be called before use."""
        await self.rag.initialize()
        self._initialized = True

    async def finalize(self):
        """Clean up resources."""
        if self._initialized:
            await self.rag.finalize()
            self._initialized = False

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    async def ingest_documents(self, directory: str = DEFAULT_DOCUMENT_DIR) -> int:
        """Ingest all documents from a directory (recursive)."""
        self._check_initialized()
        return await ingest_directory(self.rag, directory)

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    async def delete_document(self, doc_id: str):
        """Delete a document and intelligently rebuild affected KG."""
        self._check_initialized()
        await self.rag.delete_document(doc_id)

    async def delete_entity(self, entity_name: str):
        """Delete an entity and all its associated relations."""
        self._check_initialized()
        await self.rag.delete_entity(entity_name)

    # ------------------------------------------------------------------
    # Diagnose — Full LLM pipeline (preferred)
    # ------------------------------------------------------------------

    async def diagnose(
        self,
        clinical_question: str,
        lab_values: dict | None = None,
        major_criteria: list[str] | None = None,
        is_acute: bool = False,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> dict:
        """Full PJI diagnosis: KG+vector retrieval -> rerank -> LLM answer."""
        self._check_initialized()

        icm_result = None
        icm_summary = ""
        if lab_values or major_criteria:
            icm_result = self.criteria.score(
                lab_values=lab_values or {},
                major_criteria_present=major_criteria,
                is_acute=is_acute,
            )
            icm_summary = f"\n\nPre-calculated ICM 2018 Score:\n{format_icm_summary(icm_result)}"

        enriched_query = f"{clinical_question}{icm_summary}"

        answer = await self.rag.query(
            query=enriched_query,
            user_prompt=PJI_DIAGNOSIS_USER_PROMPT,
            conversation_history=conversation_history,
        )

        return {"answer": answer, "icm_score": icm_result}

    async def compare_guidelines(
        self,
        question: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> dict:
        """Compare how different guidelines address a question."""
        self._check_initialized()
        answer = await self.rag.query(
            query=question,
            user_prompt=GUIDELINE_COMPARE_USER_PROMPT,
            conversation_history=conversation_history,
        )
        return {"answer": answer}

    async def interpret_labs(
        self,
        lab_values: dict,
        is_acute: bool = False,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> dict:
        """Interpret lab results in PJI context with ICM scoring + LLM analysis."""
        self._check_initialized()

        icm_result = self.criteria.score(lab_values=lab_values, is_acute=is_acute)
        icm_summary = format_icm_summary(icm_result)

        markers = ", ".join(lab_values.keys())
        lab_text = "\n".join(f"- {k}: {v}" for k, v in lab_values.items())
        query = (
            f"PJI diagnostic thresholds and interpretation for: {markers}\n"
            f"Patient lab results:\n{lab_text}\n\n"
            f"ICM 2018 Score:\n{icm_summary}"
        )

        answer = await self.rag.query(
            query=query,
            user_prompt=LAB_INTERPRET_USER_PROMPT,
            conversation_history=conversation_history,
        )

        return {"answer": answer, "icm_score": icm_result}

    # ------------------------------------------------------------------
    # Process Snapshot — Web team API (primary endpoint)
    # ------------------------------------------------------------------

    async def process_snapshot(
        self,
        snapshot: SnapshotDataJson,
    ) -> ProcessSnapshotResponse:
        """Process a full clinical snapshot and return structured recommendations.

        Pipeline:
        1. Parse snapshot -> extract lab values, clinical context, cultures
        2. Deterministic ICM 2018 scoring
        3. RAG retrieve guideline context (parallel)
        4. LLM generates 4 recommendation JSONs + citations
        5. Parse and validate LLM output into Pydantic models
        """
        self._check_initialized()

        # Step 0: Check data completeness (deterministic, no LLM)
        completeness = check_data_completeness(snapshot)

        # Step 1: Parse snapshot
        parsed = self._parse_snapshot(snapshot)

        # Step 2: Deterministic ICM scoring
        icm_result = self.criteria.score(
            lab_values=parsed["lab_values"],
            major_criteria_present=parsed["major_criteria"],
            is_acute=parsed["is_acute"],
        )

        # Step 3: Retrieve context (parallel)
        contexts = await self._retrieve_contexts(snapshot, parsed)

        # Step 4: Generate recommendations (sequential for rate limits)
        recommendations = await self._generate_recommendations(
            parsed, icm_result, contexts,
        )

        # Step 5: Generate citations
        citations = await self._generate_citations(contexts, recommendations["raw"])

        return ProcessSnapshotResponse(
            recommendation_items=RecommendationItems(
                diagnostic_test=recommendations["diagnostic"],
                surgery_procedure=recommendations["surgery"],
                systemic_antibiotic=recommendations["systemic"],
                local_antibiotic=recommendations["local"],
            ),
            citations=citations,
            data_completeness=completeness,
        )

    @staticmethod
    def _parse_snapshot(snapshot: SnapshotDataJson) -> dict:
        """Extract all structured data from the clinical snapshot."""
        return {
            "lab_values": extract_icm_lab_values(snapshot),
            "major_criteria": extract_major_criteria(snapshot),
            "is_acute": determine_is_acute(snapshot),
            "clinical_context": build_clinical_context(snapshot),
            "culture_summary": build_culture_summary(snapshot),
            "risk_factors": extract_patient_risk_factors(snapshot),
        }

    async def _retrieve_contexts(
        self, snapshot: SnapshotDataJson, parsed: dict,
    ) -> dict[str, str]:
        """Retrieve RAG contexts for each recommendation category in parallel."""
        is_acute = parsed["is_acute"]
        clinical_ctx = parsed["clinical_context"]
        culture_sum = parsed["culture_summary"]
        assess = snapshot.clinical_records.infection_assessment

        queries = {
            "diagnosis": (
                f"PJI diagnostic criteria scoring ICM 2018 MSIS "
                f"{'acute' if is_acute else 'chronic'} infection thresholds "
                f"{clinical_ctx[:CLINICAL_CONTEXT_TRUNCATION]}"
            ),
            "surgery": (
                f"PJI surgical treatment strategy "
                f"{'acute' if is_acute else 'chronic'} infection "
                f"{'DAIR debridement' if is_acute else 'revision arthroplasty one-stage two-stage'} "
                f"{'unstable implant' if assess.implant_stability == 'UNSTABLE' else 'stable implant'}"
            ),
            "systemic": (
                f"PJI systemic antibiotic treatment regimen "
                f"{culture_sum[:CULTURE_SUMMARY_TRUNCATION_SYSTEMIC]} "
                f"IDSA guidelines duration phases IV oral"
            ),
            "local": (
                f"PJI local antibiotic cement spacer PMMA "
                f"antibiotic loading elution "
                f"{culture_sum[:CULTURE_SUMMARY_TRUNCATION_LOCAL]}"
            ),
        }

        results = await asyncio.gather(
            self.rag.retrieve_context(queries["diagnosis"]),
            self.rag.retrieve_context(queries["surgery"]),
            self.rag.retrieve_context(queries["systemic"]),
            self.rag.retrieve_context(queries["local"]),
        )

        return dict(zip(queries.keys(), results))

    async def _generate_recommendations(
        self, parsed: dict, icm_result: dict, contexts: dict[str, str],
    ) -> dict:
        """Generate 4 recommendation JSONs via direct LLM call.

        Context was already retrieved in Step 3 and is embedded in the prompts.
        We call LLM directly (bypass RAG retrieval) to avoid double-retrieval
        and keyword pollution from JSON schema boilerplate.
        See LightRAG docs "User Prompt vs. Query" section.
        """
        clinical_ctx = parsed["clinical_context"]
        culture_sum = parsed["culture_summary"]
        risk_factors = parsed["risk_factors"]

        prompts = {
            "diagnostic": build_diagnostic_test_prompt(
                clinical_ctx, contexts["diagnosis"], icm_result, culture_sum, risk_factors,
            ),
            "surgery": build_surgery_prompt(
                clinical_ctx, contexts["surgery"], culture_sum, risk_factors,
            ),
            "systemic": build_systemic_antibiotic_prompt(
                clinical_ctx, contexts["systemic"], culture_sum, risk_factors,
            ),
            "local": build_local_antibiotic_prompt(
                clinical_ctx, contexts["local"], culture_sum, risk_factors,
            ),
        }

        model_map = {
            "diagnostic": DiagnosticTestRecommendation,
            "surgery": SurgeryProcedureRecommendation,
            "systemic": SystemicAntibioticRecommendation,
            "local": LocalAntibioticRecommendation,
        }

        raw_responses = {}
        parsed_recs = {}

        for key, prompt in prompts.items():
            raw = await self.rag.llm_complete(prompt)
            raw_responses[key] = raw
            parsed_recs[key] = _parse_json_response(raw, model_map[key])

        return {**parsed_recs, "raw": raw_responses}

    async def _generate_citations(
        self, contexts: dict[str, str], raw_responses: dict[str, str],
    ) -> list[AiRagCitation]:
        """Generate citations via direct LLM call (context already retrieved)."""
        all_context = "\n".join(contexts.values())
        trunc = RECOMMENDATION_SUMMARY_TRUNCATION
        rec_summary = "\n".join(
            f"{key.upper()}: {raw[:trunc]}"
            for key, raw in raw_responses.items()
        )

        citations_prompt = build_citations_prompt(all_context, rec_summary)
        citations_json = await self.rag.llm_complete(citations_prompt)
        return _parse_citations(citations_json)

    # ------------------------------------------------------------------
    # Context-only mode (for custom LLM integration)
    # ------------------------------------------------------------------

    async def diagnose_context_only(
        self,
        clinical_question: str,
        lab_values: dict | None = None,
        major_criteria: list[str] | None = None,
        is_acute: bool = False,
    ) -> dict:
        """Returns raw context + ICM score + prompt without calling LLM."""
        self._check_initialized()

        rag_context = await self.rag.retrieve_context(clinical_question)

        icm_result = None
        icm_summary = None
        if lab_values or major_criteria:
            icm_result = self.criteria.score(
                lab_values=lab_values or {},
                major_criteria_present=major_criteria,
                is_acute=is_acute,
            )
            icm_summary = format_icm_summary(icm_result)

        prompt = build_pji_diagnosis_prompt(
            clinical_context=clinical_question,
            rag_context=rag_context,
            icm_score_summary=icm_summary,
        )

        return {
            "rag_context": rag_context,
            "icm_score": icm_result,
            "prompt_for_llm": prompt,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_initialized(self):
        if not self._initialized:
            raise RuntimeError("System not initialized. Call initialize() first.")


# ---------------------------------------------------------------------------
# JSON parsing utilities
# ---------------------------------------------------------------------------


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences from LLM response."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        end = -1 if lines[-1].strip() == "```" else len(lines)
        text = "\n".join(lines[1:end])
    return text


def _parse_json_response(raw: str, model_class: type):
    """Parse LLM JSON response into a Pydantic model."""
    text = _strip_markdown_fences(raw)
    try:
        data = json.loads(text)
        return model_class.model_validate(data)
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(
            "Failed to parse LLM response as %s: %s\nRaw (500 chars): %s",
            model_class.__name__, e, raw[:500],
        )
        return None


def _parse_citations(raw: str) -> list[AiRagCitation]:
    """Parse citations JSON array from LLM response."""
    text = _strip_markdown_fences(raw)
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [AiRagCitation.model_validate(item) for item in data]
    except (json.JSONDecodeError, Exception) as e:
        logger.warning("Failed to parse citations: %s", e)
    return []


# ---------------------------------------------------------------------------
# CLI entry point for quick testing
# ---------------------------------------------------------------------------
async def _demo():
    system = PJIDiagnosisSystem()
    await system.initialize()

    try:
        count = await system.ingest_documents()
        if count:
            logger.info("Ingested %d documents", count)
        else:
            logger.info("No data files found. Place PDFs in data/pdf/ and run again.")

        demo_labs = {
            "serum_CRP": 15,
            "serum_ESR": 45,
            "synovial_WBC": 4500,
            "synovial_PMN": 85,
            "synovial_alpha_defensin": "positive",
        }

        result = await system.diagnose(
            clinical_question="68-year-old male, 2 years post TKA, presenting with chronic knee pain and swelling. Assess for PJI.",
            lab_values=demo_labs,
            is_acute=False,
        )

        print("\n--- ICM 2018 Score ---")
        print(json.dumps(result["icm_score"], indent=2))

        print("\n--- LLM Answer ---")
        print(result["answer"])

    finally:
        await system.finalize()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_demo())
