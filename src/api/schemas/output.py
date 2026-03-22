"""Output schemas — ai_recommendation_items + ai_rag_citations.

Matches lines 265-625 of docs/api_contract.json.
"""

from __future__ import annotations

from pydantic import BaseModel

from src.api.schemas.input import SnapshotDataJson


# ============================================================================
# LOCAL_ANTIBIOTIC
# ============================================================================


class LocalAntibioticItem(BaseModel):
    antibiotic_name: str
    dosage: str
    frequency: str
    route: str  # LOCAL_CEMENT, LOCAL_BEAD, etc.
    sequence_order: int
    role: str  # PRIMARY, SYNERGISTIC
    notes: str


class DeliveryInfo(BaseModel):
    delivery_method: str  # CEMENT_SPACER, ANTIBIOTIC_BEAD, etc.
    spacer_type: str | None = None  # ARTICULATING, STATIC
    cement_brand_suggestion: str | None = None
    mixing_ratio: str | None = None


class LocalAntibioticRecommendation(BaseModel):
    category: str = "LOCAL_ANTIBIOTIC"
    title: str
    regimen_name: str
    indication: str
    duration_days: int | None = None
    duration_note: str | None = None
    delivery_info: DeliveryInfo
    antibiotics: list[LocalAntibioticItem]
    monitoring: list[str] = []
    contraindications: list[str] = []
    notes: str | None = None


# ============================================================================
# SYSTEMIC_ANTIBIOTIC
# ============================================================================


class SystemicAntibioticItem(BaseModel):
    antibiotic_name: str
    dosage: str
    frequency: str
    route: str  # IV, ORAL
    sequence_order: int
    role: str  # PRIMARY, BIOFILM_AGENT, PRIMARY_ORAL
    notes: str


class SystemicPhase(BaseModel):
    phase_name: str
    phase_order: int
    duration_weeks: int
    duration_note: str | None = None
    antibiotics: list[SystemicAntibioticItem]


class SystemicAntibioticRecommendation(BaseModel):
    category: str = "SYSTEMIC_ANTIBIOTIC"
    title: str
    regimen_name: str
    indication: str
    total_duration_weeks: int | None = None
    phases: list[SystemicPhase]
    monitoring: list[str] = []
    contraindications: list[str] = []
    notes: str | None = None


# ============================================================================
# SURGERY_PROCEDURE
# ============================================================================


class SurgeryStage(BaseModel):
    stage_order: int
    stage_name: str
    estimated_duration_minutes: int | None = None
    interval_from_stage1: str | None = None
    preconditions: list[str] = []


class SurgeryProcedureRecommendation(BaseModel):
    category: str = "SURGERY_PROCEDURE"
    surgery_strategy_type: str  # TWO_STAGE_REVISION, ONE_STAGE_REVISION, DAIR
    strategy_rationale: str
    priority_level: str  # HIGH, MEDIUM, LOW
    priority_note: str | None = None
    stages: list[SurgeryStage]
    estimated_total_treatment_time: str | None = None
    risks_and_complications: list[str] = []
    notes: str | None = None


# ============================================================================
# DIAGNOSTIC_TEST
# ============================================================================


class MajorCriterionItem(BaseModel):
    criterion: str
    result: bool | None = None
    result_detail: str
    is_decisive: bool = False


class MajorCriteriaResult(BaseModel):
    items: list[MajorCriterionItem]
    major_criteria_met: bool
    major_criteria_conclusion: str


class MinorCriterionItem(BaseModel):
    criterion: str
    score_weight: int
    result: bool | None = None
    result_detail: str
    score_awarded: int = 0


class MinorCriteriaScoringResult(BaseModel):
    items: list[MinorCriterionItem]
    total_minor_score: int
    total_minor_score_note: str


class ScoringSystem(BaseModel):
    name: str = "ICM 2018 PJI Diagnostic Criteria"
    version: str = "2018 (Philadelphia Consensus)"
    total_score: int
    interpretation: str  # INFECTED, INCONCLUSIVE, NOT_INFECTED
    confidence_note: str | None = None


class IdentifiedOrganism(BaseModel):
    name: str
    resistance_profile: str | None = None
    resistance_detail: str | None = None
    biofilm_forming: bool | None = None
    virulence_note: str | None = None


class WarningItem(BaseModel):
    type: str  # ALLERGY_ALERT, DRUG_INTERACTION, RENAL_CHECK, etc.
    severity: str  # HIGH, MEDIUM, LOW
    message: str


class AIReasoning(BaseModel):
    primary_diagnosis: str
    infection_classification: str  # ACUTE, CHRONIC
    infection_classification_reasoning: str
    identified_organism: IdentifiedOrganism | None = None
    reasoning_summary: str
    warnings: list[WarningItem] = []


class DiagnosticTestRecommendation(BaseModel):
    category: str = "DIAGNOSTIC_TEST"
    title: str
    scoring_system: ScoringSystem
    major_criteria: MajorCriteriaResult
    minor_criteria_scoring: MinorCriteriaScoringResult
    ai_reasoning: AIReasoning


# ============================================================================
# Citations
# ============================================================================


class AiRagCitation(BaseModel):
    source_type: str  # GUIDELINE, META_ANALYSIS, JOURNAL_ARTICLE, etc.
    source_title: str
    source_uri: str | None = None
    snippet: str
    relevance_score: float
    cited_for: str


# ============================================================================
# Combined API Request / Response
# ============================================================================


class ProcessSnapshotRequest(BaseModel):
    snapshot_data_json: SnapshotDataJson


class RecommendationItems(BaseModel):
    local_antibiotic: LocalAntibioticRecommendation | None = None
    systemic_antibiotic: SystemicAntibioticRecommendation | None = None
    surgery_procedure: SurgeryProcedureRecommendation | None = None
    diagnostic_test: DiagnosticTestRecommendation | None = None


class MissingDataItem(BaseModel):
    field: str
    category: str  # ICM_MAJOR, ICM_MINOR, CLINICAL, LAB
    importance: str  # CRITICAL, HIGH, MEDIUM
    message: str


class DataCompletenessReport(BaseModel):
    is_complete: bool
    missing_items: list[MissingDataItem] = []
    completeness_score: str  # e.g. "7/12 ICM criteria available"
    impact_note: str  # How missing data affects the recommendations


class ProcessSnapshotResponse(BaseModel):
    recommendation_items: RecommendationItems
    citations: list[AiRagCitation] = []
    data_completeness: DataCompletenessReport | None = None
