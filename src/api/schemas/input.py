"""Input schemas — snapshot_data_json from web team.

Matches lines 1-263 of docs/api_contract.json.
"""

from __future__ import annotations

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Snapshot sub-models
# ---------------------------------------------------------------------------


class SnapshotMetadata(BaseModel):
    snapshot_at: str
    episode_id: int
    episode_code: str
    patient_id: int
    data_sources: list[str] = []


class PatientDemographics(BaseModel):
    patient_code: str
    full_name: str
    date_of_birth: str
    gender: str


class Allergies(BaseModel):
    is_allergy: bool = False
    allergy_note: str | None = None


class SubstanceUse(BaseModel):
    is_drug: bool = False
    drug_note: str | None = None
    is_alcohol: bool = False
    alcohol_note: str | None = None
    is_smoking: bool = False
    smoking_note: str | None = None
    is_other: bool = False
    other_note: str | None = None


class MedicalHistory(BaseModel):
    medical_history: str | None = None
    process: str | None = None
    allergies: Allergies = Allergies()
    substance_use: SubstanceUse = SubstanceUse()


class Vitals(BaseModel):
    blood_pressure: str | None = None
    bmi: float | None = None


class Symptoms(BaseModel):
    fever: bool = False
    pain: bool = False
    erythema: bool = False
    swelling: bool = False
    sinus_tract: bool = False


class InfectionAssessment(BaseModel):
    suspected_infection_type: str | None = None  # ACUTE / CHRONIC
    hematogenous_suspected: bool = False
    implant_stability: str | None = None  # STABLE / UNSTABLE
    soft_tissue: str | None = None
    pmma_allergy: bool = False
    prosthesis_joint: str | None = None
    days_since_index_arthroplasty: int | None = None


class ClinicalRecords(BaseModel):
    record_id: int | None = None
    recorded_at: str | None = None
    illness_onset_date: str | None = None
    vitals: Vitals = Vitals()
    symptoms: Symptoms = Symptoms()
    infection_assessment: InfectionAssessment = InfectionAssessment()
    notations: str | None = None


class SurgeryItem(BaseModel):
    surgery_id: int | None = None
    surgery_date: str | None = None
    surgery_type: str | None = None
    findings: str | None = None


class Surgeries(BaseModel):
    items: list[SurgeryItem] = []


# ---------------------------------------------------------------------------
# Lab results
# ---------------------------------------------------------------------------


class InflammatoryMarkersBlood(BaseModel):
    esr: float | None = None
    esr_unit: str | None = None
    wbc_blood: float | None = None
    wbc_blood_unit: str | None = None
    neut: float | None = None
    neut_unit: str | None = None
    mono: float | None = None
    mono_unit: str | None = None
    crp: float | None = None
    crp_unit: str | None = None
    d_dimer: float | None = None
    d_dimer_unit: str | None = None
    serum_il6: float | None = None
    serum_il6_unit: str | None = None
    alpha_defensin: str | None = None  # POSITIVE / NEGATIVE


class Hematology(BaseModel):
    rbc: float | None = None
    rbc_unit: str | None = None
    ig: float | None = None
    ig_unit: str | None = None
    mcv: float | None = None
    mcv_unit: str | None = None
    mch: float | None = None
    mch_unit: str | None = None
    mchc: float | None = None
    mchc_unit: str | None = None


class SynovialFluid(BaseModel):
    synovial_wbc: float | None = None
    synovial_wbc_unit: str | None = None
    synovial_pmn: float | None = None
    synovial_pmn_unit: str | None = None


class BiochemicalData(BaseModel):
    glucose: float | None = None
    glucose_unit: str | None = None
    ure: float | None = None
    ure_unit: str | None = None
    creatinine: float | None = None
    creatinine_unit: str | None = None
    alt: float | None = None
    alt_unit: str | None = None
    ast: float | None = None
    ast_unit: str | None = None
    albumin: float | None = None
    albumin_unit: str | None = None
    sodium: float | None = None
    sodium_unit: str | None = None
    chlor: float | None = None
    chlor_unit: str | None = None
    kali: float | None = None
    kali_unit: str | None = None
    hba1c: float | None = None
    hba1c_unit: str | None = None


class LatestLab(BaseModel):
    lab_id: int | None = None
    created_at: str | None = None
    inflammatory_markers_blood: InflammatoryMarkersBlood = InflammatoryMarkersBlood()
    hematology: Hematology = Hematology()
    synovial_fluid: SynovialFluid = SynovialFluid()
    biochemical_data: BiochemicalData = BiochemicalData()


class HistoricalTrend(BaseModel):
    created_at: str | None = None
    crp: float | None = None
    esr: float | None = None
    wbc_blood: float | None = None
    synovial_wbc: float | None = None


class LabResults(BaseModel):
    latest: LatestLab = LatestLab()
    historical_trends: list[HistoricalTrend] = []


# ---------------------------------------------------------------------------
# Imaging & Culture
# ---------------------------------------------------------------------------


class ImageItem(BaseModel):
    image_id: int | None = None
    type: str | None = None
    created_at: str | None = None
    findings: str | None = None
    file_metadata: dict | None = None


class ImageResults(BaseModel):
    items: list[ImageItem] = []


class SensitivityItem(BaseModel):
    antibiotic_name: str
    mic_value: str | None = None
    sensitivity_code: str  # S / R / I


class CultureItem(BaseModel):
    culture_id: int | None = None
    sample_type: str | None = None
    name: str | None = None
    organism_name: str | None = None
    result_status: str | None = None  # POSITIVE / NEGATIVE
    gram_type: str | None = None
    incubation_days: int | None = None
    notes: str | None = None
    sensitivities: list[SensitivityItem] = []

    def get_organism(self) -> str | None:
        return self.name or self.organism_name


class CultureResults(BaseModel):
    items: list[CultureItem] = []


# ---------------------------------------------------------------------------
# Root snapshot model
# ---------------------------------------------------------------------------


class SnapshotDataJson(BaseModel):
    """Full snapshot input from web team."""
    snapshot_metadata: SnapshotMetadata
    patient_demographics: PatientDemographics
    medical_history: MedicalHistory = MedicalHistory()
    clinical_records: ClinicalRecords = ClinicalRecords()
    surgeries: Surgeries = Surgeries()
    lab_results: LabResults = LabResults()
    image_results: ImageResults = ImageResults()
    culture_results: CultureResults = CultureResults()
