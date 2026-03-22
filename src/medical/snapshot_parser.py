"""Parse snapshot_data_json into structured data for ICM scoring and RAG queries.

Extracts:
- Lab values mapped to ICM criteria marker IDs
- Clinical context as natural language for RAG retrieval
- Culture/sensitivity data for antibiotic recommendation
- Patient risk factors for warnings
"""

from __future__ import annotations

from collections import Counter

from src.api.schemas import SnapshotDataJson
from src.core.config import (
    ACUTE_INFECTION_DAYS_THRESHOLD,
    ANEMIA_RBC_THRESHOLD,
    D_DIMER_MGL_TO_NGML,
    DIABETES_HBA1C_THRESHOLD,
    HEPATIC_ALT_THRESHOLD,
    HEPATIC_AST_THRESHOLD,
    HYPERGLYCEMIA_GLUCOSE_THRESHOLD,
    HYPERKALEMIA_THRESHOLD,
    MALNUTRITION_ALBUMIN_THRESHOLD,
    RENAL_CREATININE_THRESHOLD,
)


def extract_icm_lab_values(snapshot: SnapshotDataJson) -> dict:
    """Map snapshot lab fields to ICM 2018 marker IDs used by pji_criteria.py."""
    lab = snapshot.lab_results.latest
    blood = lab.inflammatory_markers_blood
    synovial = lab.synovial_fluid

    values = {}

    if blood.crp is not None:
        values["serum_CRP"] = blood.crp
    if blood.d_dimer is not None:
        values["serum_D_dimer"] = blood.d_dimer * D_DIMER_MGL_TO_NGML
    if blood.esr is not None:
        values["serum_ESR"] = blood.esr
    if synovial.synovial_wbc is not None:
        values["synovial_WBC"] = synovial.synovial_wbc
    if synovial.synovial_pmn is not None:
        values["synovial_PMN"] = synovial.synovial_pmn
    if blood.alpha_defensin is not None:
        values["synovial_alpha_defensin"] = blood.alpha_defensin.lower()

    return values


def extract_major_criteria(snapshot: SnapshotDataJson) -> list[str]:
    """Detect major criteria from snapshot data."""
    criteria = []

    if snapshot.clinical_records.symptoms.sinus_tract:
        criteria.append("sinus_tract")

    positive_cultures = [
        c for c in snapshot.culture_results.items
        if c.result_status == "POSITIVE"
    ]
    if len(positive_cultures) >= 2:
        organisms = [c.get_organism() for c in positive_cultures]
        organisms = [o for o in organisms if o]
        counts = Counter(organisms)
        if any(v >= 2 for v in counts.values()):
            criteria.append("two_cultures")

    return criteria


def determine_is_acute(snapshot: SnapshotDataJson) -> bool:
    """Determine if infection is acute (<90 days) or chronic."""
    assessment = snapshot.clinical_records.infection_assessment

    if assessment.suspected_infection_type:
        return assessment.suspected_infection_type.upper() == "ACUTE"

    if assessment.days_since_index_arthroplasty is not None:
        return assessment.days_since_index_arthroplasty < ACUTE_INFECTION_DAYS_THRESHOLD

    return False


def build_clinical_context(snapshot: SnapshotDataJson) -> str:
    """Build natural language clinical context for RAG queries.

    Includes ALL clinically relevant input fields so LLM has complete picture.
    """
    sections = [
        _build_patient_section(snapshot),
        _build_infection_assessment_section(snapshot),
        _build_lab_section(snapshot),
        _build_history_section(snapshot),
    ]
    return "\n".join(s for s in sections if s)


def _build_patient_section(snapshot: SnapshotDataJson) -> str:
    """Patient demographics, history, allergies, substance use, symptoms, vitals."""
    demo = snapshot.patient_demographics
    hist = snapshot.medical_history
    clinical = snapshot.clinical_records
    lines = []

    # Demographics
    lines.append(f"Patient: {demo.gender}, DOB {demo.date_of_birth}")
    if hist.medical_history:
        lines.append(f"Medical history: {hist.medical_history}")
    if hist.process:
        lines.append(f"Current illness: {hist.process}")
    if clinical.illness_onset_date:
        lines.append(f"Illness onset date: {clinical.illness_onset_date}")

    # Allergies
    if hist.allergies.is_allergy and hist.allergies.allergy_note:
        lines.append(f"ALLERGY: {hist.allergies.allergy_note}")

    # Substance use — all fields
    su = hist.substance_use
    if su.is_smoking:
        lines.append(f"Smoking: yes{f' ({su.smoking_note})' if su.smoking_note else ''}")
    if su.is_alcohol:
        lines.append(f"Alcohol: {su.alcohol_note or 'yes'}")
    if su.is_drug:
        lines.append(f"Drug use: {su.drug_note or 'yes'}")
    if su.is_other and su.other_note:
        lines.append(f"Other substance/comorbidity note: {su.other_note}")

    # Symptoms
    symptom_map = {
        "fever": clinical.symptoms.fever,
        "pain": clinical.symptoms.pain,
        "erythema": clinical.symptoms.erythema,
        "swelling": clinical.symptoms.swelling,
        "sinus tract": clinical.symptoms.sinus_tract,
    }
    active = [name for name, present in symptom_map.items() if present]
    if active:
        lines.append(f"Symptoms: {', '.join(active)}")

    # Vitals
    if clinical.vitals.blood_pressure:
        lines.append(f"BP: {clinical.vitals.blood_pressure}")
    if clinical.vitals.bmi:
        lines.append(f"BMI: {clinical.vitals.bmi}")

    # Physical exam
    if clinical.notations:
        lines.append(f"Physical exam: {clinical.notations}")

    return "\n".join(lines)


def _build_infection_assessment_section(snapshot: SnapshotDataJson) -> str:
    """Infection assessment — all fields including hematogenous and soft tissue."""
    assess = snapshot.clinical_records.infection_assessment
    lines = []

    if assess.prosthesis_joint:
        lines.append(f"Prosthesis: {assess.prosthesis_joint}")
    if assess.days_since_index_arthroplasty is not None:
        lines.append(f"Days since index arthroplasty: {assess.days_since_index_arthroplasty}")
    if assess.suspected_infection_type:
        lines.append(f"Suspected infection type: {assess.suspected_infection_type}")
    if assess.implant_stability:
        lines.append(f"Implant stability: {assess.implant_stability}")
    if assess.hematogenous_suspected:
        lines.append("Hematogenous infection suspected: YES")
    if assess.soft_tissue:
        lines.append(f"Soft tissue status: {assess.soft_tissue}")
    if assess.pmma_allergy:
        lines.append("PMMA allergy: YES — cement spacer contraindicated")

    return "\n--- Infection Assessment ---\n" + "\n".join(lines) if lines else ""


def _build_lab_section(snapshot: SnapshotDataJson) -> str:
    """Lab results — ALL markers including hematology and electrolytes."""
    lab = snapshot.lab_results.latest
    blood = lab.inflammatory_markers_blood
    synovial = lab.synovial_fluid
    biochem = lab.biochemical_data
    hema = lab.hematology
    lines = ["\n--- Lab Results ---"]

    # Inflammatory markers (blood)
    _append_if(lines, blood.crp, "Serum CRP", blood.crp_unit or "mg/L")
    _append_if(lines, blood.esr, "ESR", blood.esr_unit or "mm/h")
    _append_if(lines, blood.d_dimer, "D-Dimer", blood.d_dimer_unit or "mg/L FEU")
    _append_if(lines, blood.serum_il6, "Serum IL-6", blood.serum_il6_unit or "pg/mL")
    if blood.alpha_defensin:
        lines.append(f"Alpha-Defensin: {blood.alpha_defensin}")
    _append_if(lines, blood.wbc_blood, "Blood WBC", blood.wbc_blood_unit or "x10^9/L")
    _append_if(lines, blood.neut, "Neutrophil%", blood.neut_unit or "%")
    _append_if(lines, blood.mono, "Monocyte%", blood.mono_unit or "%")

    # Synovial fluid
    _append_if(lines, synovial.synovial_wbc, "Synovial WBC", synovial.synovial_wbc_unit or "cells/μL")
    if synovial.synovial_pmn is not None:
        lines.append(f"Synovial PMN%: {synovial.synovial_pmn}{synovial.synovial_pmn_unit or '%'}")

    # Hematology (important for surgical planning — anemia detection)
    _append_if(lines, hema.rbc, "RBC", hema.rbc_unit or "x10^12/L")
    _append_if(lines, hema.mcv, "MCV", hema.mcv_unit or "fL")
    _append_if(lines, hema.mch, "MCH", hema.mch_unit or "pg")
    _append_if(lines, hema.mchc, "MCHC", hema.mchc_unit or "g/L")
    _append_if(lines, hema.ig, "Immature granulocytes", hema.ig_unit or "%")

    # Biochemical — ALL fields
    _append_if(lines, biochem.glucose, "Glucose", biochem.glucose_unit or "mmol/L")
    _append_if(lines, biochem.creatinine, "Creatinine", biochem.creatinine_unit or "μmol/L")
    _append_if(lines, biochem.ure, "Ure/BUN", biochem.ure_unit or "mmol/L")
    _append_if(lines, biochem.albumin, "Albumin", biochem.albumin_unit or "g/L")
    if biochem.hba1c is not None:
        lines.append(f"HbA1c: {biochem.hba1c}{biochem.hba1c_unit or '%'}")
    _append_if(lines, biochem.alt, "ALT", biochem.alt_unit or "U/L")
    _append_if(lines, biochem.ast, "AST", biochem.ast_unit or "U/L")
    _append_if(lines, biochem.sodium, "Sodium", biochem.sodium_unit or "mmol/L")
    _append_if(lines, biochem.kali, "Potassium (K+)", biochem.kali_unit or "mmol/L")
    _append_if(lines, biochem.chlor, "Chloride", biochem.chlor_unit or "mmol/L")

    return "\n".join(lines) if len(lines) > 1 else ""


def _build_history_section(snapshot: SnapshotDataJson) -> str:
    """Historical trends, imaging, surgical history."""
    lines = []

    # Historical trends
    if snapshot.lab_results.historical_trends:
        lines.append("\n--- Historical Trends ---")
        for trend in snapshot.lab_results.historical_trends:
            parts = [f"Date: {trend.created_at}"]
            if trend.crp is not None:
                parts.append(f"CRP={trend.crp}")
            if trend.esr is not None:
                parts.append(f"ESR={trend.esr}")
            if trend.wbc_blood is not None:
                parts.append(f"WBC={trend.wbc_blood}")
            if trend.synovial_wbc is not None:
                parts.append(f"Synovial WBC={trend.synovial_wbc}")
            lines.append(", ".join(parts))

    # Imaging
    if snapshot.image_results.items:
        lines.append("\n--- Imaging ---")
        for img in snapshot.image_results.items:
            lines.append(f"{img.type}: {img.findings}")

    # Surgeries
    if snapshot.surgeries.items:
        lines.append("\n--- Surgical History ---")
        for s in snapshot.surgeries.items:
            lines.append(f"{s.surgery_date} - {s.surgery_type}: {s.findings}")

    return "\n".join(lines)


def _append_if(lines: list[str], value, label: str, unit: str):
    """Append a lab line only if the value is not None."""
    if value is not None:
        lines.append(f"{label}: {value} {unit}")


def build_culture_summary(snapshot: SnapshotDataJson) -> str:
    """Build culture and sensitivity summary for antibiotic recommendation."""
    cultures = snapshot.culture_results.items
    if not cultures:
        return "No culture results available."

    lines = []
    for c in cultures:
        organism = c.get_organism() or "Unknown"
        status = c.result_status or "Unknown"
        lines.append(f"Culture ({c.sample_type}): {organism} - {status}")
        if c.gram_type:
            lines.append(f"  Gram: {c.gram_type}")
        if c.incubation_days:
            lines.append(f"  Incubation: {c.incubation_days} days")
        if c.notes:
            lines.append(f"  Note: {c.notes}")
        if c.sensitivities:
            lines.append("  Antibiogram:")
            for s in c.sensitivities:
                mic = f" (MIC {s.mic_value})" if s.mic_value else ""
                lines.append(f"    - {s.antibiotic_name}: {s.sensitivity_code}{mic}")
    return "\n".join(lines)


def extract_patient_risk_factors(snapshot: SnapshotDataJson) -> list[str]:
    """Extract risk factors for warnings generation."""
    risks = []
    hist = snapshot.medical_history
    biochem = snapshot.lab_results.latest.biochemical_data
    hema = snapshot.lab_results.latest.hematology

    # Allergies
    if hist.allergies.is_allergy and hist.allergies.allergy_note:
        risks.append(f"ALLERGY: {hist.allergies.allergy_note}")

    # PMMA (bone cement) allergy — critical for local antibiotic spacer decisions
    if snapshot.clinical_records.infection_assessment.pmma_allergy:
        risks.append("PMMA_ALLERGY: Patient allergic to polymethylmethacrylate (bone cement)")

    # Lab-based risk factors
    if biochem.hba1c is not None and biochem.hba1c > DIABETES_HBA1C_THRESHOLD:
        risks.append(f"Diabetes (HbA1c {biochem.hba1c}%)")
    if biochem.glucose is not None and biochem.glucose > HYPERGLYCEMIA_GLUCOSE_THRESHOLD:
        risks.append(f"Hyperglycemia (Glucose {biochem.glucose} mmol/L)")
    if biochem.creatinine is not None and biochem.creatinine > RENAL_CREATININE_THRESHOLD:
        risks.append(f"Impaired renal function (Creatinine {biochem.creatinine} μmol/L)")
    if biochem.alt is not None and biochem.alt > HEPATIC_ALT_THRESHOLD:
        risks.append(f"Elevated ALT ({biochem.alt} U/L)")
    if biochem.ast is not None and biochem.ast > HEPATIC_AST_THRESHOLD:
        risks.append(f"Elevated AST ({biochem.ast} U/L)")
    if biochem.albumin is not None and biochem.albumin < MALNUTRITION_ALBUMIN_THRESHOLD:
        risks.append(f"Low albumin ({biochem.albumin} g/L) - malnutrition risk")
    if biochem.kali is not None and biochem.kali > HYPERKALEMIA_THRESHOLD:
        risks.append(f"Hyperkalemia (K+ {biochem.kali} mmol/L) - caution with TMP-SMX")

    # Anemia detection (RBC or MCHC low)
    if hema.rbc is not None and hema.rbc < ANEMIA_RBC_THRESHOLD:
        risks.append(f"Anemia (RBC {hema.rbc} x10^12/L) - pre-operative optimization needed")

    # Substance use
    if hist.substance_use.is_smoking:
        risks.append(f"Smoker{f': {hist.substance_use.smoking_note}' if hist.substance_use.smoking_note else ''}")
    if hist.substance_use.is_alcohol:
        risks.append(f"Alcohol use: {hist.substance_use.alcohol_note or 'yes'}")
    if hist.substance_use.is_drug:
        risks.append(f"Drug use: {hist.substance_use.drug_note or 'yes'}")

    # Medical history keywords
    if hist.medical_history:
        _extract_history_risks(hist.medical_history, risks)

    return risks


def _extract_history_risks(medical_history: str, risks: list[str]):
    """Detect risk factors from medical history free text."""
    text = medical_history.lower()
    keyword_risks = [
        (("tiểu đường", "diabetes"), "Diabetes mellitus (from history)"),
        (("huyết áp", "hypertension"), "Hypertension (from history)"),
        (("immunosuppres", "ức chế miễn dịch"), "Immunosuppression (from history)"),
    ]
    for keywords, risk_label in keyword_risks:
        if any(kw in text for kw in keywords):
            risks.append(risk_label)
