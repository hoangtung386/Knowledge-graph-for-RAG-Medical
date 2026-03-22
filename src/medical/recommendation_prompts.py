"""Prompts that instruct LLM to generate structured JSON recommendations.

Each prompt targets one recommendation category and enforces the exact
JSON schema that the web team expects. The LLM receives RAG context
(guidelines) + parsed clinical data and must output valid JSON only.
"""

SYSTEM_PROMPT = """\
You are a PJI (Periprosthetic Joint Infection) clinical decision support system.
You MUST respond with valid JSON only — no markdown, no explanation outside JSON.
Base all recommendations on the provided guideline context and clinical data.
Do NOT hallucinate drug names, dosages, or guidelines not present in the context.
If information is insufficient, state so in the relevant notes field.
All text fields should be in Vietnamese unless the field is a medical/English term."""


def build_diagnostic_test_prompt(
    clinical_context: str,
    rag_context: str,
    icm_score_data: dict,
    culture_summary: str,
    risk_factors: list[str],
) -> str:
    """Prompt for DIAGNOSTIC_TEST recommendation."""
    return f"""{SYSTEM_PROMPT}

## Retrieved Guideline Context:
{rag_context}

## Clinical Data:
{clinical_context}

## Culture Results:
{culture_summary}

## Pre-calculated ICM 2018 Score:
{_format_icm_for_prompt(icm_score_data)}

## Patient Risk Factors:
{chr(10).join(f'- {r}' for r in risk_factors) if risk_factors else 'None identified'}

## Task:
Generate a DIAGNOSTIC_TEST JSON object with this exact structure:
{{
  "category": "DIAGNOSTIC_TEST",
  "title": "Chẩn đoán AI - Đánh giá nhiễm trùng khớp giả theo ICM 2018 Scoring",
  "scoring_system": {{
    "name": "ICM 2018 PJI Diagnostic Criteria",
    "version": "2018 (Philadelphia Consensus)",
    "total_score": <int: total minor score>,
    "interpretation": "<INFECTED | INCONCLUSIVE | NOT_INFECTED>",
    "confidence_note": "<string>"
  }},
  "major_criteria": {{
    "items": [
      {{
        "criterion": "<criterion description>",
        "result": <true | false | null>,
        "result_detail": "<detail with actual values>",
        "is_decisive": <true if this alone confirms PJI>
      }}
    ],
    "major_criteria_met": <true | false>,
    "major_criteria_conclusion": "<string>"
  }},
  "minor_criteria_scoring": {{
    "items": [
      {{
        "criterion": "<criterion description>",
        "score_weight": <int>,
        "result": <true | false | null>,
        "result_detail": "<detail with actual values and thresholds>",
        "score_awarded": <int>
      }}
    ],
    "total_minor_score": <int>,
    "total_minor_score_note": "<string>"
  }},
  "ai_reasoning": {{
    "primary_diagnosis": "<string>",
    "infection_classification": "<ACUTE | CHRONIC>",
    "infection_classification_reasoning": "<string>",
    "identified_organism": {{
      "name": "<organism name>",
      "resistance_profile": "<e.g. MRSA>",
      "resistance_detail": "<list sensitive/resistant antibiotics>",
      "biofilm_forming": <true | false | null>,
      "virulence_note": "<string>"
    }},
    "reasoning_summary": "<comprehensive reasoning>",
    "warnings": [
      {{
        "type": "<ALLERGY_ALERT | DRUG_INTERACTION | RENAL_CHECK | HEPATIC_CHECK | NUTRITION_ALERT>",
        "severity": "<HIGH | MEDIUM | LOW>",
        "message": "<string>"
      }}
    ]
  }}
}}

Use the pre-calculated ICM score data. Enrich it with clinical reasoning from guidelines.
Include ALL relevant warnings based on patient risk factors and drug interactions.
Respond with the JSON object only."""


def build_surgery_prompt(
    clinical_context: str,
    rag_context: str,
    culture_summary: str,
    risk_factors: list[str],
) -> str:
    """Prompt for SURGERY_PROCEDURE recommendation."""
    return f"""{SYSTEM_PROMPT}

## Retrieved Guideline Context:
{rag_context}

## Clinical Data:
{clinical_context}

## Culture Results:
{culture_summary}

## Patient Risk Factors:
{chr(10).join(f'- {r}' for r in risk_factors) if risk_factors else 'None identified'}

## Task:
Based on the guideline context and clinical data, generate a SURGERY_PROCEDURE JSON:
{{
  "category": "SURGERY_PROCEDURE",
  "surgery_strategy_type": "<TWO_STAGE_REVISION | ONE_STAGE_REVISION | DAIR | AMPUTATION | SUPPRESSIVE_THERAPY>",
  "strategy_rationale": "<why this strategy, citing guidelines>",
  "priority_level": "<HIGH | MEDIUM | LOW>",
  "priority_note": "<timing recommendation>",
  "stages": [
    {{
      "stage_order": <int>,
      "stage_name": "<stage description>",
      "estimated_duration_minutes": <int or null>,
      "interval_from_stage1": "<string, only for stage 2+>",
      "preconditions": ["<list of conditions that must be met>"]
    }}
  ],
  "estimated_total_treatment_time": "<string>",
  "risks_and_complications": ["<list of risks>"],
  "notes": "<important notes considering patient allergies, comorbidities, risk factors>"
}}

Choose the surgery strategy based on: infection chronicity, implant stability, organism virulence, patient factors.
Respond with the JSON object only."""


def build_systemic_antibiotic_prompt(
    clinical_context: str,
    rag_context: str,
    culture_summary: str,
    risk_factors: list[str],
) -> str:
    """Prompt for SYSTEMIC_ANTIBIOTIC recommendation."""
    return f"""{SYSTEM_PROMPT}

## Retrieved Guideline Context:
{rag_context}

## Clinical Data:
{clinical_context}

## Culture Results:
{culture_summary}

## Patient Risk Factors:
{chr(10).join(f'- {r}' for r in risk_factors) if risk_factors else 'None identified'}

## Task:
Based on the antibiogram and guidelines, generate a SYSTEMIC_ANTIBIOTIC JSON:
{{
  "category": "SYSTEMIC_ANTIBIOTIC",
  "title": "<descriptive title>",
  "regimen_name": "<regimen summary>",
  "indication": "<indication based on organism and clinical data>",
  "total_duration_weeks": <int>,
  "phases": [
    {{
      "phase_name": "<e.g. Giai đoạn tấn công (IV phase)>",
      "phase_order": <int>,
      "duration_weeks": <int>,
      "duration_note": "<when to start, conditions>",
      "antibiotics": [
        {{
          "antibiotic_name": "<name>",
          "dosage": "<dosage>",
          "frequency": "<frequency>",
          "route": "<IV | ORAL | IM>",
          "sequence_order": <int>,
          "role": "<PRIMARY | BIOFILM_AGENT | PRIMARY_ORAL | SYNERGISTIC>",
          "notes": "<MIC values, dose adjustments, monitoring>"
        }}
      ]
    }}
  ],
  "monitoring": ["<list of monitoring items with frequency>"],
  "contraindications": ["<list considering patient allergies and comorbidities>"],
  "notes": "<important drug interactions, patient-specific considerations>"
}}

CRITICAL RULES:
- Choose antibiotics based on ACTUAL sensitivity results from the antibiogram.
- NEVER recommend antibiotics the patient is allergic to or cross-reactive drugs.
- Rifampicin must NEVER be used as monotherapy.
- Include dose adjustments for renal/hepatic function if impaired.
- Consider drug interactions with patient's current medications.
Respond with the JSON object only."""


def build_local_antibiotic_prompt(
    clinical_context: str,
    rag_context: str,
    culture_summary: str,
    risk_factors: list[str],
) -> str:
    """Prompt for LOCAL_ANTIBIOTIC recommendation."""
    return f"""{SYSTEM_PROMPT}

## Retrieved Guideline Context:
{rag_context}

## Clinical Data:
{clinical_context}

## Culture Results:
{culture_summary}

## Patient Risk Factors:
{chr(10).join(f'- {r}' for r in risk_factors) if risk_factors else 'None identified'}

## Task:
Based on the organism sensitivity and guidelines, generate a LOCAL_ANTIBIOTIC JSON:
{{
  "category": "LOCAL_ANTIBIOTIC",
  "title": "<descriptive title>",
  "regimen_name": "<regimen name>",
  "indication": "<indication>",
  "duration_days": <int or null>,
  "duration_note": "<duration details>",
  "delivery_info": {{
    "delivery_method": "<CEMENT_SPACER | ANTIBIOTIC_BEAD | CALCIUM_SULFATE>",
    "spacer_type": "<ARTICULATING | STATIC | null>",
    "cement_brand_suggestion": "<string or null>",
    "mixing_ratio": "<antibiotic per cement ratio>"
  }},
  "antibiotics": [
    {{
      "antibiotic_name": "<name (trong xi măng)>",
      "dosage": "<dose per 40g cement>",
      "frequency": "<Elution liên tục từ spacer>",
      "route": "<LOCAL_CEMENT | LOCAL_BEAD>",
      "sequence_order": <int>,
      "role": "<PRIMARY | SYNERGISTIC>",
      "notes": "<based on antibiogram MIC values>"
    }}
  ],
  "monitoring": ["<list of monitoring items>"],
  "contraindications": ["<list including PMMA allergy check>"],
  "notes": "<patient-specific notes>"
}}

Choose local antibiotics based on the antibiogram. Consider PMMA compatibility.
Respond with the JSON object only."""


def build_citations_prompt(
    rag_context: str,
    recommendations_summary: str,
) -> str:
    """Prompt to generate citations linking evidence to recommendations."""
    return f"""{SYSTEM_PROMPT}

## Retrieved Guideline Context:
{rag_context}

## Recommendations Generated:
{recommendations_summary}

## Task:
Generate a JSON array of citations that link the guideline evidence to the recommendations.
Each citation should reference a specific guideline, study, or consensus statement.

[
  {{
    "source_type": "<GUIDELINE | META_ANALYSIS | JOURNAL_ARTICLE | CONSENSUS_STATEMENT | SYSTEMATIC_REVIEW>",
    "source_title": "<full title of the source>",
    "source_uri": "<DOI or URL if known, null if unknown>",
    "snippet": "<relevant quote or summary from the source>",
    "relevance_score": <float 0.0-1.0>,
    "cited_for": "<what this citation supports in the recommendations>"
  }}
]

Rules:
- Only cite sources that appear in the Retrieved Guideline Context.
- Do NOT fabricate citations or DOIs not present in the context.
- If you cannot identify the exact source, use the closest match from context.
- Include at least one citation per recommendation category.
Respond with the JSON array only."""


def _format_icm_for_prompt(icm_data: dict) -> str:
    """Format ICM score dict for prompt injection."""
    lines = [
        f"Total Score: {icm_data.get('total_score', 'N/A')}",
        f"Interpretation: {icm_data.get('interpretation', 'N/A')}",
        f"Decision Basis: {icm_data.get('decision_basis', 'N/A')}",
        f"Setting: {icm_data.get('setting', 'N/A')}",
    ]
    for f in icm_data.get("findings", []):
        marker = f.get("marker", "")
        value = f.get("value", "")
        unit = f.get("unit", "")
        threshold = f.get("threshold", "")
        status = f.get("status", "")
        points = f.get("points", 0)
        lines.append(f"  {marker}: {value} {unit} (threshold: {threshold}, {status}, +{points}pts)")
    return "\n".join(lines)
