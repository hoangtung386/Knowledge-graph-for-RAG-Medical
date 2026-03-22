"""PJI diagnostic criteria scoring based on ICM 2018 guidelines.

References:
- Parvizi J, et al. The 2018 Definition of Periprosthetic Joint Infection.
  J Arthroplasty. 2018;33(5):1309-1314.
"""

from dataclasses import dataclass
from enum import Enum

from src.core.config import QUALITATIVE_POSITIVE_VALUES


class PJIResult(str, Enum):
    INFECTED = "PJI confirmed"
    INCONCLUSIVE = "PJI inconclusive - further workup needed"
    NOT_INFECTED = "PJI unlikely"


# ---------------------------------------------------------------------------
# ICM 2018 Major Criteria
# Any ONE major criterion = PJI confirmed (regardless of score)
# ---------------------------------------------------------------------------
MAJOR_CRITERIA = [
    {
        "id": "sinus_tract",
        "name": "Sinus tract communicating with the joint",
        "description": "Sinus tract communicating with the prosthetic joint or visualization of the prosthesis",
    },
    {
        "id": "two_cultures",
        "name": "Two positive cultures with phenotypically identical organisms",
        "description": "Same pathogen isolated from two or more separate tissue/fluid samples",
    },
]


# ---------------------------------------------------------------------------
# ICM 2018 Minor Criteria (Weighted Scoring)
# Threshold values for acute (<90 days) and chronic (>=90 days) PJI
# ---------------------------------------------------------------------------
@dataclass
class MinorCriterion:
    id: str
    name: str
    unit: str
    threshold_acute: float | str | None
    threshold_chronic: float | str | None
    score: int  # Points if threshold met


MINOR_CRITERIA: list[MinorCriterion] = [
    # Serum markers
    MinorCriterion(
        id="serum_CRP", name="Serum CRP", unit="mg/L",
        threshold_acute=100, threshold_chronic=10, score=2,
    ),
    MinorCriterion(
        id="serum_D_dimer", name="Serum D-Dimer", unit="ng/mL",
        threshold_acute=860, threshold_chronic=860, score=2,
    ),
    MinorCriterion(
        id="serum_ESR", name="Serum ESR", unit="mm/hr",
        threshold_acute=None, threshold_chronic=30, score=1,
    ),
    # Synovial markers
    MinorCriterion(
        id="synovial_WBC", name="Synovial WBC count", unit="cells/uL",
        threshold_acute=10000, threshold_chronic=3000, score=3,
    ),
    MinorCriterion(
        id="synovial_PMN", name="Synovial PMN%", unit="%",
        threshold_acute=90, threshold_chronic=80, score=2,
    ),
    MinorCriterion(
        id="synovial_alpha_defensin", name="Synovial Alpha-Defensin", unit="",
        threshold_acute="positive", threshold_chronic="positive", score=3,
    ),
    MinorCriterion(
        id="synovial_LE", name="Synovial Leukocyte Esterase", unit="",
        threshold_acute="++", threshold_chronic="++", score=3,
    ),
    MinorCriterion(
        id="synovial_CRP", name="Synovial CRP", unit="mg/L",
        threshold_acute=6.9, threshold_chronic=6.9, score=1,
    ),
    # Histology / tissue
    MinorCriterion(
        id="positive_histology", name="Positive histological analysis (>5 PMN/HPF)",
        unit="PMN/HPF", threshold_acute=5, threshold_chronic=5, score=3,
    ),
    MinorCriterion(
        id="single_culture", name="Single positive culture", unit="",
        threshold_acute="positive", threshold_chronic="positive", score=2,
    ),
]

MINOR_CRITERIA_MAP: dict[str, MinorCriterion] = {c.id: c for c in MINOR_CRITERIA}


class ICM2018Criteria:
    """ICM 2018 PJI scoring calculator.

    Decision logic:
    - Any major criterion present -> PJI confirmed
    - Minor criteria total >= 6  -> PJI confirmed
    - Minor criteria 2-5         -> Inconclusive
    - Minor criteria < 2         -> PJI unlikely
    """

    CONFIRMED_THRESHOLD = 6
    INCONCLUSIVE_THRESHOLD = 2

    def score(
        self,
        lab_values: dict,
        major_criteria_present: list[str] | None = None,
        is_acute: bool = False,
    ) -> dict:
        """Calculate ICM 2018 PJI score.

        Args:
            lab_values: Dict of marker_id -> value.
            major_criteria_present: List of major criteria IDs that are present.
            is_acute: True if infection is acute (<90 days post-op).

        Returns:
            Dict with total_score, interpretation, findings, and details.
        """
        setting = "acute" if is_acute else "chronic"

        # Check major criteria first
        if major_criteria_present:
            findings = self._evaluate_major_criteria(major_criteria_present)
            return {
                "total_score": "N/A (major criterion)",
                "interpretation": PJIResult.INFECTED.value,
                "decision_basis": "major_criterion",
                "findings": findings,
                "guideline": "ICM 2018",
                "setting": setting,
            }

        # Score minor criteria
        findings, total_score = self._evaluate_minor_criteria(lab_values, is_acute)
        return {
            "total_score": total_score,
            "interpretation": self._interpret(total_score),
            "decision_basis": "minor_criteria_score",
            "findings": findings,
            "guideline": "ICM 2018",
            "setting": setting,
        }

    @staticmethod
    def _evaluate_major_criteria(criteria_ids: list[str]) -> list[dict]:
        findings = []
        for mc_id in criteria_ids:
            for mc in MAJOR_CRITERIA:
                if mc["id"] == mc_id:
                    findings.append({
                        "marker": mc["name"],
                        "status": "MAJOR CRITERION PRESENT",
                        "detail": mc["description"],
                    })
        return findings

    @staticmethod
    def _evaluate_minor_criteria(
        lab_values: dict, is_acute: bool
    ) -> tuple[list[dict], int]:
        findings = []
        total_score = 0

        for criterion in MINOR_CRITERIA:
            if criterion.id not in lab_values:
                continue

            value = lab_values[criterion.id]
            threshold = criterion.threshold_acute if is_acute else criterion.threshold_chronic

            if threshold is None:
                findings.append({
                    "marker": criterion.name,
                    "value": value,
                    "unit": criterion.unit,
                    "status": "not applicable in this setting",
                    "points": 0,
                })
                continue

            is_positive = _check_threshold(value, threshold)
            points = criterion.score if is_positive else 0
            total_score += points

            findings.append({
                "marker": criterion.name,
                "value": value,
                "unit": criterion.unit,
                "threshold": threshold,
                "status": "positive" if is_positive else "negative",
                "points": points,
            })

        return findings, total_score

    def _interpret(self, score: int) -> str:
        if score >= self.CONFIRMED_THRESHOLD:
            return PJIResult.INFECTED.value
        if score >= self.INCONCLUSIVE_THRESHOLD:
            return PJIResult.INCONCLUSIVE.value
        return PJIResult.NOT_INFECTED.value


def _check_threshold(value, threshold) -> bool:
    """Check if a lab value meets the threshold (numeric or qualitative)."""
    if isinstance(threshold, str):
        return str(value).strip().lower() in QUALITATIVE_POSITIVE_VALUES
    return float(value) >= float(threshold)


def format_icm_summary(icm_result: dict) -> str:
    """Format ICM score result into readable text for prompts."""
    lines = [
        f"Guideline: {icm_result['guideline']}",
        f"Setting: {icm_result['setting']}",
        f"Total Score: {icm_result['total_score']}",
        f"Interpretation: {icm_result['interpretation']}",
        f"Decision Basis: {icm_result['decision_basis']}",
        "",
        "Findings:",
    ]
    for f in icm_result.get("findings", []):
        marker = f["marker"]
        status = f["status"]
        points = f.get("points", "N/A")
        value = f.get("value", "")
        unit = f.get("unit", "")
        threshold = f.get("threshold", "")
        lines.append(
            f"  - {marker}: {value} {unit} "
            f"(threshold: {threshold}, status: {status}, points: {points})"
        )
    return "\n".join(lines)
