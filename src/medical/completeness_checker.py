"""Deterministic data completeness checker.

Analyzes the snapshot to identify missing clinical data that would
improve diagnostic accuracy. No LLM needed — pure rule-based logic.
"""

from __future__ import annotations

from src.api.schemas import SnapshotDataJson
from src.api.schemas.output import DataCompletenessReport, MissingDataItem


# ---------------------------------------------------------------------------
# ICM 2018 criteria field mapping
# ---------------------------------------------------------------------------

# Minor criteria: marker_id -> (snapshot path description, importance)
_ICM_MINOR_FIELDS = [
    ("serum_CRP", "lab_results.latest.inflammatory_markers_blood.crp", "Serum CRP", "HIGH"),
    ("serum_D_dimer", "lab_results.latest.inflammatory_markers_blood.d_dimer", "Serum D-Dimer", "HIGH"),
    ("serum_ESR", "lab_results.latest.inflammatory_markers_blood.esr", "Serum ESR", "MEDIUM"),
    ("synovial_WBC", "lab_results.latest.synovial_fluid.synovial_wbc", "Synovial WBC count", "CRITICAL"),
    ("synovial_PMN", "lab_results.latest.synovial_fluid.synovial_pmn", "Synovial PMN%", "HIGH"),
    ("synovial_alpha_defensin", "lab_results.latest.inflammatory_markers_blood.alpha_defensin", "Alpha-Defensin", "HIGH"),
]

# Clinical fields important for recommendation quality
_CLINICAL_FIELDS = [
    ("infection_type", "Suspected infection type (ACUTE/CHRONIC)", "HIGH"),
    ("implant_stability", "Implant stability (STABLE/UNSTABLE)", "HIGH"),
    ("prosthesis_joint", "Prosthesis joint location", "MEDIUM"),
    ("days_since_arthroplasty", "Days since index arthroplasty", "MEDIUM"),
]


def check_data_completeness(snapshot: SnapshotDataJson) -> DataCompletenessReport:
    """Check snapshot for missing data that affects diagnostic accuracy.

    Returns a deterministic report — no LLM involved.
    """
    missing = []
    lab = snapshot.lab_results.latest
    blood = lab.inflammatory_markers_blood
    synovial = lab.synovial_fluid
    assess = snapshot.clinical_records.infection_assessment

    # --- ICM Minor Criteria ---
    icm_available = 0
    icm_total = len(_ICM_MINOR_FIELDS)

    _check_icm_field(missing, blood.crp, "serum_CRP", "Serum CRP", "HIGH",
                     "CRP huyết thanh (2 điểm ICM) — cần thiết cho scoring viêm")
    _check_icm_field(missing, blood.d_dimer, "serum_D_dimer", "Serum D-Dimer", "HIGH",
                     "D-Dimer huyết thanh (2 điểm ICM) — marker đông máu/viêm")
    _check_icm_field(missing, blood.esr, "serum_ESR", "Serum ESR", "MEDIUM",
                     "Tốc độ máu lắng (1 điểm ICM, chỉ dùng cho chronic)")
    _check_icm_field(missing, synovial.synovial_wbc, "synovial_WBC", "Synovial WBC", "CRITICAL",
                     "Bạch cầu dịch khớp (3 điểm ICM) — marker quan trọng nhất")
    _check_icm_field(missing, synovial.synovial_pmn, "synovial_PMN", "Synovial PMN%", "HIGH",
                     "PMN% dịch khớp (2 điểm ICM)")
    _check_icm_field(missing, blood.alpha_defensin, "synovial_alpha_defensin", "Alpha-Defensin", "HIGH",
                     "Alpha-Defensin (3 điểm ICM) — độ đặc hiệu >96%")

    icm_available = icm_total - sum(1 for m in missing if m.category == "ICM_MINOR")

    # Histology — not in lab_results, always missing from snapshot
    missing.append(MissingDataItem(
        field="positive_histology",
        category="ICM_MINOR",
        importance="HIGH",
        message="Giải phẫu bệnh mô quanh khớp (3 điểm ICM) — cần sinh thiết mô trong phẫu thuật",
    ))

    # Leukocyte Esterase — not in lab_results schema
    missing.append(MissingDataItem(
        field="synovial_LE",
        category="ICM_MINOR",
        importance="MEDIUM",
        message="Leukocyte Esterase dịch khớp (3 điểm ICM) — test nhanh tại chỗ",
    ))

    # Synovial CRP — not in lab_results schema
    missing.append(MissingDataItem(
        field="synovial_CRP",
        category="ICM_MINOR",
        importance="MEDIUM",
        message="CRP dịch khớp (1 điểm ICM)",
    ))

    # --- ICM Major Criteria data ---
    cultures = snapshot.culture_results.items
    if not cultures:
        missing.append(MissingDataItem(
            field="culture_results",
            category="ICM_MAJOR",
            importance="CRITICAL",
            message="Không có kết quả nuôi cấy — cần ít nhất 2 mẫu để đánh giá major criterion",
        ))
    elif len([c for c in cultures if c.result_status == "POSITIVE"]) == 0:
        # Cultures exist but none positive — that's data, not missing
        pass
    # Check if any positive culture lacks sensitivity data
    for c in cultures:
        if c.result_status == "POSITIVE" and not c.sensitivities:
            organism = c.get_organism() or "unknown"
            sample = c.sample_type or "unknown"
            missing.append(MissingDataItem(
                field=f"sensitivities_{organism}_{sample}",
                category="CLINICAL",
                importance="CRITICAL",
                message=f"Kháng sinh đồ chưa có cho {organism} ({sample}) — không thể chọn kháng sinh chính xác",
            ))

    # --- Clinical assessment fields ---
    if not assess.suspected_infection_type and assess.days_since_index_arthroplasty is None:
        missing.append(MissingDataItem(
            field="infection_type",
            category="CLINICAL",
            importance="HIGH",
            message="Chưa xác định ACUTE/CHRONIC — ảnh hưởng ngưỡng ICM và chiến lược phẫu thuật",
        ))

    if not assess.implant_stability:
        missing.append(MissingDataItem(
            field="implant_stability",
            category="CLINICAL",
            importance="HIGH",
            message="Chưa đánh giá ổn định implant — ảnh hưởng chọn DAIR vs revision",
        ))

    # --- Build summary ---
    icm_total_possible = 10  # 2 major + 8 minor in our schema
    completeness = f"{icm_available}/{len(_ICM_MINOR_FIELDS)} ICM minor criteria có dữ liệu"

    critical_count = sum(1 for m in missing if m.importance == "CRITICAL")
    high_count = sum(1 for m in missing if m.importance == "HIGH")

    if critical_count > 0:
        impact = (
            f"Thiếu {critical_count} dữ liệu CRITICAL — "
            "kết quả chẩn đoán và phác đồ có thể không chính xác. "
            "Khuyến nghị bổ sung trước khi áp dụng."
        )
    elif high_count > 0:
        impact = (
            f"Thiếu {high_count} dữ liệu quan trọng — "
            "kết quả vẫn có giá trị tham khảo nhưng nên bổ sung để tăng độ tin cậy."
        )
    else:
        impact = "Dữ liệu đầy đủ cho chẩn đoán ICM 2018."

    # Filter out the "always missing" items (histology, LE, synovial CRP)
    # that are structural — web team can't provide them
    actionable_missing = [
        m for m in missing
        if m.field not in ("positive_histology", "synovial_LE", "synovial_CRP")
    ]

    return DataCompletenessReport(
        is_complete=len(actionable_missing) == 0,
        missing_items=missing,
        completeness_score=completeness,
        impact_note=impact,
    )


def _check_icm_field(
    missing: list[MissingDataItem],
    value,
    field_id: str,
    field_name: str,
    importance: str,
    message: str,
):
    """Add to missing list if value is None."""
    if value is None:
        missing.append(MissingDataItem(
            field=field_id,
            category="ICM_MINOR",
            importance=importance,
            message=message,
        ))
