from __future__ import annotations

from typing import Any

from app.schemas import MedicationAdherenceRiskEnum, PatientAssessmentRequest, RiskLevelEnum


def map_score_to_risk_level(score: float) -> RiskLevelEnum:
    if score < 0.34:
        return RiskLevelEnum.low
    if score < 0.67:
        return RiskLevelEnum.medium
    return RiskLevelEnum.high


def mock_call_autoai(payload: PatientAssessmentRequest) -> dict[str, Any]:
    score = 0.10

    score += min(payload.prior_admissions_12m * 0.15, 0.45)
    score += min(payload.comorbidity_count * 0.05, 0.20)
    score += 0.10 if not payload.follow_up_scheduled else 0.0
    score += (
        0.08 if payload.medication_adherence_risk == MedicationAdherenceRiskEnum.high
        else 0.03 if payload.medication_adherence_risk == MedicationAdherenceRiskEnum.medium
        else 0.0
    )
    score += 0.05 if payload.diabetes else 0.0
    score += 0.05 if payload.hypertension else 0.0

    score = max(0.0, min(score, 0.99))
    risk_level = map_score_to_risk_level(score)

    return {
        "risk_score": round(score, 2),
        "risk_level": risk_level.value,
    }