from __future__ import annotations

import json
from typing import Any

from app.schemas import MedicationAdherenceRiskEnum, PatientAssessmentRequest, PromptLabOutput


PROMPT_TEMPLATE = """You are a clinical risk explanation assistant.

Your task is to generate a concise readmission risk explanation based only on the provided patient data and the model-generated risk score and risk level.

Use the provided risk_score and risk_level as the source of truth.
Do not change them.
Do not add medical facts that are not supported by the input.
Do not include warnings, disclaimers, markdown, or extra commentary.

Return strict JSON only with exactly these keys:
- summary
- key_factors
- recommended_actions

Requirements:
- summary must be 1 to 3 sentences
- key_factors must be an array of exactly 3 short strings
- recommended_actions must be an array of exactly 3 short strings
- do not include any extra keys
- do not wrap the JSON in code fences

Input:
{
  "patient_age": "{patient_age}",
  "patient_sex": "{patient_sex}",
  "prior_admissions_12m": "{prior_admissions_12m}",
  "length_of_last_stay": "{length_of_last_stay}",
  "comorbidity_count": "{comorbidity_count}",
  "diabetes": "{diabetes}",
  "hypertension": "{hypertension}",
  "discharge_disposition": "{discharge_disposition}",
  "follow_up_scheduled": "{follow_up_scheduled}",
  "medication_adherence_risk": "{medication_adherence_risk}",
  "clinical_note": "{clinical_note}",
  "risk_score": "{risk_score}",
  "risk_level": "{risk_level}"
}
"""


def build_prompt(payload: PatientAssessmentRequest, risk_score: float, risk_level: str) -> str:
    return PROMPT_TEMPLATE.format(
        patient_age=payload.age,
        patient_sex=payload.sex.value,
        prior_admissions_12m=payload.prior_admissions_12m,
        length_of_last_stay=payload.length_of_last_stay,
        comorbidity_count=payload.comorbidity_count,
        diabetes=str(payload.diabetes).lower(),
        hypertension=str(payload.hypertension).lower(),
        discharge_disposition=payload.discharge_disposition.value,
        follow_up_scheduled=str(payload.follow_up_scheduled).lower(),
        medication_adherence_risk=payload.medication_adherence_risk.value,
        clinical_note=(payload.clinical_note or ""),
        risk_score=f"{risk_score:.2f}",
        risk_level=risk_level,
    )


def validate_prompt_output(raw_output: Any) -> PromptLabOutput:
    if isinstance(raw_output, str):
        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Prompt output was not valid JSON: {exc}") from exc
    elif isinstance(raw_output, dict):
        parsed = raw_output
    else:
        raise ValueError("Prompt output must be a JSON string or dict.")

    validated = PromptLabOutput.model_validate(parsed)

    required_keys = {"summary", "key_factors", "recommended_actions"}
    if set(parsed.keys()) != required_keys:
        raise ValueError("Prompt output contains missing or extra keys.")

    if not all(isinstance(x, str) for x in validated.key_factors):
        raise ValueError("key_factors must contain only strings.")

    if not all(isinstance(x, str) for x in validated.recommended_actions):
        raise ValueError("recommended_actions must contain only strings.")

    return validated


def mock_call_prompt_lab(
    prompt_text: str,
    payload: PatientAssessmentRequest,
    risk_score: float,
    risk_level: str,
) -> str:
    _ = prompt_text

    factors: list[str] = []
    actions: list[str] = []

    if payload.prior_admissions_12m > 0:
        factors.append(f"{payload.prior_admissions_12m} prior admission(s) in the last 12 months")
    if payload.comorbidity_count >= 3:
        factors.append("Elevated comorbidity burden")
    if not payload.follow_up_scheduled:
        factors.append("No follow-up appointment scheduled")
    if payload.medication_adherence_risk == MedicationAdherenceRiskEnum.high:
        factors.append("High medication adherence risk")

    while len(factors) < 3:
        factors.append("Clinical history indicates ongoing readmission risk")

    if not payload.follow_up_scheduled:
        actions.append("Schedule follow-up within 7 days")
    if payload.medication_adherence_risk != MedicationAdherenceRiskEnum.low:
        actions.append("Review medications and adherence plan")
    actions.append("Coordinate discharge and outpatient care review")

    while len(actions) < 3:
        actions.append("Monitor symptoms and reinforce care plan")

    summary = (
        f"The patient is assessed as {risk_level} risk for readmission "
        f"with a model score of {risk_score:.2f}. "
        f"The strongest contributors are prior utilization, overall clinical burden, and follow-up readiness."
    )

    return json.dumps(
        {
            "summary": summary,
            "key_factors": factors[:3],
            "recommended_actions": actions[:3],
        }
    )