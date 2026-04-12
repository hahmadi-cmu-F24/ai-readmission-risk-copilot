from __future__ import annotations

import json
import uuid
from enum import Enum
from typing import Any, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict, field_validator


app = FastAPI(
    title="Heart Failure Readmission Copilot API",
    version="0.1.0",
)


# ----------------------------
# Enums
# ----------------------------

class SexEnum(str, Enum):
    female = "female"
    male = "male"
    other = "other"
    unknown = "unknown"


class DischargeDispositionEnum(str, Enum):
    home = "home"
    skilled_nursing = "skilled_nursing"
    rehab = "rehab"
    home_health = "home_health"
    other = "other"


class MedicationAdherenceRiskEnum(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class RiskLevelEnum(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


# ----------------------------
# Request / response models
# ----------------------------

class PatientAssessmentRequest(BaseModel):
    age: int = Field(..., ge=0, le=120)
    sex: SexEnum
    prior_admissions_12m: int = Field(..., ge=0, le=50)
    length_of_last_stay: int = Field(..., ge=0, le=365)
    comorbidity_count: int = Field(..., ge=0, le=50)
    diabetes: bool
    hypertension: bool
    discharge_disposition: DischargeDispositionEnum
    follow_up_scheduled: bool
    medication_adherence_risk: MedicationAdherenceRiskEnum
    clinical_note: Optional[str] = Field(default=None, max_length=2000)

    model_config = ConfigDict(extra="forbid")


class PromptLabOutput(BaseModel):
    summary: str = Field(..., min_length=1, max_length=1000)
    key_factors: List[str] = Field(..., min_length=1, max_length=3)
    recommended_actions: List[str] = Field(..., min_length=1, max_length=3)

    model_config = ConfigDict(extra="forbid")

    @field_validator("key_factors", "recommended_actions")
    @classmethod
    def validate_string_list(cls, value: List[str]) -> List[str]:
        if not all(isinstance(item, str) and item.strip() for item in value):
            raise ValueError("All items must be non-empty strings.")
        return value


class PatientAssessmentResponse(BaseModel):
    assessment_id: str
    risk_score: float = Field(..., ge=0.0, le=1.0)
    risk_level: RiskLevelEnum
    summary: str
    key_factors: List[str]
    recommended_actions: List[str]


class ErrorDetail(BaseModel):
    field: str
    issue: str


class ErrorBody(BaseModel):
    code: str
    message: str
    details: List[ErrorDetail] = []


class ErrorResponse(BaseModel):
    error: ErrorBody


# ----------------------------
# Prompt template with guardrails
# ----------------------------

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


# ----------------------------
# Helpers
# ----------------------------

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
    """
    Accepts either a dict or a JSON string and validates strict structure.
    Raises ValueError on invalid output.
    """
    parsed: Any

    if isinstance(raw_output, str):
        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Prompt output was not valid JSON: {exc}") from exc
    elif isinstance(raw_output, dict):
        parsed = raw_output
    else:
        raise ValueError("Prompt output must be a JSON string or dict.")

    try:
        return PromptLabOutput.model_validate(parsed)
    except Exception as exc:
        raise ValueError(f"Prompt output failed schema validation: {exc}") from exc


def map_score_to_risk_level(score: float) -> RiskLevelEnum:
    if score < 0.34:
        return RiskLevelEnum.low
    if score < 0.67:
        return RiskLevelEnum.medium
    return RiskLevelEnum.high


def make_error_response(code: str, message: str, details: List[dict[str, str]]) -> ErrorResponse:
    return ErrorResponse(
        error=ErrorBody(
            code=code,
            message=message,
            details=[ErrorDetail(**item) for item in details],
        )
    )


# ----------------------------
# Mocked Watsonx integrations
# ----------------------------

def mock_call_autoai(payload: PatientAssessmentRequest) -> dict[str, Any]:
    """
    Mocked tabular prediction.
    Replace this with your real watsonx AutoAI deployment call.
    """
    score = 0.10

    score += min(payload.prior_admissions_12m * 0.15, 0.45)
    score += min(payload.comorbidity_count * 0.05, 0.20)
    score += 0.10 if not payload.follow_up_scheduled else 0.0
    score += 0.08 if payload.medication_adherence_risk == MedicationAdherenceRiskEnum.high else 0.03 if payload.medication_adherence_risk == MedicationAdherenceRiskEnum.medium else 0.0
    score += 0.05 if payload.diabetes else 0.0
    score += 0.05 if payload.hypertension else 0.0

    score = max(0.0, min(score, 0.99))
    risk_level = map_score_to_risk_level(score)

    return {
        "risk_score": round(score, 2),
        "risk_level": risk_level.value,
    }


def mock_call_prompt_lab(prompt_text: str, payload: PatientAssessmentRequest, risk_score: float, risk_level: str) -> str:
    """
    Mocked LLM output.
    Replace this with your real Prompt Lab / deployed prompt call.
    """
    _ = prompt_text  # kept so the call shape matches real usage later

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

    return json.dumps({
        "summary": summary,
        "key_factors": factors[:3],
        "recommended_actions": actions[:3],
    })


# ----------------------------
# Routes
# ----------------------------

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/assess",
    response_model=PatientAssessmentResponse,
    responses={
        400: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
    },
)
def assess(payload: PatientAssessmentRequest) -> PatientAssessmentResponse:
    try:
        autoai_result = mock_call_autoai(payload)
        risk_score = float(autoai_result["risk_score"])
        risk_level = str(autoai_result["risk_level"])

        prompt_text = build_prompt(payload, risk_score, risk_level)
        raw_prompt_output = mock_call_prompt_lab(prompt_text, payload, risk_score, risk_level)
        prompt_output = validate_prompt_output(raw_prompt_output)

        return PatientAssessmentResponse(
            assessment_id=str(uuid.uuid4()),
            risk_score=risk_score,
            risk_level=risk_level,
            summary=prompt_output.summary,
            key_factors=prompt_output.key_factors,
            recommended_actions=prompt_output.recommended_actions,
        )

    except ValueError as exc:
        error = make_error_response(
            code="PROMPT_OUTPUT_ERROR",
            message="Prompt output was invalid.",
            details=[{"field": "prompt_output", "issue": str(exc)}],
        )
        raise HTTPException(status_code=502, detail=error.model_dump()["error"])

    except Exception as exc:
        error = make_error_response(
            code="ASSESSMENT_ERROR",
            message="Failed to generate assessment.",
            details=[{"field": "server", "issue": str(exc)}],
        )
        raise HTTPException(status_code=502, detail=error.model_dump()["error"])