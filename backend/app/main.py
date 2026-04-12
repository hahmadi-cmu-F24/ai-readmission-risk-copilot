from __future__ import annotations

import json
import os
import uuid
from enum import Enum
from typing import Any, List, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

load_dotenv()

IAM_TOKEN_URL = "https://iam.cloud.ibm.com/identity/token"
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")
WATSONX_DEPLOYMENT_URL = os.getenv("WATSONX_DEPLOYMENT_URL")

app = FastAPI(
    title="Heart Failure Readmission Copilot API",
    version="0.1.0",
)


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

Patient data:
patient_age: {patient_age}
patient_sex: {patient_sex}
prior_admissions_12m: {prior_admissions_12m}
length_of_last_stay: {length_of_last_stay}
comorbidity_count: {comorbidity_count}
diabetes: {diabetes}
hypertension: {hypertension}
discharge_disposition: {discharge_disposition}
follow_up_scheduled: {follow_up_scheduled}
medication_adherence_risk: {medication_adherence_risk}
clinical_note: {clinical_note}
risk_score: {risk_score}
risk_level: {risk_level}
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
        risk_score=f"{risk_score:.4f}",
        risk_level=risk_level,
    )


def validate_prompt_output(raw_output: Any) -> PromptLabOutput:
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

    validated = PromptLabOutput.model_validate(parsed)

    required_keys = {"summary", "key_factors", "recommended_actions"}
    if set(parsed.keys()) != required_keys:
        raise ValueError("Prompt output contains missing or extra keys.")

    if not all(isinstance(x, str) for x in validated.key_factors):
        raise ValueError("key_factors must contain only strings.")

    if not all(isinstance(x, str) for x in validated.recommended_actions):
        raise ValueError("recommended_actions must contain only strings.")

    return validated


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


async def get_access_token() -> str:
    if not WATSONX_API_KEY:
        raise ValueError("WATSONX_API_KEY is not set.")

    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
        "apikey": WATSONX_API_KEY,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(IAM_TOKEN_URL, headers=headers, data=data)
        response.raise_for_status()
        token_data = response.json()

    access_token = token_data.get("access_token")
    if not access_token:
        raise ValueError(f"No access token returned from IAM: {token_data}")
    return access_token


def build_scoring_payload(payload: PatientAssessmentRequest) -> dict[str, Any]:
    # Match the Watsonx model feature order shown on the model details screen.
    fields = [
        "age",
        "comorbidity_count",
        "diabetes",
        "discharge_disposition",
        "follow_up_scheduled",
        "hypertension",
        "length_of_last_stay",
        "medication_adherence_risk",
        "prior_admissions_12m",
        "sex",
    ]

    values = [[
        payload.age,
        payload.comorbidity_count,
        payload.diabetes,
        payload.discharge_disposition.value,
        payload.follow_up_scheduled,
        payload.hypertension,
        payload.length_of_last_stay,
        payload.medication_adherence_risk.value,
        payload.prior_admissions_12m,
        payload.sex.value,
    ]]

    return {
        "input_data": [
            {
                "fields": fields,
                "values": values,
            }
        ]
    }


async def call_autoai_model(payload: PatientAssessmentRequest) -> dict[str, Any]:
    if not WATSONX_DEPLOYMENT_URL:
        raise ValueError("WATSONX_DEPLOYMENT_URL is not set.")

    access_token = await get_access_token()
    scoring_payload = build_scoring_payload(payload)

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            WATSONX_DEPLOYMENT_URL,
            headers=headers,
            json=scoring_payload,
        )
        response.raise_for_status()
        result = response.json()

    predictions = result.get("predictions", [])
    if not predictions:
        raise ValueError(f"No predictions returned from Watsonx deployment: {result}")

    pred_block = predictions[0]
    values = pred_block.get("values", [])
    if not values or not values[0]:
        raise ValueError(f"Unexpected Watsonx response shape: {result}")

    row = values[0]

    probability = None
    predicted_label = None

    for item in row:
        if isinstance(item, (int, float)) and 0.0 <= float(item) <= 1.0:
            probability = float(item)
        if item in (0, 1, "0", "1"):
            predicted_label = int(item)

    if probability is None and predicted_label is not None:
        probability = float(predicted_label)

    if probability is None:
        raise ValueError(f"Could not parse probability from Watsonx response: {result}")

    risk_level = map_score_to_risk_level(probability)

    return {
        "risk_score": round(probability, 4),
        "risk_level": risk_level.value,
    }


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


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Heart Failure Readmission Copilot API is running."}


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
async def assess(payload: PatientAssessmentRequest) -> PatientAssessmentResponse:
    try:
        autoai_result = await call_autoai_model(payload)
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

    except httpx.HTTPStatusError as exc:
        error = make_error_response(
            code="WATSONX_HTTP_ERROR",
            message="Watsonx request failed.",
            details=[{"field": "watsonx", "issue": exc.response.text}],
        )
        raise HTTPException(status_code=502, detail=error.model_dump()["error"])

    except ValueError as exc:
        error = make_error_response(
            code="ASSESSMENT_ERROR",
            message="Assessment failed.",
            details=[{"field": "assessment", "issue": str(exc)}],
        )
        raise HTTPException(status_code=502, detail=error.model_dump()["error"])

    except Exception as exc:
        error = make_error_response(
            code="ASSESSMENT_ERROR",
            message="Failed to generate assessment.",
            details=[{"field": "server", "issue": str(exc)}],
        )
        raise HTTPException(status_code=502, detail=error.model_dump()["error"])