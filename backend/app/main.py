from __future__ import annotations

import uuid

from fastapi import FastAPI, HTTPException

from app.schemas import (
    ErrorBody,
    ErrorDetail,
    ErrorResponse,
    PatientAssessmentRequest,
    PatientAssessmentResponse,
)
from app.services.prompt_services import build_prompt, mock_call_prompt_lab, validate_prompt_output
from app.services.autoai_services import call_autoai_model

app = FastAPI(
    title="Heart Failure Readmission Copilot API",
    version="0.1.0",
)


def make_error_response(code: str, message: str, details: list[dict[str, str]]) -> ErrorResponse:
    return ErrorResponse(
        error=ErrorBody(
            code=code,
            message=message,
            details=[ErrorDetail(**item) for item in details],
        )
    )


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