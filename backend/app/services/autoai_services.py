from __future__ import annotations

import os
from typing import Any

import httpx
from dotenv import load_dotenv

from app.schemas import PatientAssessmentRequest, RiskLevelEnum

load_dotenv()

IAM_TOKEN_URL = "https://iam.cloud.ibm.com/identity/token"
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")
WATSONX_DEPLOYMENT_URL = os.getenv("WATSONX_DEPLOYMENT_URL")


def map_probability_to_risk_level(probability: float) -> RiskLevelEnum:
    if probability < 0.34:
        return RiskLevelEnum.low
    if probability < 0.67:
        return RiskLevelEnum.medium
    return RiskLevelEnum.high


def _to_model_payload(payload: PatientAssessmentRequest) -> dict[str, Any]:
    # Match the model schema shown in Watsonx
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

    return {
        "input_data": [
            {
                "fields": fields,
                "values": values,
            }
        ]
    }


async def _get_access_token() -> str:
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
        raise ValueError("No access token returned from IAM.")
    return access_token


async def call_autoai_model(payload: PatientAssessmentRequest) -> dict[str, Any]:
    if not WATSONX_DEPLOYMENT_URL:
        raise ValueError("WATSONX_DEPLOYMENT_URL is not set.")

    access_token = await _get_access_token()
    scoring_payload = _to_model_payload(payload)

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

    # Expected Watsonx response shape
    predictions = result.get("predictions", [])
    if not predictions:
        raise ValueError("No predictions returned from Watsonx deployment.")

    pred = predictions[0]

    # Try probability first if available
    probability = None
    if "predictions" in pred and pred["predictions"]:
        # Some deployments return [{'values': [...], 'predictions': [...]}]-like shapes
        first_prediction = pred["predictions"][0]
        if isinstance(first_prediction, dict):
            probability = (
                first_prediction.get("probability")
                or first_prediction.get("score")
            )

    # Fallback: inspect values
    if probability is None:
        values = pred.get("values", [])
        if values and values[0]:
            row = values[0]
            # Use last numeric item as a fallback
            numeric_items = [x for x in row if isinstance(x, (int, float))]
            if numeric_items:
                probability = float(numeric_items[-1])

    if probability is None:
        raise ValueError(f"Unable to parse prediction response: {result}")

    risk_level = map_probability_to_risk_level(float(probability))

    return {
        "risk_score": round(float(probability), 4),
        "risk_level": risk_level.value,
    }