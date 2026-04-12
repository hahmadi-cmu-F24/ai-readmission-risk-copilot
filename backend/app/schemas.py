from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


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