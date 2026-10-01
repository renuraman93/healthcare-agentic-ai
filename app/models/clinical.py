"""
Pydantic models for structured clinical data extraction.
"""

from pydantic import BaseModel, Field


class Medication(BaseModel):
    name: str
    dose: str | None = None
    frequency: str | None = None


class Vitals(BaseModel):
    blood_pressure: str | None = None
    heart_rate: str | None = None
    temperature: str | None = None
    weight: str | None = None
    oxygen_saturation: str | None = None


class ClinicalExtraction(BaseModel):
    conditions: list[str] = Field(default_factory=list)
    medications: list[Medication] = Field(default_factory=list)
    vitals: Vitals = Field(default_factory=Vitals)
    allergies: list[str] = Field(default_factory=list)